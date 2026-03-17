// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "mori/collective/allreduce/twoshot_allreduce_sdma_class.hpp"
#include "mori/collective/allreduce/twoshot_sdma_kernel.hpp"
#include "mori/collective/allreduce/twoshot_sdma_async_kernel.hpp"
#include "mori/shmem/shmem.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <mpi.h>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <type_traits>

namespace mori {
namespace collective {

// ---------------------------------------------------------------------------
// Delegating constructor
// ---------------------------------------------------------------------------
template <typename T>
AllreduceSdma<T>::AllreduceSdma(int myPe, int npes, size_t transit_buffer_size,
                                bool copy_output_to_user, bool /*use_graph_mode*/)
    : AllreduceSdma(myPe, npes, 0, transit_buffer_size,
                    copy_output_to_user, false) {
}

// ---------------------------------------------------------------------------
// Main constructor
// ---------------------------------------------------------------------------
template <typename T>
AllreduceSdma<T>::AllreduceSdma(int myPe, int npes,
                                size_t /*input_buffer_size*/,
                                size_t output_buffer_size,
                                bool copy_output_to_user,
                                bool /*use_graph_mode*/)
    : myPe_(myPe),
      npes_(npes),
      dtype_size_(sizeof(T)),
      max_blocks_(getDeviceMaxBlocks()),
      flags_(nullptr, ShmemDeleter()),
      barrierPtr_(nullptr),
      barrierMem_(nullptr, ShmemDeleter()),
      input_transit_buffer_(nullptr),
      input_transit_buffer_size_(0),
      input_transit_buffer_ptr_(nullptr, ShmemDeleter()),
      output_transit_buffer_(nullptr),
      output_transit_buffer_size_(output_buffer_size),
      output_transit_buffer_ptr_(nullptr, ShmemDeleter()),
      async_in_progress_(false),
      async_input_(nullptr),
      async_output_(nullptr),
      async_total_count_(0),
      async_stream_(nullptr),
      async_start_time_(0.0),
      copy_output_to_user_(copy_output_to_user),
      launch_ready_epoch_(0),
      my_last_completed_sequence_(-1) {

    // 1. Allocate SDMA completion flags
    size_t flagsSize = npes_ * sizeof(uint64_t);
    void* flags = shmem::ShmemMalloc(flagsSize);
    if (!flags) throw std::runtime_error("Failed to allocate flags memory");
    flags_.reset(static_cast<uint64_t*>(flags));
    memset(flags_.get(), 0, flagsSize);
    flagsObj_ = shmem::ShmemQueryMemObjPtr(flags_.get());
    if (!flagsObj_.IsValid())
        throw std::runtime_error("Failed to get valid flags memory object");

    // 2. Allocate CrossPeBarrier (device-scope broadcast flag, ~128 bytes)
    size_t barrierSize = sizeof(CrossPeBarrier);
    void* bMem = shmem::ShmemMalloc(barrierSize);
    if (!bMem) throw std::runtime_error("Failed to allocate barrier memory");
    barrierMem_.reset(bMem);
    barrierPtr_ = reinterpret_cast<CrossPeBarrier*>(bMem);
    hipError_t me = hipMemset(bMem, 0, barrierSize);
    if (me != hipSuccess)
        throw std::runtime_error("Failed to zero-init barrier memory");

    // 2.5. Allocate launch_ready (npes uint32_t) for GPU-side launch sync when stream!=nullptr
    size_t launchReadySize = npes_ * sizeof(uint32_t);
    void* lr = shmem::ShmemMalloc(launchReadySize);
    if (!lr) throw std::runtime_error("Failed to allocate launch_ready memory");
    launch_ready_.reset(static_cast<uint32_t*>(lr));
    memset(launch_ready_.get(), 0, launchReadySize);
    launch_ready_obj_ = shmem::ShmemQueryMemObjPtr(launch_ready_.get());
    if (!launch_ready_obj_.IsValid())
        throw std::runtime_error("Failed to get valid launch_ready memory object");

    // 3. Allocate output transit buffer (gather + reduce + allgather)
    output_transit_buffer_ = shmem::ShmemMalloc(output_transit_buffer_size_);
    if (!output_transit_buffer_)
        throw std::runtime_error("Failed to allocate output transit buffer");
    output_transit_buffer_ptr_.reset(output_transit_buffer_);

    output_transit_buffer_obj_ =
        shmem::ShmemSymmetricRegister(output_transit_buffer_,
                                      output_transit_buffer_size_);
    if (!output_transit_buffer_obj_.IsValid())
        throw std::runtime_error("Failed to register output transit buffer");

    printf("AllreduceSdma(SDMA) initialized: PE %d of %d, max_blocks=%d\n",
           myPe_, npes_, max_blocks_);
    printf("  Flags: %zu bytes at %p\n", flagsSize, flags_.get());
    printf("  Barrier: %zu bytes at %p\n", barrierSize, bMem);
    printf("  Output transit buffer: %.2f MB at %p\n",
           output_transit_buffer_size_ / (1024.0 * 1024.0),
           output_transit_buffer_);
}

// ---------------------------------------------------------------------------
template <typename T>
AllreduceSdma<T>::~AllreduceSdma() {
    if (async_in_progress_) {
        cancel_async();
    }
    if (flags_) {
        printf("AllreduceSdma destroyed: PE %d\n", myPe_);
    }
}

// ---------------------------------------------------------------------------
template <typename T>
bool AllreduceSdma<T>::ensure_buffer_size(void*& buffer,
                                         std::unique_ptr<void, ShmemDeleter>& buffer_ptr,
                                         size_t& current_size,
                                         application::SymmMemObjPtr& buffer_obj,
                                         size_t required_size,
                                         const char* buffer_name) {
    if (required_size <= current_size) {
        return true;
    }

    // If buffer is not large enough, reallocate
    printf("PE %d: %s too small: required %.2f MB, current %.2f MB\n",
           myPe_, buffer_name,
           required_size / (1024.0 * 1024.0),
           current_size / (1024.0 * 1024.0));

    // First release the old one
    buffer_ptr.reset();

    // Allocate new one
    current_size = required_size;
    buffer = shmem::ShmemMalloc(current_size);
    if (buffer == nullptr) {
        fprintf(stderr, "PE %d: Failed to reallocate %s of size %.2f MB\n",
                myPe_, buffer_name, current_size / (1024.0 * 1024.0));
        return false;
    }
    buffer_ptr.reset(buffer);

    // Re-register
    buffer_obj = shmem::ShmemSymmetricRegister(buffer, current_size);
    if (!buffer_obj.IsValid()) {
        fprintf(stderr, "PE %d: Failed to re-register %s\n", myPe_, buffer_name);
        return false;
    }

    printf("PE %d: %s reallocated to %.2f MB\n",
           myPe_, buffer_name, current_size / (1024.0 * 1024.0));
    return true;
}

// copy_input_to_transit implementation
template <typename T>
void AllreduceSdma<T>::copy_input_to_transit(T* input, size_t total_count, hipStream_t stream) {
    size_t input_bytes = total_count * dtype_size_;

    // Verify pointer validity
    if (input == nullptr) {
        fprintf(stderr, "PE %d: Input pointer is null\n", myPe_);
        throw std::runtime_error("Input pointer is null");
    }

    if (input_transit_buffer_ == nullptr) {
        fprintf(stderr, "PE %d: Input transit buffer is null\n", myPe_);
        throw std::runtime_error("Input transit buffer is null");
    }

    // Copy from user input buffer to input transit buffer
    // No explicit sync needed — same-stream operations are ordered by the GPU
    hipError_t err = hipSuccess;
    if (stream != nullptr) {
        err = hipMemcpyAsync(input_transit_buffer_, input, input_bytes,
                           hipMemcpyDeviceToDevice, stream);
    } else {
        err = hipMemcpy(input_transit_buffer_, input, input_bytes,
                       hipMemcpyDeviceToDevice);
    }

    if (err != hipSuccess) {
        fprintf(stderr, "PE %d: Failed to copy input to transit buffer: %s\n",
                myPe_, hipGetErrorString(err));
        throw std::runtime_error("Input copy failed");
    }
}

// copy_output_to_user implementation
// For AllReduce: output is total_count elements (same size as input, NOT npes * total_count)
// For large buffers (>256MB): chunked hipMemcpyAsync to avoid ROCm driver hang on single 1GB+ async copy
constexpr size_t COPY_CHUNK_BYTES = 256 * 1024 * 1024;

template <typename T>
void AllreduceSdma<T>::copy_output_to_user(T* output, size_t total_count, hipStream_t stream) {
    size_t bytes = total_count * dtype_size_;
    if (!output)  throw std::runtime_error("Output pointer is null");
    if (!output_transit_buffer_)
        throw std::runtime_error("Output transit buffer is null");
    if (bytes > output_transit_buffer_size_) {
        fprintf(stderr, "PE %d: copy_output_to_user: bytes %zu > buffer size %zu\n",
                myPe_, bytes, output_transit_buffer_size_);
        throw std::runtime_error("Output copy failed: buffer too small");
    }
    if (bytes == 0) return;

    if (stream == nullptr) {
        hipError_t err = hipMemcpy(output, output_transit_buffer_, bytes, hipMemcpyDeviceToDevice);
        if (err != hipSuccess) {
            fprintf(stderr, "PE %d: copy_output_to_user failed: %s\n",
                    myPe_, hipGetErrorString(err));
            throw std::runtime_error("Output copy failed");
        }
        return;
    }

    // Async path: chunk large copies to avoid ROCm driver hang on 1GB+ single hipMemcpyAsync
    size_t offset = 0;
    while (offset < bytes) {
        size_t chunk = std::min(bytes - offset, COPY_CHUNK_BYTES);
        hipError_t err = hipMemcpyAsync(
            reinterpret_cast<char*>(output) + offset,
            static_cast<const char*>(output_transit_buffer_) + offset,
            chunk, hipMemcpyDeviceToDevice, stream);
        if (err != hipSuccess) {
            if (sizeof(T) == 4 && err == hipErrorInvalidValue) {
                err = hipMemcpy(reinterpret_cast<char*>(output) + offset,
                                static_cast<const char*>(output_transit_buffer_) + offset,
                                chunk, hipMemcpyDeviceToDevice);
            }
            if (err != hipSuccess) {
                fprintf(stderr, "PE %d: copy_output_to_user failed at offset %zu: %s\n",
                        myPe_, offset, hipGetErrorString(err));
                throw std::runtime_error("Output copy failed");
            }
        }
        offset += chunk;
    }
}

// ---------------------------------------------------------------------------
// operator()
// ---------------------------------------------------------------------------
template <typename T>
bool AllreduceSdma<T>::operator()(T* input, T* output, size_t total_count, hipStream_t stream) {
    try {
        size_t required_output_size = total_count * dtype_size_;
        if (!ensure_buffer_size(output_transit_buffer_, output_transit_buffer_ptr_,
                                output_transit_buffer_size_, output_transit_buffer_obj_,
                                required_output_size, "output transit buffer")) {
            return false;
        }

        // Step 0: Stream-ordered reset of SDMA flags.
        // For async overlap, host memset may race with previous in-flight collective.
        // Use same stream ordering to guarantee "reset happens after previous kernels".
        hipError_t err = hipSuccess;
        if (stream != nullptr) {
            err = hipMemsetAsync(flags_.get(), 0, npes_ * sizeof(uint64_t), stream);
            if (err != hipSuccess) {
                fprintf(stderr, "PE %d: hipMemsetAsync(flags) failed: %s\n",
                        myPe_, hipGetErrorString(err));
                return false;
            }
        } else {
            memset(flags_.get(), 0, npes_ * sizeof(uint64_t));
        }

        // Step 0.5: GPU-side launch sync when stream!=nullptr (NCCL-style, no CPU barrier).
        // Ensures all PEs have launched before any SdmaReduceScatter proceeds.
        if (stream != nullptr && launch_ready_ && launch_ready_obj_.IsValid()) {
            const uint32_t launch_epoch = launch_ready_epoch_.fetch_add(1, std::memory_order_relaxed) + 1;
            LaunchSyncKernel<<<1, 1, 0, stream>>>(myPe_, npes_, launch_ready_obj_, launch_epoch);
            err = hipGetLastError();
            if (err != hipSuccess) {
                fprintf(stderr, "PE %d: LaunchSyncKernel failed: %s\n",
                        myPe_, hipGetErrorString(err));
                return false;
            }
        }

        // Step 1: SdmaReduceScatter — SDMA scatter + local reduce
        constexpr int pack_size = packed_t<T>::P::size;
        int threads = 512;
        int packedPerRank = static_cast<int>(
            ((total_count / npes_ + pack_size - 1) / pack_size));
        int blocks = std::min(max_blocks_,
                              (packedPerRank + threads - 1) / threads);
        if (blocks < 1) blocks = 1;

        SdmaReduceScatterKernel<T><<<blocks, threads, 0, stream>>>(
            myPe_, npes_,
            input,
            output_transit_buffer_obj_,
            flagsObj_,
            barrierPtr_,
            total_count);

        err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "PE %d: SdmaReduceScatter launch failed: %s\n",
                    myPe_, hipGetErrorString(err));
            return false;
        }

        // Step 1.5: Flush L2 to HBM so AllGather SDMA reads fresh data.
        // Must run after ReduceScatter completes (all blocks done) to avoid
        // race where early blocks flush before late blocks finish (large msg bug).
        FlushL2Kernel<<<1, 1, 0, stream>>>();

        // Step 2: AllGather via SDMA
        AllGatherSdmaKernel<T><<<1, 512, 0, stream>>>(
            myPe_, npes_,
            output_transit_buffer_obj_,
            flagsObj_,
            barrierPtr_,
            total_count);

        err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "PE %d: AllGather launch failed: %s\n",
                    myPe_, hipGetErrorString(err));
            return false;
        }

        // Step 3: Copy result to user buffer
        if (copy_output_to_user_) {
            // NCCL-style overlap: when stream!=nullptr, skip sync so C++ returns immediately after queueing.
            // MORI_SDMA_DEFER_SYNC=1 时跳过，实现 overlap；默认保留 sync 保证稳定性。
            const char* defer_sync = std::getenv("MORI_SDMA_DEFER_SYNC");
            const bool skip_sync = (defer_sync && std::strchr(defer_sync, '1'));
            if (!skip_sync && sizeof(T) == 4 && stream != nullptr) {
                hipError_t sync_err = hipStreamSynchronize(stream);
                if (sync_err != hipSuccess) {
                    fprintf(stderr, "PE %d: hipStreamSynchronize before copy failed: %s\n",
                            myPe_, hipGetErrorString(sync_err));
                    return false;
                }
            }
            copy_output_to_user(output, total_count, stream);
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "PE %d: AllReduce failed: %s\n", myPe_, e.what());
        return false;
    }
    return true;
}

// ================ Async API Implementations ================

template <typename T>
bool AllreduceSdma<T>::start_async(T* input, T* output, size_t total_count, hipStream_t stream) {
    bool expected = false;
    if (!async_in_progress_.compare_exchange_strong(expected, true)) {
        printf("PE %d: Another async operation is already in progress\n", myPe_);
        return false;
    }

    async_input_ = input;
    async_output_ = output;
    async_total_count_ = total_count;
    async_stream_ = stream;
    async_start_time_ = MPI_Wtime();

    try {
        size_t elementCountPerRank = total_count / npes_;
        size_t required_output_size = elementCountPerRank * npes_ * dtype_size_;
        if (!ensure_buffer_size(output_transit_buffer_, output_transit_buffer_ptr_,
                                output_transit_buffer_size_, output_transit_buffer_obj_,
                                required_output_size, "output transit buffer")) {
            async_in_progress_ = false;
            return false;
        }

        // Step 1: SdmaReduceScatter — same as operator()
        constexpr int pack_size = packed_t<T>::P::size;
        int threads = 512;
        int packedPerRank = static_cast<int>(
            ((total_count / npes_ + pack_size - 1) / pack_size));
        int blocks = std::min(max_blocks_,
                              (packedPerRank + threads - 1) / threads);
        if (blocks < 1) blocks = 1;

        SdmaReduceScatterKernel<T><<<blocks, threads, 0, stream>>>(
            myPe_, npes_,
            input,
            output_transit_buffer_obj_,
            flagsObj_,
            barrierPtr_,
            total_count);

        // Step 1.5: Flush L2 to HBM so AllGather SDMA reads fresh data (same as sync path).
        FlushL2Kernel<<<1, 1, 0, stream>>>();

        // Step 2: AllGather PUT only — sends data, returns immediately
        // The wait is deferred to wait_async so the user can run GEMM on CU
        AllGatherAsyncPutKernel<T><<<1, 512, 0, stream>>>(
            myPe_, npes_,
            output_transit_buffer_obj_,
            flagsObj_,
            barrierPtr_,
            total_count);

        hipError_t kernel_err = hipGetLastError();
        if (kernel_err != hipSuccess) {
            fprintf(stderr, "PE %d: Async kernel launch failed: %s\n",
                    myPe_, hipGetErrorString(kernel_err));
            throw std::runtime_error("Kernel launch failed");
        }

        return true;

    } catch (const std::exception& e) {
        fprintf(stderr, "PE %d: Failed to start async operation: %s\n", myPe_, e.what());
        async_in_progress_ = false;
        return false;
    }
}

template <typename T>
double AllreduceSdma<T>::wait_async(hipStream_t stream) {
    if (!async_in_progress_) {
        printf("PE %d: No async operation in progress\n", myPe_);
        return -1.0;
    }

    try {
        hipStream_t wait_stream = (stream != nullptr) ? stream : async_stream_;

        // Wait for AllGather SDMA transfers to complete
        // Remote PEs wrote to our local signalPtrs via SDMA ATOMIC
        AllGatherAsyncWaitKernel<<<1, 64, 0, wait_stream>>>(
            myPe_, npes_, output_transit_buffer_obj_, barrierPtr_, async_total_count_);

        // Copy result to user buffer (if enabled)
        if (copy_output_to_user_) {
            // 4-byte types (float): sync before copy, same as sync operator().
            // Without this, copy mode fails (e.g. second half wrong) while non-copy passes.
            if (sizeof(T) == 4 && wait_stream != nullptr) {
                hipError_t sync_err = hipStreamSynchronize(wait_stream);
                if (sync_err != hipSuccess) {
                    fprintf(stderr, "PE %d: hipStreamSynchronize before copy failed: %s\n",
                            myPe_, hipGetErrorString(sync_err));
                    throw std::runtime_error("Stream sync before copy failed");
                }
            }
            copy_output_to_user(async_output_, async_total_count_, wait_stream);
        }

        // Single synchronization at the end
        if (wait_stream != nullptr) {
            hipError_t err = hipStreamSynchronize(wait_stream);
            if (err != hipSuccess) {
                fprintf(stderr, "PE %d: Stream synchronization failed: %s\n",
                        myPe_, hipGetErrorString(err));
                throw std::runtime_error("Stream synchronization failed");
            }
        } else {
            hipError_t err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                fprintf(stderr, "PE %d: Device synchronization failed: %s\n",
                        myPe_, hipGetErrorString(err));
                throw std::runtime_error("Device synchronization failed");
            }
        }

        double end_time = MPI_Wtime();
        double duration = end_time - async_start_time_;

        async_in_progress_ = false;
        async_input_ = nullptr;
        async_output_ = nullptr;
        async_total_count_ = 0;
        async_stream_ = nullptr;
        async_start_time_ = 0.0;

        return duration;

    } catch (const std::exception& e) {
        fprintf(stderr, "PE %d: Async wait failed: %s\n", myPe_, e.what());
        cancel_async();
        return -1.0;
    }
}

template <typename T>
void AllreduceSdma<T>::cancel_async() {
    if (async_in_progress_) {
        printf("PE %d: Cancelling async operation\n", myPe_);
        async_in_progress_ = false;
        async_input_ = nullptr;
        async_output_ = nullptr;
        async_total_count_ = 0;
        async_stream_ = nullptr;
        async_start_time_ = 0.0;
    }
}

// ================ END: Async API Implementations ================

// allreduce_inplace implementation
// ---------------------------------------------------------------------------
template <typename T>
bool AllreduceSdma<T>::allreduce_inplace(T* data, size_t total_count,
                                          hipStream_t stream) {
    const char* multi_bucket = std::getenv("MORI_SDMA_MULTI_BUCKET");
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    // Only use MPI-based serialization when MORI_SDMA_MULTI_BUCKET=1 AND MPI was already initialized (e.g. mpirun).
    // With torchrun/other launchers MPI is not init; caller must use Python-side per-bucket sync instead.
    const bool use_sequence = (multi_bucket && std::strchr(multi_bucket, '1') && mpi_initialized);

    if (use_sequence) {
        const int my_sequence = my_last_completed_sequence_ + 1;
        int send_val = my_last_completed_sequence_;
        int recv_val = -1;
        while (true) {
            int mpi_err = MPI_Allreduce(&send_val, &recv_val, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (mpi_err != MPI_SUCCESS) {
                fprintf(stderr, "PE %d: MORI_SDMA_MULTI_BUCKET MPI_Allreduce(MIN) failed: %d\n",
                        myPe_, mpi_err);
                return false;
            }
            if (recv_val >= my_sequence - 1)
                break;
        }
    }

    bool saved = copy_output_to_user_;
    copy_output_to_user_ = true;
    bool ok = (*this)(data, data, total_count, stream);
    copy_output_to_user_ = saved;

    if (use_sequence)
        my_last_completed_sequence_ = my_last_completed_sequence_ + 1;

    if (!ok) return false;
    // Optional: block until stream completes (drop-in compatible with torch.distributed.all_reduce)
    const char* blocking = std::getenv("MORI_SDMA_BLOCKING");
    if (blocking && std::strchr(blocking, '1')) {
        if (stream != nullptr) {
            hipError_t err = hipStreamSynchronize(stream);
            if (err != hipSuccess) {
                fprintf(stderr, "PE %d: MORI_SDMA_BLOCKING hipStreamSynchronize failed: %s\n",
                        myPe_, hipGetErrorString(err));
                return false;
            }
        } else {
            hipError_t err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                fprintf(stderr, "PE %d: MORI_SDMA_BLOCKING hipDeviceSynchronize failed: %s\n",
                        myPe_, hipGetErrorString(err));
                return false;
            }
        }
    }
    return true;
}

template <typename T>
bool AllreduceSdma<T>::allreduce_inplace_avg(T* data, size_t total_count,
                                             int world_size, hipStream_t stream) {
    if (world_size <= 0) {
        fprintf(stderr, "PE %d: invalid world_size=%d in allreduce_inplace_avg\n", myPe_, world_size);
        return false;
    }
    if (!allreduce_inplace(data, total_count, stream)) {
        return false;
    }
    if constexpr (std::is_same_v<T, float>) {
        hipStream_t scale_stream = stream;
        const int threads = 256;
        const int blocks = std::min(max_blocks_, static_cast<int>((total_count + threads - 1) / threads));
        const float inv = 1.0f / static_cast<float>(world_size);
        ScaleFp32Kernel<<<std::max(1, blocks), threads, 0, scale_stream>>>(
            reinterpret_cast<float*>(data), total_count, inv);
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "PE %d: ScaleFp32Kernel launch failed: %s\n",
                    myPe_, hipGetErrorString(err));
            return false;
        }
        return true;
    } else {
        fprintf(stderr, "PE %d: allreduce_inplace_avg currently supports fp32 only\n", myPe_);
        return false;
    }
}

// ---------------------------------------------------------------------------
template <typename T>
void AllreduceSdma<T>::resetFlags() {
    if (flags_) {
        memset(flags_.get(), 0, npes_ * sizeof(uint64_t));
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
template class AllreduceSdma<uint32_t>;
template class AllreduceSdma<uint64_t>;
template class AllreduceSdma<int32_t>;
template class AllreduceSdma<int64_t>;
template class AllreduceSdma<float>;
template class AllreduceSdma<double>;
template class AllreduceSdma<half>;
template class AllreduceSdma<__hip_bfloat16>;

} // namespace collective
} // namespace mori
