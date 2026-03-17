#!/usr/bin/env python3
"""
Multi-bucket AllReduce Async test: 模拟 DDP allreduce_hook 的 async 调用场景。

- 使用 start_async / wait_async 接口（非 allreduce_inplace）
- 多段不同尺寸的 grad-bucket，共用同一个 AllreduceSdma 实例
- 每 bucket 调用顺序：start_async(input, output, count) → wait_async()
- Per-bucket 同步：进入每个 bucket 前 all_reduce(MIN) 等待「所有 rank 已完成 bucket_index-1」
- 支持 copy / non-copy 模式（copy_output_to_user）

DDP hook 典型流程：
  backward 触发 bucket_i 的 hook → start_async(bucket_i) → 继续 backward（可与 AllGather 重叠）→ wait_async()
"""

import os
import random
import numpy as np
import torch
import torch.distributed as dist
import mori.shmem as shmem
from mori.ccl import AllreduceSdma
from tests.python.utils import TorchDistContext, get_free_port


ELEM_SIZE_FP32 = 4
PACK_SIZE_FP32 = 4
MAX_BUCKET_BYTES = 1024 * 1024 * 1024
MAX_ELEMS_FP32 = MAX_BUCKET_BYTES // ELEM_SIZE_FP32


def _to_numpy(tensor):
    if tensor.dtype in (torch.bfloat16, torch.float16):
        return tensor.float().numpy()
    return tensor.numpy()


def _verify_allreduce_result(cpu_data, numel, my_pe, npes, expected_sum=None, bucket_index=None):
    if expected_sum is None:
        expected_sum = sum((pe + 1) * 1000 for pe in range(npes))
    chunk = cpu_data[:numel].astype(np.float32)
    exp = np.full(numel, expected_sum, dtype=np.float32)
    if np.allclose(chunk, exp, rtol=1e-5, atol=0.01):
        return True
    bad = np.where(~np.isclose(chunk, exp, rtol=1e-5, atol=0.01))[0]
    bid = f" bucket {bucket_index}" if bucket_index is not None else ""
    print(f"  PE {my_pe}{bid}: FAIL numel={numel} expected={expected_sum} "
          f"mismatches={len(bad)} first_bad_idx={bad[0]} got={chunk[bad[0]]}")
    return False


def _run_async_multi_bucket(rank, world_size, port, num_buckets, seed, max_elems_per_bucket,
                            copy_output, fixed_elems=None):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        assert my_pe == rank and npes == world_size

        device = torch.device(f"cuda:{rank}")
        stream = torch.cuda.Stream(device=device)
        dtype = torch.float32
        fill_value = (my_pe + 1) * 1000.0

        align = npes * PACK_SIZE_FP32
        if fixed_elems is not None:
            n = (fixed_elems // align) * align
            if n < align:
                n = align
            bucket_elems = [n] * num_buckets
        else:
            rng = random.Random(seed)
            bucket_elems = []
            max_n = max(1, max_elems_per_bucket)
            for _ in range(num_buckets):
                n_raw = rng.randint(1, max_n)
                n = (n_raw // align) * align
                if n < align:
                    n = align
                bucket_elems.append(n)

        max_elems = max(bucket_elems)
        max_bytes = max_elems * ELEM_SIZE_FP32
        output_buf_size = npes * (max_elems // npes + 64) * ELEM_SIZE_FP32

        ar = AllreduceSdma(
            my_pe, npes,
            input_buffer_size=max_bytes,
            output_buffer_size=output_buf_size,
            copy_output_to_user=copy_output,
            dtype=dtype,
        )

        inputs = []
        outputs = []
        for n in bucket_elems:
            inputs.append(torch.full((n,), fill_value, dtype=dtype, device=device))
            outputs.append(torch.zeros(n, dtype=dtype, device=device))

        last_done = torch.zeros(1, dtype=torch.long, device=device)
        my_last_completed = -1
        expected_sum = sum((pe + 1) * 1000 for pe in range(npes))
        all_ok = True

        torch.cuda.synchronize()
        dist.barrier()

        for bucket_index in range(num_buckets):
            need_wait = bucket_index - 1
            while need_wait >= 0:
                last_done.zero_()
                last_done[0] = my_last_completed
                dist.all_reduce(last_done, op=dist.ReduceOp.MIN)
                if last_done.item() >= need_wait:
                    break

            ar.reset_flags()

            inp = inputs[bucket_index]
            out = outputs[bucket_index]
            inp.fill_(fill_value)
            n_elems = inp.numel()

            ok = ar.start_async(inp, out, n_elems, stream=stream)
            if not ok:
                raise RuntimeError(f"PE {rank} bucket {bucket_index}: start_async returned False")

            ar.wait_async(stream=stream)
            stream.synchronize()

            # 立即校验：non-copy 时 transit 会被下一 bucket 覆盖，必须在此处读取
            if copy_output:
                verify_data = _to_numpy(out.cpu())
            else:
                transit = ar.get_output_transit_buffer(device=out)
                verify_data = _to_numpy(transit.cpu()[:n_elems])
            if not _verify_allreduce_result(verify_data, n_elems, my_pe, npes, expected_sum, bucket_index=bucket_index):
                all_ok = False

            my_last_completed = bucket_index

        dist.barrier()

        if rank == 0:
            copy_str = "copy" if copy_output else "non-copy"
            print(f"  Multi-bucket async allreduce ({copy_str}): {num_buckets} buckets, "
                  f"max_elems/bucket={max(bucket_elems)}, correctness={'PASS' if all_ok else 'FAIL'}")

        torch.cuda.synchronize()
        dist.barrier()
        del ar
        shmem.shmem_finalize()

        if not all_ok:
            raise AssertionError(f"PE {rank}: multi-bucket async allreduce verification failed")


def test_allreduce_async_multi_bucket(
    num_buckets=12,
    seed=42,
    world_size=4,
    max_elems_per_bucket=None,
    copy_output=True,
    fixed_elems=None,
):
    """多 bucket、不同尺寸、async (start_async/wait_async) 单测，模拟 DDP allreduce_hook。"""
    if max_elems_per_bucket is None and fixed_elems is None:
        max_elems_per_bucket = min(MAX_ELEMS_FP32, 4 * 1024 * 1024)
    elif fixed_elems is not None:
        max_elems_per_bucket = max(max_elems_per_bucket or 0, fixed_elems)
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    port = get_free_port()
    torch.multiprocessing.spawn(
        _run_async_multi_bucket,
        args=(world_size, port, num_buckets, seed, max_elems_per_bucket, copy_output, fixed_elems),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-bucket async allreduce (DDP hook, start_async/wait_async)")
    parser.add_argument("--num-buckets", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--max-elems-per-bucket", type=int, default=None)
    parser.add_argument("--num-buckets-1", action="store_true", help="Single bucket only")
    parser.add_argument("--fixed-elems", type=int, default=None)
    parser.add_argument("--no-copy", action="store_true", help="Use transit buffer directly (non-copy)")
    args = parser.parse_args()
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    max_el = args.max_elems_per_bucket or min(MAX_ELEMS_FP32, 4 * 1024 * 1024)
    nb = 1 if args.num_buckets_1 else args.num_buckets
    fe = f", fixed_elems={args.fixed_elems}" if args.fixed_elems else ""
    print(f"Multi-bucket async allreduce test: buckets={nb}, world_size={args.world_size}, "
          f"max_elems/bucket={max_el}, copy={not args.no_copy}{fe}")
    test_allreduce_async_multi_bucket(
        num_buckets=nb,
        seed=args.seed,
        world_size=args.world_size,
        max_elems_per_bucket=args.max_elems_per_bucket,
        copy_output=not args.no_copy,
        fixed_elems=args.fixed_elems,
    )
    print("Done.")
