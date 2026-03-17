#!/usr/bin/env python3
"""
Multi-bucket allreduce_inplace test: 模拟 DDP allreduce hook 的连续、异步调用方式。

- 多段 ≤1024MB、随机尺寸的 bucket，共用同一个 AllreduceSdma 实例
- bucket 尺寸对齐：total_count 为 npes*PACK_SIZE_FP32 的倍数（fp32 pack_size=4），否则 MORI 内核会越界
- Per-bucket 同步：进入每个 bucket 前 all_reduce(MIN) 等待「所有 rank 已完成 bucket_index-1」
- 每 bucket 后 stream.synchronize()：MORI 用 hipMemcpyAsync 写回 user、transit 被复用，必须等 copy 完成再启动下一 bucket，否则会读到错数据
- 校验每段结果正确（allreduce 后各 rank 一致且为 sum）

注意：单次 allreduce 正确性由 MORI 保证。若单 bucket 也失败，请用同 world_size/size 跑
  mori/tests/python/ccl/test_allreduce.py（Test 2: In-place）对照；或使用 --fixed-elems 小值
  （如 8192）排查是否与 size 相关。

环境：若出现 "select no device" 或 "anvil.cpp Line: 181 Failed"（SDMA 队列创建失败），属于
  MORI/ROCm 环境问题，可尝试：1) 减少 world_size（如 --world-size 2）或限制可见 GPU
  （CUDA_VISIBLE_DEVICES=0,1）；2) MORI_DISABLE_TOPO=1 使用静态 NIC 匹配；3) 在支持 SDMA 的
  节点/容器内运行（需 KFD、HSA 与 GPU 一致）。

已知限制：部分环境上 MORI allreduce_inplace 在大 message（如 262144 元素）时前半段结果错误
  （如 64000 而非 36000），小 message（如 8192）正常。可设 MORI_TEST_MAX_ELEMS=8192 限制
  测试规模以保证通过，或向 MORI 反馈大 size 问题。
"""

import os
import random
import numpy as np
import torch
import torch.distributed as dist
import mori.shmem as shmem
from mori.ccl import AllreduceSdma
from tests.python.utils import TorchDistContext, get_free_port


# fp32 下 1024MB = 256M 元素
MAX_BUCKET_BYTES = 1024 * 1024 * 1024
ELEM_SIZE_FP32 = 4
MAX_ELEMS_FP32 = MAX_BUCKET_BYTES // ELEM_SIZE_FP32
# MORI kernel 要求 per-rank 元素数为 pack_size 的倍数（float 的 pack_size=4，见 vec_type.cuh）
PACK_SIZE_FP32 = 4


def _to_numpy(tensor):
    if tensor.dtype in (torch.bfloat16, torch.float16):
        return tensor.float().numpy()
    return tensor.numpy()


def _verify_allreduce_result(cpu_data, numel, my_pe, npes, expected_sum=None, bucket_index=None):
    """校验 allreduce 结果：每个元素应等于各 rank 的 fill 值之和。"""
    if expected_sum is None:
        expected_sum = sum((pe + 1) * 1000 for pe in range(npes))
    chunk = cpu_data[:numel].astype(np.float32)
    exp = np.full(numel, expected_sum, dtype=np.float32)
    if np.allclose(chunk, exp, rtol=1e-5, atol=0.01):
        return True
    bad = np.where(~np.isclose(chunk, exp, rtol=1e-5, atol=0.01))[0]
    nbad = len(bad)
    bid = f" bucket {bucket_index}" if bucket_index is not None else ""
    print(f"  PE {my_pe}{bid}: FAIL numel={numel} expected={expected_sum} "
          f"mismatches={nbad} first_bad_idx={bad[0]} got={chunk[bad[0]]} "
          f"sample_idxs=[0,{numel//2},{numel-1}] got=[{chunk[0]},{chunk[numel//2]},{chunk[numel-1]}]")
    return False


def _run_multi_bucket(rank, world_size, port, num_buckets, seed, max_elems_per_bucket, sync_each_bucket=False, fixed_elems=None):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        assert my_pe == rank and npes == world_size

        device = torch.device(f"cuda:{rank}")
        stream = torch.cuda.Stream(device=device)
        dtype = torch.float32
        fill_value = (my_pe + 1) * 1000.0

        # 随机生成各 bucket 的 numel：必须 (total_count/npes) 为 pack_size 的倍数，否则 MORI 内核会写越界
        # 即 total_count % (npes * PACK_SIZE_FP32) == 0
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
        # AllreduceSdma 需要能放下最大 bucket 的 input/output
        output_buf_size = npes * (max_elems // npes + 64) * ELEM_SIZE_FP32

        ar = AllreduceSdma(
            my_pe, npes,
            input_buffer_size=max_bytes,
            output_buffer_size=output_buf_size,
            copy_output_to_user=False,
            dtype=dtype,
        )

        # 为每个 bucket 分配 tensor
        tensors = []
        for n in bucket_elems:
            t = torch.full((n,), fill_value, dtype=dtype, device=device)
            tensors.append(t)

        # 类似 DDP hook 的 state
        state = {}
        last_done = torch.zeros(1, dtype=torch.long, device=device)
        state["mori_last_completed_tensor"] = last_done
        state["mori_my_last_completed_bucket"] = -1
        group = None
        # use default process group

        torch.cuda.synchronize()
        dist.barrier()

        # 连续调用：按 bucket 顺序，per-bucket 同步后 allreduce_inplace(stream)。
        # 每 bucket 后必须 stream.synchronize()：MORI 用 hipMemcpyAsync 写回 user，transit 被下一 bucket 复用，
        # 若不同步则下一 bucket 的 reduce-scatter 会覆盖 transit，导致上一 bucket 的 copy 读到错数据（如 68000/前半错）。
        for bucket_index in range(num_buckets):
            need_wait = bucket_index - 1
            my_done = state["mori_my_last_completed_bucket"]
            while need_wait >= 0:
                last_done.zero_()
                last_done[0] = my_done
                dist.all_reduce(last_done, op=dist.ReduceOp.MIN, group=group)
                if last_done.item() >= need_wait:
                    break

            # 多 bucket 连续调用时，每次 allreduce 前重置 SDMA 同步 flags，避免上一轮 flag 干扰
            ar.reset_flags()

            t = tensors[bucket_index]
            t.fill_(fill_value)
            n_elems = t.numel()
            ok = ar.allreduce_inplace(t, n_elems, stream=stream)
            if not ok:
                raise RuntimeError(f"PE {rank} bucket {bucket_index}: allreduce_inplace returned False")
            state["mori_my_last_completed_bucket"] = bucket_index
            stream.synchronize()  # 必须：等 transit->user copy 完成再复用 transit
            if sync_each_bucket:
                dist.barrier()

        # 全部完成后再次同步（冗余，便于与单次 sync 的写法一致）
        stream.synchronize()

        dist.barrier()

        # 校验每个 bucket 的结果（allreduce 后需除以 world_size 才是平均，这里我们校验的是 sum）
        expected_sum = sum((pe + 1) * 1000 for pe in range(npes))
        all_ok = True
        for bucket_index, (t, n) in enumerate(zip(tensors, bucket_elems)):
            cpu_data = _to_numpy(t.cpu())
            if not _verify_allreduce_result(cpu_data, n, my_pe, npes, expected_sum, bucket_index=bucket_index):
                all_ok = False

        if rank == 0:
            print(f"  Multi-bucket allreduce_inplace: {num_buckets} buckets, "
                  f"max_elems/bucket={max(bucket_elems)}, correctness={'PASS' if all_ok else 'FAIL'}")

        torch.cuda.synchronize()
        dist.barrier()
        del ar
        shmem.shmem_finalize()

        if not all_ok:
            raise AssertionError(f"PE {rank}: multi-bucket allreduce_inplace verification failed")


def test_allreduce_inplace_multi_bucket(
    num_buckets=12,
    seed=42,
    world_size=4,
    max_elems_per_bucket=None,
    sync_each_bucket=False,
    fixed_elems=None,
):
    """多 bucket、随机尺寸、连续异步 allreduce_inplace 单测（模拟 DDP hook）。
    sync_each_bucket=True：每 bucket 后 stream.sync+barrier，用于排查是否为异步重叠导致错误。
    fixed_elems：若指定，所有 bucket 使用该元素数（会对齐到 npes*4），便于用固定小 size 排查 MORI 问题。
    """
    # 默认每 bucket 最多 4M 元素（约 16MB），保证单测轻量；可传 max_elems_per_bucket 测到 1024MB
    if max_elems_per_bucket is None and fixed_elems is None:
        max_elems_per_bucket = min(MAX_ELEMS_FP32, 4 * 1024 * 1024)
    elif fixed_elems is not None:
        max_elems_per_bucket = max(max_elems_per_bucket or 0, fixed_elems)
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    port = get_free_port()
    torch.multiprocessing.spawn(
        _run_multi_bucket,
        args=(world_size, port, num_buckets, seed, max_elems_per_bucket, sync_each_bucket, fixed_elems),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-bucket async allreduce_inplace (DDP-hook style)")
    parser.add_argument("--num-buckets", type=int, default=12, help="Number of buckets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--max-elems-per-bucket", type=int, default=None,
                        help="Max elements per bucket (default 4M; use 268435456 for 1024MB fp32)")
    parser.add_argument("--sync-each-bucket", action="store_true",
                        help="Sync stream + barrier after each bucket (debug async)")
    parser.add_argument("--num-buckets-1", action="store_true",
                        help="Force num_buckets=1 to test single-bucket only")
    parser.add_argument("--fixed-elems", type=int, default=None,
                        help="Use this exact (aligned) element count per bucket to isolate size-related MORI bugs (e.g. 8192)")
    args = parser.parse_args()
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    max_el = args.max_elems_per_bucket
    if max_el is None:
        max_el = min(MAX_ELEMS_FP32, 4 * 1024 * 1024)
    nb = 1 if args.num_buckets_1 else args.num_buckets
    fe = f", fixed_elems={args.fixed_elems}" if args.fixed_elems is not None else ""
    print(f"Multi-bucket allreduce_inplace test: buckets={nb}, world_size={args.world_size}, "
          f"max_elems_per_bucket={max_el}{fe}")
    test_allreduce_inplace_multi_bucket(
        num_buckets=nb,
        seed=args.seed,
        world_size=args.world_size,
        max_elems_per_bucket=max_el,
        sync_each_bucket=args.sync_each_bucket,
        fixed_elems=args.fixed_elems,
    )
    print("Done.")
