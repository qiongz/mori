#!/usr/bin/env python3
"""
AllReduce bandwidth benchmark: MORI vs NCCL, sync vs async, in-place vs out-of-place.

Sweeps message size (2x), fp32, collects algo_bw and bus_bw for:
  - mori_outplace, mori_inplace (sync)
  - mori_async_outplace, mori_async_outplace_nocopy (async, DDP hook style)
  - mori_async_inplace_work (async work-handle, NCCL-like)
  - nccl_outplace, nccl_inplace

Plots bw vs message-size. Config: benchmark_allreduce_config.yaml
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.distributed as dist
import mori.shmem as shmem
from mori.ccl import AllreduceSdma
from tests.python.utils import TorchDistContext, get_free_port

# Ensure repo root and ccl dir in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "..", "..", ".."))
sys.path.insert(0, _repo_root)
sys.path.insert(0, _script_dir)
from test_allreduce import (
    _to_numpy,
    _verify_allreduce_result,
    _run_benchmark,
    _RCCL_DTYPE_MAP,
)


def _get_bw_stats(exec_times, data_bytes, npes, rank):
    """Compute algo_bw and bus_bw from exec_times. Returns (algo_bw, bus_bw) for rank 0."""
    if len(exec_times) > 0:
        avg_time = np.mean(exec_times)
        min_time = np.min(exec_times)
        max_time = np.max(exec_times)
    else:
        avg_time = min_time = max_time = 0.0

    min_t = torch.tensor([min_time], dtype=torch.float64)
    max_t = torch.tensor([max_time], dtype=torch.float64)
    avg_t = torch.tensor([avg_time], dtype=torch.float64)

    dist.all_reduce(min_t, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_t, op=dist.ReduceOp.MAX)
    dist.all_reduce(avg_t, op=dist.ReduceOp.SUM)

    g_avg = avg_t.item() / npes
    algo_bw = data_bytes / g_avg / (1024.0 ** 3) if g_avg > 0 else 0.0
    bus_bw = algo_bw * 2 * (npes - 1) / npes if npes > 1 else algo_bw
    return (algo_bw, bus_bw)


def _bench_mori_outplace(rank, my_pe, npes, elems, data_bytes, output_buf_size,
                        fill_value, dtype, device, stream, iterations, warmup):
    ar = AllreduceSdma(
        my_pe, npes,
        input_buffer_size=data_bytes,
        output_buffer_size=output_buf_size,
        copy_output_to_user=False,
        dtype=dtype,
    )
    input_tensor = torch.full((elems,), fill_value, dtype=dtype, device=device)
    output_tensor = torch.zeros(elems, dtype=dtype, device=device)
    torch.cuda.synchronize()
    dist.barrier()

    times = _run_benchmark(
        lambda: ar(input_tensor, output_tensor, elems, stream),
        iterations, warmup, stream, rank,
    )
    dist.barrier()
    del ar
    return _get_bw_stats(times, data_bytes, npes, rank)


def _bench_mori_inplace(rank, my_pe, npes, elems, data_bytes, output_buf_size,
                       fill_value, dtype, device, stream, iterations, warmup):
    ar = AllreduceSdma(
        my_pe, npes,
        input_buffer_size=data_bytes,
        output_buffer_size=output_buf_size,
        copy_output_to_user=False,
        dtype=dtype,
    )
    inplace_tensor = torch.full((elems,), fill_value, dtype=dtype, device=device)
    torch.cuda.synchronize()
    dist.barrier()

    def _fn():
        inplace_tensor.fill_(fill_value)
        return ar.allreduce_inplace(inplace_tensor, elems, stream)

    times = _run_benchmark(_fn, iterations, warmup, stream, rank)
    dist.barrier()
    del ar
    return _get_bw_stats(times, data_bytes, npes, rank)


def _bench_mori_async_outplace(rank, my_pe, npes, elems, data_bytes, output_buf_size,
                              fill_value, dtype, device, stream, iterations, warmup,
                              copy_output=True):
    """MORI async: start_async + wait_async (DDP hook style)."""
    ar = AllreduceSdma(
        my_pe, npes,
        input_buffer_size=data_bytes,
        output_buffer_size=output_buf_size,
        copy_output_to_user=copy_output,
        dtype=dtype,
    )
    input_tensor = torch.full((elems,), fill_value, dtype=dtype, device=device)
    output_tensor = torch.zeros(elems, dtype=dtype, device=device)
    torch.cuda.synchronize()
    dist.barrier()

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    exec_times = []
    for i in range(warmup + iterations):
        input_tensor.fill_(fill_value)
        dist.barrier()
        ev_start.record(stream)
        ar.start_async(input_tensor, output_tensor, elems, stream)
        ar.wait_async(stream)
        ev_end.record(stream)
        stream.synchronize()
        elapsed = ev_start.elapsed_time(ev_end) / 1000.0
        if i >= warmup:
            exec_times.append(elapsed)
    dist.barrier()
    del ar
    return _get_bw_stats(exec_times, data_bytes, npes, rank)


def _bench_mori_async_inplace_work(rank, my_pe, npes, elems, data_bytes, output_buf_size,
                                   fill_value, dtype, device, stream, iterations, warmup):
    """MORI async in-place work handle: allreduce_inplace_async + work.wait()."""
    if dtype != torch.float32:
        raise RuntimeError("mori_async_inplace_work currently supports fp32 only")
    ar = AllreduceSdma(
        my_pe, npes,
        input_buffer_size=data_bytes,
        output_buffer_size=output_buf_size,
        copy_output_to_user=True,
        dtype=dtype,
    )
    inplace_tensor = torch.full((elems,), fill_value, dtype=dtype, device=device)
    torch.cuda.synchronize()
    dist.barrier()

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    exec_times = []
    for i in range(warmup + iterations):
        inplace_tensor.fill_(fill_value)
        dist.barrier()
        ev_start.record(stream)
        work = ar.allreduce_inplace_async(inplace_tensor, elems, stream=stream)
        if work is None:
            raise RuntimeError("allreduce_inplace_async returned None")
        work.wait()
        ev_end.record(stream)
        stream.synchronize()
        elapsed = ev_start.elapsed_time(ev_end) / 1000.0
        if i >= warmup:
            exec_times.append(elapsed)
    dist.barrier()
    del ar
    return _get_bw_stats(exec_times, data_bytes, npes, rank)


def _bench_nccl_outplace(rank, my_pe, npes, elems, data_bytes,
                         fill_value, dtype, device, stream, iterations, warmup):
    rccl_dtype = _RCCL_DTYPE_MAP.get(dtype, torch.float32)
    rccl_fill = float(fill_value) if rccl_dtype in (torch.float16, torch.bfloat16, torch.float32) else fill_value
    input_tensor = torch.full((elems,), rccl_fill, dtype=rccl_dtype, device=device)
    output_tensor = torch.zeros(elems, dtype=rccl_dtype, device=device)
    torch.cuda.synchronize()
    dist.barrier()

    exec_times = []
    for i in range(warmup + iterations):
        output_tensor.copy_(input_tensor)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(output_tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if i >= warmup:
            exec_times.append(elapsed)
    dist.barrier()
    return _get_bw_stats(exec_times, data_bytes, npes, rank)


def _bench_nccl_inplace(rank, my_pe, npes, elems, data_bytes,
                        fill_value, dtype, device, stream, iterations, warmup):
    rccl_dtype = _RCCL_DTYPE_MAP.get(dtype, torch.float32)
    rccl_fill = float(fill_value) if rccl_dtype in (torch.float16, torch.bfloat16, torch.float32) else fill_value
    inplace_tensor = torch.full((elems,), rccl_fill, dtype=rccl_dtype, device=device)
    torch.cuda.synchronize()
    dist.barrier()

    exec_times = []
    for i in range(warmup + iterations):
        inplace_tensor.fill_(rccl_fill)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(inplace_tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if i >= warmup:
            exec_times.append(elapsed)
    dist.barrier()
    return _get_bw_stats(exec_times, data_bytes, npes, rank)


def _bench_worker(rank, world_size, port, size_mb, elems, iterations, warmup,
                 dtype, modes, result_file):
    """Worker: run each mode, collect results, write to result_file (rank 0)."""
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        assert my_pe == rank and npes == world_size

        elem_size = torch.tensor([], dtype=dtype).element_size()
        data_bytes = elems * elem_size
        output_buf_size = npes * (elems // npes + 64) * elem_size
        device = torch.device(f"cuda:{rank}")
        stream = torch.cuda.Stream(device=device)
        fill_value = (my_pe + 1) * 1000

        results = {"size_mb": size_mb, "modes": {}}

        for mode in modes:
            try:
                if mode == "mori_outplace":
                    bw = _bench_mori_outplace(
                        rank, my_pe, npes, elems, data_bytes, output_buf_size,
                        fill_value, dtype, device, stream, iterations, warmup,
                    )
                elif mode == "mori_inplace":
                    bw = _bench_mori_inplace(
                        rank, my_pe, npes, elems, data_bytes, output_buf_size,
                        fill_value, dtype, device, stream, iterations, warmup,
                    )
                elif mode == "nccl_outplace":
                    bw = _bench_nccl_outplace(
                        rank, my_pe, npes, elems, data_bytes,
                        fill_value, dtype, device, stream, iterations, warmup,
                    )
                elif mode == "nccl_inplace":
                    bw = _bench_nccl_inplace(
                        rank, my_pe, npes, elems, data_bytes,
                        fill_value, dtype, device, stream, iterations, warmup,
                    )
                elif mode == "mori_async_outplace":
                    bw = _bench_mori_async_outplace(
                        rank, my_pe, npes, elems, data_bytes, output_buf_size,
                        fill_value, dtype, device, stream, iterations, warmup,
                        copy_output=True,
                    )
                elif mode == "mori_async_outplace_nocopy":
                    bw = _bench_mori_async_outplace(
                        rank, my_pe, npes, elems, data_bytes, output_buf_size,
                        fill_value, dtype, device, stream, iterations, warmup,
                        copy_output=False,
                    )
                elif mode == "mori_async_inplace_work":
                    bw = _bench_mori_async_inplace_work(
                        rank, my_pe, npes, elems, data_bytes, output_buf_size,
                        fill_value, dtype, device, stream, iterations, warmup,
                    )
                else:
                    continue

                if rank == 0 and bw is not None:
                    results["modes"][mode] = {"algo_bw": bw[0], "bus_bw": bw[1]}
            except Exception as e:
                if rank == 0:
                    results["modes"][mode] = {"error": str(e)}
                    print(f"  {mode} failed: {e}")

        if rank == 0 and result_file:
            with open(result_file, "w") as f:
                json.dump(results, f, indent=2)

        torch.cuda.synchronize()
        dist.barrier()
        shmem.shmem_finalize()


def sweep_sizes(min_mb, max_mb, multiplier):
    """Generate size list: min_mb, min_mb*multiplier, ... until <= max_mb."""
    sizes = []
    s = min_mb
    while s <= max_mb:
        sizes.append(s)
        s *= multiplier
    return sizes


def load_config(config_path):
    """Load YAML config. Fallback to defaults if pyyaml not available."""
    defaults = {
        "size_mb_sweep": {"min_mb": 1, "max_mb": 8192, "multiplier": 2},
        "dtype": "fp32",
        "modes": [
            "mori_outplace", "mori_inplace",
            "mori_async_outplace", "mori_async_outplace_nocopy",
            "nccl_outplace", "nccl_inplace",
        ],
        "world_size": 8,
        "iterations": 10,
        "warmup": 10,
        "output_dir": ".",
        "plot_format": "png",
        "plot_dpi": 150,
    }
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        if cfg:
            for k, v in defaults.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
    except Exception:
        pass
    return defaults


def _run_single_size(size_mb, config, result_file, verbose=True):
    """Run benchmark for one size, return results dict."""
    dtype_map = {"fp32": torch.float32, "float32": torch.float32}
    dtype = dtype_map.get(config.get("dtype", "fp32"), torch.float32)
    elem_size = 4
    elems = (size_mb * 1024 * 1024) // elem_size

    # Align for MORI (npes * pack_size)
    npes = config.get("world_size", 8)
    align = npes * 4
    elems = (elems // align) * align
    if elems < align:
        elems = align

    port = get_free_port()
    modes = config.get("modes", ["mori_outplace", "mori_inplace", "nccl_outplace", "nccl_inplace"])
    iterations = config.get("iterations", 10)
    warmup = config.get("warmup", 10)

    if verbose:
        print(f"  Size {size_mb} MB, elems={elems:,} ... ", end="", flush=True)

    torch.multiprocessing.spawn(
        _bench_worker,
        args=(npes, port, size_mb, elems, iterations, warmup, dtype, modes, result_file),
        nprocs=npes,
        join=True,
    )

    with open(result_file) as f:
        results = json.load(f)
    if verbose:
        print("done")
    return results


def plot_results(all_results, config, output_path):
    """Plot algo_bw and bus_bw vs message-size for each mode."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    sizes = sorted([r["size_mb"] for r in all_results])
    modes = list(all_results[0]["modes"].keys()) if all_results else []

    colors = {
        "mori_outplace": "#2ecc71", "mori_inplace": "#27ae60",
        "mori_async_outplace": "#1abc9c", "mori_async_outplace_nocopy": "#16a085",
        "mori_async_inplace_work": "#0e6655",
        "nccl_outplace": "#3498db", "nccl_inplace": "#2980b9",
    }
    markers = {
        "mori_outplace": "o", "mori_inplace": "s",
        "mori_async_outplace": "v", "mori_async_outplace_nocopy": "<",
        "mori_async_inplace_work": "P",
        "nccl_outplace": "^", "nccl_inplace": "D",
    }
    linestyles = {
        "mori_async_outplace": "--", "mori_async_outplace_nocopy": "--",
        "mori_async_inplace_work": "--",
    }  # async: dashed; sync: solid (default)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AllReduce Bandwidth vs Message Size (fp32) — Sync & Async", fontsize=14)

    for bw_key, ax_title in [("algo_bw", "Algo BW (GB/s)"), ("bus_bw", "Bus BW (GB/s)")]:
        ax = axes[0] if bw_key == "algo_bw" else axes[1]
        ax.set_title(ax_title)
        ax.set_xlabel("Message Size (MB)")
        ax.set_xscale("log", base=2)
        # Use explicit message-size ticks from the measured sweep.
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes], rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        for mode in modes:
            xs, ys = [], []
            for r in all_results:
                if mode in r["modes"] and bw_key in r["modes"][mode] and "error" not in r["modes"][mode]:
                    xs.append(r["size_mb"])
                    ys.append(r["modes"][mode][bw_key])
            if xs:
                label_map = {
                    "mori_outplace": "MORI (sync out)",
                    "mori_inplace": "MORI (sync in)",
                    "mori_async_outplace": "MORI (async copy)",
                    "mori_async_outplace_nocopy": "MORI (async no-copy)",
                    "mori_async_inplace_work": "MORI (async in/work)",
                    "nccl_outplace": "NCCL (out)",
                    "nccl_inplace": "NCCL (in)",
                }
                label = label_map.get(mode, mode.replace("_", " ").title())
                ls = linestyles.get(mode, "-")
                ax.plot(xs, ys, marker=markers.get(mode, "o"), color=colors.get(mode, None),
                        linestyle=ls, label=label, linewidth=2, markersize=6)

        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    dpi = config.get("plot_dpi", 150)
    out_dir = config.get("output_dir", ".")
    out = os.path.join(out_dir, output_path) if not os.path.isabs(output_path) else output_path
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out}")


def _load_results_json(path):
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Invalid results JSON format: expected list, got {type(data)}")
    return data


def main():
    _default_config = os.path.join(_script_dir, "benchmark_allreduce_config.yaml")
    parser = argparse.ArgumentParser(description="AllReduce bandwidth benchmark (MORI vs NCCL)")
    parser.add_argument("--config", default=_default_config,
                       help="Config YAML path")
    parser.add_argument("--output", default="benchmark_allreduce_results.json",
                       help="Output JSON path")
    parser.add_argument("--plot", default="benchmark_allreduce_bw.png",
                       help="Output plot path")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--redraw", action="store_true",
                        help="Redraw plot from existing results JSON (skip benchmark run)")
    parser.add_argument("--results-json", default=None,
                        help="Existing results JSON path for --redraw (default: output path)")
    parser.add_argument("--modes", nargs="+", default=None,
                        help="Override modes (e.g. mori_async_outplace nccl_inplace)")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.modes is not None:
        config["modes"] = args.modes

    os.environ.setdefault("MORI_ENABLE_SDMA", "1")

    out_dir = config.get("output_dir", ".")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, args.output)
    all_results = []

    if args.redraw:
        redraw_json = args.results_json if args.results_json else out_json
        if not os.path.isabs(redraw_json):
            redraw_json = os.path.join(out_dir, redraw_json)
        print("AllReduce Bandwidth Plot Redraw")
        print(f"  Results JSON: {redraw_json}")
        all_results = _load_results_json(redraw_json)
        print(f"  Loaded {len(all_results)} size points")
    else:
        sweep = config.get("size_mb_sweep", {})
        sizes = sweep_sizes(
            sweep.get("min_mb", 1),
            sweep.get("max_mb", 8192),
            sweep.get("multiplier", 2),
        )

        print("AllReduce Bandwidth Benchmark")
        print(f"  Sizes (MB): {sizes}")
        print(f"  Modes: {config.get('modes')}")
        print(f"  World size: {config.get('world_size')}")
        print("-" * 60)

        import tempfile
        for size_mb in sizes:
            fd, result_file = tempfile.mkstemp(suffix=".json")
            os.close(fd)
            try:
                r = _run_single_size(size_mb, config, result_file, verbose=args.verbose)
                all_results.append(r)
            finally:
                if os.path.exists(result_file):
                    os.unlink(result_file)

        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved: {out_json}")

    if not args.no_plot:
        plot_results(all_results, config, args.plot)


if __name__ == "__main__":
    main()
