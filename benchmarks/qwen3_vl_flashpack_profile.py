"""Profile cold and warm disk-to-GPU loading for Qwen3-VL weights.

FlashPack conversion is deliberately outside the timed path. Each measured
run uses a new Python process so CUDA allocator state cannot leak across runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def drop_file_cache(path: Path) -> None:
    """Ask Linux to evict this file's pages without dropping global caches."""
    if not hasattr(os, "posix_fadvise"):
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def load_once(method: str, path: Path) -> dict[str, Any]:
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if method.startswith("safetensors"):
        if method == "safetensors_fast_gpu":
            os.environ["SAFETENSORS_FAST_GPU"] = "1"
        else:
            os.environ.pop("SAFETENSORS_FAST_GPU", None)
        from safetensors.torch import load_file

        loaded = load_file(str(path), device="cuda")
        tensor_count = len(loaded)
    elif method == "flashpack":
        # FlashPack main currently defaults to 16 readers, two 64 MiB pinned
        # buffers per reader (2 GiB total). That failed on the RTX 3090 WSL
        # test host. Keep the benchmark's default bounded and let callers
        # override every value explicitly when tuning another machine.
        os.environ.setdefault("FLASHPACK_READ_THREADS", "4")
        os.environ.setdefault("FLASHPACK_READ_CHUNK_BYTES", str(32 * 1024 * 1024))
        os.environ.setdefault("FLASHPACK_CACHE_PINNED", "0")
        from flashpack.deserialization import read_flashpack_file

        loaded, metadata = read_flashpack_file(path=str(path), device="cuda")
        tensor_count = len(metadata["index"])
    else:
        raise ValueError(f"Unknown method: {method}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_bytes = torch.cuda.max_memory_allocated()
    del loaded
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "seconds": elapsed,
        "tensor_count": tensor_count,
        "peak_gpu_bytes": peak_bytes,
    }


def worker(method: str, path: Path, runs: int, cold_only: bool) -> None:
    samples = []
    for _ in range(runs):
        drop_file_cache(path)
        cold = load_once(method, path)
        warm = None if cold_only else load_once(method, path)
        samples.append({"method": method, "cold": cold, "warm": warm})
    print(json.dumps(samples[0] if runs == 1 else {"samples": samples}))


def prepare(safetensors_path: Path, flashpack_path: Path) -> dict[str, Any]:
    import torch
    from flashpack import is_flashpack_file, pack_to_file
    from flashpack.deserialization import (
        iterate_from_flash_tensor,
        read_flashpack_file,
    )
    from safetensors import safe_open
    from safetensors.torch import load_file

    conversion_seconds = 0.0
    if not flashpack_path.exists() or not is_flashpack_file(str(flashpack_path)):
        flashpack_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        state_dict = load_file(str(safetensors_path), device="cpu")
        pack_to_file(
            state_dict,
            str(flashpack_path),
            target_dtype=None,
            silent=False,
        )
        conversion_seconds = time.perf_counter() - started
        del state_dict
        gc.collect()

    storage, metadata = read_flashpack_file(str(flashpack_path), device="cpu")
    packed_tensors = dict(iterate_from_flash_tensor(storage, metadata))
    names = list(packed_tensors)
    sample_names = [names[0], names[len(names) // 2], names[-1]]
    exact = {}
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as source:
        for name in sample_names:
            exact[name] = bool(torch.equal(source.get_tensor(name), packed_tensors[name]))
    del packed_tensors, storage
    gc.collect()
    drop_file_cache(safetensors_path)
    drop_file_cache(flashpack_path)
    return {
        "conversion_seconds": conversion_seconds,
        "flashpack_bytes": flashpack_path.stat().st_size,
        "tensor_count": len(metadata["index"]),
        "sample_exact": exact,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def summarize(samples: list[dict[str, Any]], file_bytes: int) -> dict[str, Any]:
    summary = {}
    for cache_state in ("cold", "warm"):
        seconds = [sample[cache_state]["seconds"] for sample in samples]
        median = statistics.median(seconds)
        summary[cache_state] = {
            "seconds": seconds,
            "p50_seconds": median,
            "p95_seconds": percentile(seconds, 0.95),
            "p50_throughput_gbps": file_bytes * 8 / median / 1e9,
            "peak_gpu_bytes": max(
                sample[cache_state]["peak_gpu_bytes"] for sample in samples
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safetensors", type=Path, required=True)
    parser.add_argument("--flashpack", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("safetensors", "safetensors_fast_gpu", "flashpack"))
    parser.add_argument("--worker-runs", type=int, default=1)
    parser.add_argument("--cold-only", action="store_true")
    args = parser.parse_args()

    if args.worker:
        worker(
            args.worker,
            args.flashpack if args.worker == "flashpack" else args.safetensors,
            args.worker_runs,
            args.cold_only,
        )
        return

    preparation = prepare(args.safetensors, args.flashpack)
    methods = ("safetensors", "safetensors_fast_gpu", "flashpack")
    result: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint": "Qwen/Qwen3-VL-2B-Instruct",
        "safetensors_bytes": args.safetensors.stat().st_size,
        "flashpack_reader": {
            "threads": int(os.environ.get("FLASHPACK_READ_THREADS", "4")),
            "chunk_bytes": int(
                os.environ.get("FLASHPACK_READ_CHUNK_BYTES", str(32 * 1024 * 1024))
            ),
            "cache_pinned": os.environ.get("FLASHPACK_CACHE_PINNED", "0"),
        },
        "preparation": preparation,
        "methods": {},
    }
    for method in methods:
        samples = []
        for _ in range(args.runs):
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--safetensors",
                    str(args.safetensors),
                    "--flashpack",
                    str(args.flashpack),
                    "--worker",
                    method,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            samples.append(json.loads(completed.stdout.strip().splitlines()[-1]))
        file_bytes = (
            preparation["flashpack_bytes"]
            if method == "flashpack"
            else args.safetensors.stat().st_size
        )
        result["methods"][method] = summarize(samples, file_bytes)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
