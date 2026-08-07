"""Direct Qwen3-VL Transformers benchmark with quality-preserving artifacts.

This runner measures the same local model path used by Modern VLM without
requiring a running ComfyUI server. It records preprocessing, user-visible
time-to-first-text, end-to-end latency, decode throughput, peak VRAM, and the
complete output for exact cross-iteration comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TextIteratorStreamer,
)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resize_to_longest_edge(image: Image.Image, longest_edge: int | None) -> Image.Image:
    if longest_edge is None or max(image.size) <= longest_edge:
        return image
    scale = longest_edge / max(image.size)
    size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(size, Image.Resampling.BOX)


def prepare_inputs(
    processor: Any,
    image: Image.Image,
    prompt: str,
    *,
    min_pixels: int | None,
    max_pixels: int | None,
) -> dict[str, torch.Tensor]:
    image_part: dict[str, Any] = {"type": "image", "image": image}
    if min_pixels is not None:
        image_part["min_pixels"] = min_pixels
    if max_pixels is not None:
        image_part["max_pixels"] = max_pixels
    messages = [
        {
            "role": "user",
            "content": [image_part, {"type": "text", "text": prompt}],
        }
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def run_sample(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int,
    cache_implementation: str,
    min_pixels: int | None,
    max_pixels: int | None,
    disable_compile: bool,
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    inputs = prepare_inputs(
        processor,
        image,
        prompt,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    prepared_at = time.perf_counter()
    inputs = {name: value.to(model.device) for name, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    generated: list[torch.Tensor] = []
    errors: list[BaseException] = []

    def generate() -> None:
        try:
            with torch.inference_mode():
                generated.append(
                    model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        cache_implementation=cache_implementation,
                        disable_compile=disable_compile,
                        streamer=streamer,
                    )
                )
        except BaseException as exc:
            errors.append(exc)
            streamer.end()

    first_text_at: float | None = None
    chunks: list[str] = []
    worker = threading.Thread(target=generate, daemon=True)
    worker.start()
    for chunk in streamer:
        if chunk and first_text_at is None:
            first_text_at = time.perf_counter()
        chunks.append(chunk)
    worker.join()
    if errors:
        raise errors[0]
    torch.cuda.synchronize()
    finished = time.perf_counter()
    output_ids = generated[0][:, input_length:]
    output_tokens = int(output_ids.shape[-1])
    output = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    ttft_seconds = (first_text_at or finished) - started
    decode_seconds = max(0.0, finished - (first_text_at or finished))
    return {
        "preprocess_ms": round((prepared_at - started) * 1000, 3),
        "ttft_ms": round(ttft_seconds * 1000, 3),
        "e2e_ms": round((finished - started) * 1000, 3),
        "output_tokens": output_tokens,
        "output_tokens_per_second": (
            round(max(0, output_tokens - 1) / decode_seconds, 3)
            if output_tokens > 1 and decode_seconds > 0
            else None
        ),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "input_tokens": input_length,
        "vision_tokens": int(inputs.get("pixel_values", torch.empty(0)).shape[0]),
        "output": output,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def metric(name: str) -> list[float]:
        return [float(sample[name]) for sample in samples]

    rates = [
        float(sample["output_tokens_per_second"])
        for sample in samples
        if sample["output_tokens_per_second"] is not None
    ]
    return {
        "preprocess_ms_mean": round(statistics.fmean(metric("preprocess_ms")), 3),
        "ttft_ms": {
            "p50": round(percentile(metric("ttft_ms"), 0.50), 3),
            "p95": round(percentile(metric("ttft_ms"), 0.95), 3),
        },
        "e2e_ms": {
            "p50": round(percentile(metric("e2e_ms"), 0.50), 3),
            "p95": round(percentile(metric("e2e_ms"), 0.95), 3),
        },
        "output_tokens_per_second_mean": round(statistics.fmean(rates), 3),
        "peak_vram_gib": round(max(metric("peak_vram_gib")), 3),
        "output_tokens_mean": round(statistics.fmean(metric("output_tokens")), 3),
        "outputs_identical": len({sample["output_sha256"] for sample in samples}) == 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="Describe this image precisely in one sentence.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--attention",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--cache", choices=("dynamic", "static"), default="dynamic")
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--longest-edge", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--expected-output-sha256")
    parser.add_argument(
        "--float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")
    if args.runs < 1 or args.warmups < 0:
        parser.error("--runs must be positive and --warmups non-negative")
    torch.set_float32_matmul_precision(args.float32_matmul_precision)
    image_path = args.image.resolve()
    source_image = Image.open(image_path).convert("RGB")
    image = resize_to_longest_edge(source_image, args.longest_edge)

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attention,
        device_map="cuda",
    ).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    samples = []
    for index in range(args.warmups + args.runs):
        sample = run_sample(
            model,
            processor,
            image,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            cache_implementation=args.cache,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            disable_compile=args.disable_compile,
        )
        print(
            f"{index + 1}/{args.warmups + args.runs} "
            f"ttft={sample['ttft_ms']:.1f}ms "
            f"e2e={sample['e2e_ms']:.1f}ms "
            f"tok/s={sample['output_tokens_per_second']}"
        )
        if index >= args.warmups:
            samples.append(sample)

    artifact = {
        "schema": "comfyui-vlm/transformers-benchmark",
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "label": args.label,
        "model": args.model,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "media": {
            "path": os.fspath(image_path),
            "sha256": sha256_file(image_path),
            "source_width": source_image.width,
            "source_height": source_image.height,
            "processed_width": image.width,
            "processed_height": image.height,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "transformers": __import__("transformers").__version__,
            "flash_attn": (
                __import__("flash_attn").__version__
                if args.attention == "flash_attention_2"
                else None
            ),
        },
        "settings": {
            "attention": args.attention,
            "cache": args.cache,
            "disable_compile": args.disable_compile,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "longest_edge": args.longest_edge,
            "max_new_tokens": args.max_new_tokens,
            "warmups": args.warmups,
            "runs": args.runs,
            "float32_matmul_precision": args.float32_matmul_precision,
        },
        "model_load_seconds": round(load_seconds, 3),
        "quality_gate": {
            "method": "byte-identical output SHA-256",
            "reference_sha256": args.expected_output_sha256,
            "passed": (
                all(
                    sample["output_sha256"] == args.expected_output_sha256
                    for sample in samples
                )
                if args.expected_output_sha256
                else None
            ),
        },
        "summary": aggregate(samples),
        "samples": samples,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.label}.json"
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
