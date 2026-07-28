"""Opt-in real-weight smoke test for the Modern VLM node.

This is intentionally excluded from pytest because it downloads multi-gigabyte
models. Run one checkpoint per process so CUDA and file-handle cleanup are also
exercised:

    python tests/manual_model_smoke.py --model "Qwen 3.5 2B"
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from ComfyUI_VLM_nodes.nodes.modern_vlm import MODEL_CATALOG, ModernVLMPredictor


def test_image() -> torch.Tensor:
    image = torch.zeros((1, 96, 128, 3), dtype=torch.float32)
    image[:, 20:76, 28:104, 0] = 1.0
    return image


def test_video() -> torch.Tensor:
    frames = torch.zeros((4, 96, 128, 3), dtype=torch.float32)
    for index in range(4):
        left = 12 + index * 18
        frames[index, 30:66, left : left + 24, 1] = 1.0
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_CATALOG)
    parser.add_argument(
        "--memory-mode",
        default="ComfyUI managed (BF16)",
        choices=[
            "ComfyUI managed (BF16)",
            "4-bit NF4 (bitsandbytes)",
            "8-bit (bitsandbytes)",
            "CPU",
        ],
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        free_before, total = torch.cuda.mem_get_info()
    else:
        free_before = total = 0

    predictor = ModernVLMPredictor(
        args.model,
        "",
        args.memory_mode,
        "Auto (SDPA)",
    )
    try:
        prompt = (
            "In this four-frame video, what color object moves horizontally? "
            "Answer with the color and shape."
            if args.video
            else (
                "Describe the dominant colors, shapes, and motion in one "
                "short factual sentence."
            )
        )
        response = predictor.generate(
            None if args.video else test_image(),
            prompt,
            "",
            args.max_new_tokens,
            0.0,
            0.9,
            test_video() if args.video else None,
            2.0,
        )
        if not response.strip():
            raise RuntimeError("The model returned an empty response.")
        if args.video and "green" not in response.lower():
            raise RuntimeError(
                f"The video frames were not understood; response was: {response}"
            )
        if not args.video and "red" not in response.lower():
            raise RuntimeError(
                f"The image was not understood; response was: {response}"
            )
    finally:
        predictor.close()

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
        free_after, _ = torch.cuda.mem_get_info()
    else:
        peak = free_after = 0
    record = {
        "model": args.model,
        "repo_id": MODEL_CATALOG[args.model].repo_id,
        "memory_mode": args.memory_mode,
        "video": args.video,
        "response": response,
        "seconds": round(time.perf_counter() - started, 2),
        "cuda_total_gib": round(total / 1024**3, 2),
        "cuda_free_before_gib": round(free_before / 1024**3, 2),
        "cuda_free_after_gib": round(free_after / 1024**3, 2),
        "cuda_peak_allocated_gib": round(peak / 1024**3, 2),
    }
    print("MODEL_SMOKE_JSON=" + json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
