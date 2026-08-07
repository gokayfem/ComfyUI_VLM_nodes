"""Run the core Qwen3-VL optimization matrix in one loaded-model process."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from PIL import Image
from qwen3_vl_transformers import aggregate, resize_to_longest_edge, run_sample
from transformers import AutoModelForImageTextToText, AutoProcessor

VARIANTS = (
    {
        "id": "00",
        "label": "BF16 SDPA / dynamic cache / source resolution",
        "longest_edge": None,
        "cache": "dynamic",
        "warmups": 2,
    },
    {
        "id": "01a",
        "label": "BF16 SDPA / dynamic cache / 672px edge",
        "longest_edge": 672,
        "cache": "dynamic",
        "warmups": 2,
    },
    {
        "id": "01b",
        "label": "BF16 SDPA / dynamic cache / 448px edge",
        "longest_edge": 448,
        "cache": "dynamic",
        "warmups": 2,
    },
    {
        "id": "02",
        "label": "BF16 SDPA / static compiled cache / 448px edge",
        "longest_edge": 448,
        "cache": "static",
        "warmups": 6,
        "exact_reference": "01b",
    },
)

DEFAULT_CONCEPT_GROUPS = (
    ("woman", "person"),
    ("golden retriever", "dog"),
    ("beach", "sand"),
    ("high-five", "high five"),
)


def evaluate_concepts(output: str, groups: tuple[tuple[str, ...], ...]) -> dict:
    normalized = output.casefold()
    matched = [next((term for term in group if term in normalized), None) for group in groups]
    return {
        "passed": all(matched),
        "matched": matched,
        "required": [list(group) for group in groups],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--prompt", default="Describe this image precisely in one sentence.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/qwen3-vl-2b-matrix-tf5.json"),
    )
    args = parser.parse_args()
    source_image = Image.open(args.image).convert("RGB")

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    results = []
    output_hashes: dict[str, str] = {}
    baseline_hash: str | None = None
    baseline_summary = None
    for variant in VARIANTS:
        image = resize_to_longest_edge(source_image, variant["longest_edge"])
        warmup_samples = []
        measured_samples = []
        total = int(variant["warmups"]) + args.runs
        for index in range(total):
            sample = run_sample(
                model,
                processor,
                image,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                cache_implementation=str(variant["cache"]),
                min_pixels=None,
                max_pixels=None,
                disable_compile=False,
            )
            target = warmup_samples if index < int(variant["warmups"]) else measured_samples
            target.append(sample)
            print(
                f"{variant['id']} {index + 1}/{total} "
                f"ttft={sample['ttft_ms']:.1f}ms "
                f"e2e={sample['e2e_ms']:.1f}ms "
                f"tok/s={sample['output_tokens_per_second']}",
                flush=True,
            )
        summary = aggregate(measured_samples)
        if baseline_hash is None:
            baseline_hash = measured_samples[0]["output_sha256"]
            baseline_summary = summary
        output_hashes[str(variant["id"])] = measured_samples[0]["output_sha256"]
        rubric_results = [
            evaluate_concepts(sample["output"], DEFAULT_CONCEPT_GROUPS)
            for sample in measured_samples
        ]
        exact_reference = variant.get("exact_reference")
        exact_hash = (
            output_hashes[str(exact_reference)] if exact_reference is not None else None
        )
        exact_passed = (
            all(sample["output_sha256"] == exact_hash for sample in measured_samples)
            if exact_hash is not None
            else None
        )
        rubric_passed = all(result["passed"] for result in rubric_results)
        speedup = {
            "ttft": round(
                baseline_summary["ttft_ms"]["p50"] / summary["ttft_ms"]["p50"], 3
            ),
            "e2e": round(
                baseline_summary["e2e_ms"]["p50"] / summary["e2e_ms"]["p50"], 3
            ),
            "throughput": round(
                summary["output_tokens_per_second_mean"]
                / baseline_summary["output_tokens_per_second_mean"],
                3,
            ),
        }
        results.append(
            {
                **variant,
                "processed_width": image.width,
                "processed_height": image.height,
                "quality_gate": {
                    "method": "required visual concepts"
                    + (
                        f" plus byte-identical output against variant {exact_reference}"
                        if exact_reference is not None
                        else ""
                    ),
                    "passed": rubric_passed and exact_passed is not False,
                    "concepts": rubric_results[0],
                    "exact_output_reference": exact_reference,
                    "exact_output_passed": exact_passed,
                    "exact_output_vs_baseline": all(
                        sample["output_sha256"] == baseline_hash
                        for sample in measured_samples
                    ),
                },
                "speedup_vs_baseline": speedup,
                "summary": summary,
                "warmup_samples": warmup_samples,
                "samples": measured_samples,
            }
        )

    artifact = {
        "schema": "comfyui-vlm/optimization-matrix",
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "media": {
            "path": str(args.image.resolve()),
            "source_width": source_image.width,
            "source_height": source_image.height,
        },
        "prompt": args.prompt,
        "model_load_seconds": round(load_seconds, 3),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "transformers": __import__("transformers").__version__,
        },
        "runs_per_variant": args.runs,
        "variants": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{v["id"]: v["summary"]} for v in results], indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
