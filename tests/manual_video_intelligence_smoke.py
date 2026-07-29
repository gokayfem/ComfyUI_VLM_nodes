"""Run adaptive temporal reasoning on a real local video and real VLM.

Example:
    python tests/manual_video_intelligence_smoke.py \
        /mnt/d/002.mp4 \
        --model "Qwen 3 VL 2B Instruct" \
        --output /mnt/d/comfyui-repair/video-intelligence-audit/result.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import av
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if REPOSITORY.name != "ComfyUI_VLM_nodes":
    specification = importlib.util.spec_from_file_location(
        "ComfyUI_VLM_nodes",
        REPOSITORY / "__init__.py",
        submodule_search_locations=[str(REPOSITORY)],
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load package from {REPOSITORY}.")
    package = importlib.util.module_from_spec(specification)
    sys.modules["ComfyUI_VLM_nodes"] = package
    specification.loader.exec_module(package)

from ComfyUI_VLM_nodes.nodes.modern_vlm import ModernVLMPredictor
from ComfyUI_VLM_nodes.nodes.video_intelligence import (
    build_video_reasoning_prompt,
    parse_video_reasoning_output,
    resize_video_for_analysis,
    sample_video_frames,
)


def load_video(path: Path) -> tuple[torch.Tensor, float]:
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if rate is None:
            raise RuntimeError("The video does not report a frame rate.")
        frames = [
            torch.from_numpy(frame.to_ndarray(format="rgb24")).to(torch.float32)
            / 255.0
            for frame in container.decode(stream)
        ]
    finally:
        container.close()
    if not frames:
        raise RuntimeError("The video contains no decodable frames.")
    return torch.stack(frames), float(rate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--model",
        default="Qwen 3 VL 2B Instruct",
    )
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--analysis-max-side", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frames, fps = load_video(args.video)
    sampled, selection, diagnostics = sample_video_frames(
        frames,
        fps=fps,
        max_frames=args.max_frames,
        strategy="Hybrid: scene + motion + tracks",
        minimum_gap_seconds=0.2,
    )
    prompt = build_video_reasoning_prompt(
        selection,
        task="Detailed temporal summary",
        question="What happens, and how do the people behave over time?",
        max_events=12,
    )
    analysis_frames = resize_video_for_analysis(
        sampled,
        max_side=args.analysis_max_side,
    )
    predictor = ModernVLMPredictor(
        args.model,
        "",
        "ComfyUI managed (BF16)",
        "Auto (SDPA)",
    )
    started = time.perf_counter()
    try:
        raw = predictor.generate(
            images=None,
            prompt=prompt,
            system_prompt=(
                "You are a precise temporal video analyst. Return one JSON "
                "object that obeys the supplied schema."
            ),
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            video_frames=analysis_frames,
            fps=fps,
            video_selection=selection,
        )
    finally:
        predictor.close()
    reasoning_seconds = time.perf_counter() - started
    result = {
        "video": str(args.video),
        "model": args.model,
        "source_shape": list(frames.shape),
        "fps": fps,
        "selection": selection.to_dict(),
        "sampling": diagnostics,
        "analysis_shape": list(analysis_frames.shape),
        "reasoning_seconds": reasoning_seconds,
        "raw_response": raw,
        "cuda_peak_gib": (
            torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available()
            else 0.0
        ),
    }
    try:
        summary, events, normalized = parse_video_reasoning_output(raw, selection)
    except (TypeError, ValueError) as exc:
        result["structured_output_valid"] = False
        result["structured_output_error"] = str(exc)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        raise
    result.update(
        {
            "structured_output_valid": True,
            "summary": summary,
            "events": events.to_dict(),
            "normalized_response": json.loads(normalized),
        }
    )
    encoded = json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
