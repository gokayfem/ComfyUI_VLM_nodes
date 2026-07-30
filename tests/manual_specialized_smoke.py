"""Opt-in real-weight smoke tests for specialized model backends.

Each invocation downloads and runs one real checkpoint. Keeping one model per
process verifies teardown and prevents one backend's CUDA state from masking
another backend's behavior.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from _bootstrap import bootstrap

bootstrap()

BACKENDS = (
    "florence-base",
    "florence-large",
    "moondream2",
    "qwen2vl-2b",
    "qwen2vl-2b-video",
    "qwen2vl-7b-4bit",
    "molmo-1b",
    "molmo-7b-d-4bit",
    "molmo-7b-o-4bit",
    "kosmos2",
    "uform",
    "mcllava",
    "joytag",
    "paligemma-caption",
    "minicpm-gguf-q4",
    "audioldm2",
)


def test_image() -> torch.Tensor:
    image = torch.zeros((1, 192, 256, 3), dtype=torch.float32)
    image[:, 48:144, 56:200, 0] = 1.0
    return image


def test_video() -> torch.Tensor:
    frames = torch.zeros((4, 96, 128, 3), dtype=torch.float32)
    for index in range(4):
        left = 12 + index * 18
        frames[index, 30:66, left : left + 24, 1] = 1.0
    return frames


def _run(backend: str):
    image = test_image()
    prompt = "What color is the large rectangle? Answer briefly."

    if backend.startswith("florence-"):
        from ComfyUI_VLM_nodes.nodes.florence2 import FlorencePredictor
        from ComfyUI_VLM_nodes.nodes.runtime import tensor_batch_to_pil

        label = {
            "florence-base": "Florence-2 base FT (fast)",
            "florence-large": "Florence-2 large FT (recommended)",
        }[backend]
        predictor = FlorencePredictor(label)
        try:
            raw, parsed = predictor.run(
                tensor_batch_to_pil(image)[0],
                "<MORE_DETAILED_CAPTION>",
                "",
                96,
                3,
            )
            return {"response": raw, "parsed": parsed}
        finally:
            predictor.close()

    if backend == "moondream2":
        from ComfyUI_VLM_nodes.nodes.moondream2 import Moondream2Predictor

        predictor = Moondream2Predictor()
        try:
            return {"response": predictor.generate(image, prompt)}
        finally:
            predictor.close()

    if backend.startswith("qwen2vl-"):
        from ComfyUI_VLM_nodes.nodes.qwen2vl import Qwen2VLPredictor

        model_name, memory_mode = {
            "qwen2vl-2b": ("Qwen2-VL-2B", "ComfyUI managed (BF16)"),
            "qwen2vl-2b-video": (
                "Qwen2-VL-2B",
                "ComfyUI managed (BF16)",
            ),
            "qwen2vl-7b-4bit": ("Qwen2-VL-7B", "Maximum Savings (4-bit)"),
        }[backend]
        predictor = Qwen2VLPredictor(
            model_name,
            memory_mode,
            "Auto (SDPA)",
            256 * 28 * 28,
            1280 * 28 * 28,
        )
        try:
            if backend.endswith("-video"):
                return {
                    "response": predictor.generate_video(
                        None,
                        test_video(),
                        (
                            "What color object moves horizontally? Answer with "
                            "the color and shape."
                        ),
                        48,
                        0.0,
                        0.9,
                        2.0,
                    )
                }
            return {
                "response": predictor.generate_images(
                    image, prompt, 48, 0.0, 0.9
                )
            }
        finally:
            predictor.close()

    if backend.startswith("molmo-"):
        from ComfyUI_VLM_nodes.nodes.molmo import MolmoPredictor
        from ComfyUI_VLM_nodes.nodes.runtime import tensor_batch_to_pil

        model_name, memory_mode = {
            "molmo-1b": (
                "MolmoE-1B (Efficient)",
                "Full Precision (45GB+ Required)",
            ),
            "molmo-7b-d-4bit": (
                "Molmo-7B-D (Best 7B)",
                "4-bit Quantized (15GB+ Required)",
            ),
            "molmo-7b-o-4bit": (
                "Molmo-7B-O (Alternative 7B)",
                "4-bit Quantized (15GB+ Required)",
            ),
        }[backend]
        predictor = MolmoPredictor(model_name, memory_mode, True)
        try:
            response = predictor.generate(
                tensor_batch_to_pil(image)[0], prompt, 48, 0.0, 0.9, 20
            )
            return {"response": response}
        finally:
            predictor.close()

    if backend == "kosmos2":
        from ComfyUI_VLM_nodes.nodes.kosmos2 import KosmosModelPredictor

        predictor = KosmosModelPredictor()
        try:
            return {"response": predictor.generate(image, prompt, 48)}
        finally:
            predictor.close()

    if backend == "uform":
        from ComfyUI_VLM_nodes.nodes.uform import UformGen2QwenChat

        predictor = UformGen2QwenChat()
        try:
            return {"response": predictor.chat(image, prompt, 48)}
        finally:
            predictor.close()

    if backend == "mcllava":
        from ComfyUI_VLM_nodes.nodes.mcllava import MCLLaVAModelPredictor

        predictor = MCLLaVAModelPredictor()
        try:
            return {
                "response": predictor.generate(
                    image, prompt, 0.0, 0.9, 4, 728, 48
                )
            }
        finally:
            predictor.close()

    if backend == "joytag":
        from ComfyUI_VLM_nodes.nodes.joytag import JoyTagPredictor

        predictor = JoyTagPredictor()
        try:
            return {"response": predictor.predict(image, 10, 0.1)}
        finally:
            predictor.close()

    if backend == "paligemma-caption":
        from ComfyUI_VLM_nodes.nodes.paligemma import (
            PALIGEMMA_MODELS,
            PaliPredictor,
        )
        from ComfyUI_VLM_nodes.nodes.runtime import tensor_batch_to_pil

        predictor = PaliPredictor(PALIGEMMA_MODELS[0], "bfloat16", "None")
        try:
            return {
                "response": predictor.generate(
                    tensor_batch_to_pil(image)[0],
                    "caption en",
                    max_new_tokens=64,
                    do_sample=False,
                )
            }
        finally:
            predictor.close()

    if backend == "minicpm-gguf-q4":
        from ComfyUI_VLM_nodes.nodes.minicpm import MiniCPMPredictor

        predictor = MiniCPMPredictor("Q4_K_M (4.7GB)", 4096, -1, 8)
        try:
            return {
                "response": predictor.generate(
                    image, prompt, 0.0, 0.9, 40, 1.05, 48
                )
            }
        finally:
            predictor.close()

    if backend == "audioldm2":
        from ComfyUI_VLM_nodes.nodes.audioldm2 import AudioLDM2Predictor

        predictor = AudioLDM2Predictor(cpu_offload=True)
        try:
            audio, sample_rate = predictor.generate(
                "a short clean bell chime",
                "",
                1.0,
                2.5,
                123,
                1,
                2,
            )
            return {
                "response": f"audio {audio.shape}",
                "sample_rate": sample_rate,
                "finite": bool(torch.isfinite(torch.from_numpy(audio)).all()),
            }
        finally:
            predictor.close()

    raise AssertionError(f"Unhandled backend: {backend}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=BACKENDS)
    args = parser.parse_args()

    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        free_before, total = torch.cuda.mem_get_info()
    else:
        free_before = total = 0

    result = _run(args.backend)
    response = str(result.get("response", ""))
    if not response.strip():
        raise RuntimeError("The model returned an empty response.")
    expected = "green" if args.backend.endswith("-video") else "red"
    if args.backend != "audioldm2" and expected not in response.lower():
        raise RuntimeError(
            f"The model did not identify the {expected} test object: {response}"
        )
    if args.backend == "audioldm2" and not result["finite"]:
        raise RuntimeError("AudioLDM2 returned non-finite samples.")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
        free_after, _ = torch.cuda.mem_get_info()
    else:
        peak = free_after = 0
    result.update(
        backend=args.backend,
        seconds=round(time.perf_counter() - started, 2),
        cuda_total_gib=round(total / 1024**3, 2),
        cuda_free_before_gib=round(free_before / 1024**3, 2),
        cuda_free_after_gib=round(free_after / 1024**3, 2),
        cuda_peak_allocated_gib=round(peak / 1024**3, 2),
    )
    print("SPECIALIZED_SMOKE_JSON=" + json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
