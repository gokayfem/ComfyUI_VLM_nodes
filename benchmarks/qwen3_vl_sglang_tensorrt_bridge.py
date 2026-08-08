"""Inject a serialized TensorRT Qwen3-VL vision engine into SGLang.

The bridge is deliberately static-shape and quality-safe. Requests matching the
compiled 448px benchmark grid use TensorRT; every other shape takes SGLang's
unchanged native vision path.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger("sglang.tensorrt_bridge")
ENGINE_ENV = "QWEN3_VL_TRT_ENGINE"
EXPECTED_GRID = ((1, 18, 28),)


def _load_engine(path: Path) -> torch.nn.Module:
    # Importing Torch-TensorRT registers the serialized engine operators used by
    # the ExportedProgram.
    import torch_tensorrt  # noqa: F401

    started = time.perf_counter()
    engine = torch.export.load(path).module().cuda()
    LOGGER.info(
        "Loaded Qwen3-VL TensorRT vision engine path=%s elapsed=%.3fs",
        path,
        time.perf_counter() - started,
    )
    return engine


def _grid_tuple(grid: torch.Tensor) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in grid.cpu().tolist())


def install_bridge() -> bool:
    engine_value = os.environ.get(ENGINE_ENV)
    if not engine_value:
        return False
    engine_path = Path(engine_value).expanduser().resolve()
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT vision engine not found: {engine_path}")

    from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration

    if getattr(Qwen3VLForConditionalGeneration, "_trt_bridge_installed", False):
        return True

    native_get_image_feature = Qwen3VLForConditionalGeneration.get_image_feature

    def get_image_feature(self: Any, items: list[Any]) -> torch.Tensor:
        image_grid_thw = torch.concat(
            [item.image_grid_thw for item in items], dim=0
        )
        if _grid_tuple(image_grid_thw) != EXPECTED_GRID:
            self._trt_bridge_fallbacks = getattr(self, "_trt_bridge_fallbacks", 0) + 1
            return native_get_image_feature(self, items)

        engine = getattr(self, "_trt_vision_engine", None)
        if engine is None:
            engine = _load_engine(engine_path)
            self._trt_vision_engine = engine

        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            device="cuda", dtype=torch.bfloat16
        )
        outputs = engine(pixel_values.contiguous())
        # Output 0 is the unmerged vision state. SGLang consumes the merged
        # language embedding followed by all three packed deep-stack features.
        packed = torch.cat(tuple(outputs[1:]), dim=-1)
        if packed.shape != (126, 8192):
            raise RuntimeError(
                f"Unexpected TensorRT packed vision shape: {tuple(packed.shape)}"
            )
        self._trt_bridge_hits = getattr(self, "_trt_bridge_hits", 0) + 1
        return packed

    Qwen3VLForConditionalGeneration.get_image_feature = get_image_feature
    Qwen3VLForConditionalGeneration._trt_bridge_installed = True
    LOGGER.info(
        "Installed static Qwen3-VL TensorRT/SGLang bridge engine=%s grid=%s",
        engine_path,
        EXPECTED_GRID,
    )
    return True


install_bridge()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", type=Path)
    args = parser.parse_args()
    if args.smoke_test is None:
        return
    engine = _load_engine(args.smoke_test.resolve())
    sample = torch.zeros((504, 1536), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        outputs = engine(sample)
    torch.cuda.synchronize()
    print([list(output.shape) for output in outputs])


if __name__ == "__main__":
    main()
