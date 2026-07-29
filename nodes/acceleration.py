"""Model-agnostic acceleration utilities for image and video VLM workflows.

These nodes reduce visual work *before* it reaches a model. They are therefore
portable across Transformers, llama.cpp, Photon, hosted APIs, CUDA, ROCm, MPS,
XPU, and CPU runtimes. No model is downloaded and no global PyTorch setting is
changed when this module is imported or executed.
"""

from __future__ import annotations

import json
import math
from typing import Any

import torch
import torch.nn.functional as functional


RESIZE_QUALITY = (
    "Fast (area)",
    "Quality (bicubic)",
)
PERFORMANCE_PROFILES = {
    "Live / robotics": {
        "max_frames": 24,
        "max_megapixels": 0.5,
        "max_edge": 896,
        "batch_size": 8,
        "unload_after": False,
    },
    "Fast video": {
        "max_frames": 48,
        "max_megapixels": 0.75,
        "max_edge": 1024,
        "batch_size": 8,
        "unload_after": False,
    },
    "Balanced": {
        "max_frames": 64,
        "max_megapixels": 1.0,
        "max_edge": 1344,
        "batch_size": 4,
        "unload_after": False,
    },
    "High detail": {
        "max_frames": 96,
        "max_megapixels": 2.0,
        "max_edge": 2048,
        "batch_size": 2,
        "unload_after": False,
    },
    "Low VRAM handoff": {
        "max_frames": 32,
        "max_megapixels": 0.75,
        "max_edge": 1024,
        "batch_size": 1,
        "unload_after": True,
    },
}


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )


def _validate_image_batch(images: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a ComfyUI IMAGE tensor.")
    single = images.ndim == 3
    value = images.unsqueeze(0) if single else images
    if value.ndim != 4:
        raise ValueError(
            f"Expected an HWC/BHWC or CHW/BCHW IMAGE tensor, got {tuple(images.shape)}."
        )
    if value.shape[-1] in (1, 3, 4):
        return value, single
    if value.shape[1] in (1, 3, 4):
        return value.permute(0, 2, 3, 1), single
    raise ValueError(f"Unsupported image channel shape: {tuple(images.shape)}.")


def optimize_image_pixels(
    images: torch.Tensor,
    *,
    max_megapixels: float,
    max_edge: int,
    multiple: int,
    resize_quality: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Downscale a batch once to a bounded visual-token pixel budget."""

    value, single = _validate_image_batch(images)
    if not math.isfinite(float(max_megapixels)) or max_megapixels <= 0:
        raise ValueError("max_megapixels must be finite and positive.")
    if not isinstance(max_edge, int) or max_edge < 32:
        raise ValueError("max_edge must be at least 32 pixels.")
    if multiple not in {1, 14, 28, 32}:
        raise ValueError("multiple must be one of 1, 14, 28, or 32.")
    if resize_quality not in RESIZE_QUALITY:
        raise ValueError(f"Unknown resize quality {resize_quality!r}.")

    height, width = int(value.shape[1]), int(value.shape[2])
    pixel_budget = float(max_megapixels) * 1_000_000
    scale = min(
        1.0,
        float(max_edge) / max(width, height),
        math.sqrt(pixel_budget / (width * height)),
    )

    def bounded_dimension(dimension: int) -> int:
        target = max(1, math.floor(dimension * scale))
        if multiple == 1 or target < multiple:
            return target
        return max(multiple, (target // multiple) * multiple)

    output_width = bounded_dimension(width)
    output_height = bounded_dimension(height)
    output = value
    resized_image = (output_height, output_width) != (height, width)
    if resized_image:
        nchw = value.permute(0, 3, 1, 2)
        if resize_quality == "Fast (area)":
            resized = functional.interpolate(
                nchw,
                size=(output_height, output_width),
                mode="area",
            )
        else:
            resized = functional.interpolate(
                nchw,
                size=(output_height, output_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        output = resized.permute(0, 2, 3, 1).clamp(0.0, 1.0)
    report = {
        "frames": int(value.shape[0]),
        "input_width": width,
        "input_height": height,
        "output_width": output_width,
        "output_height": output_height,
        "input_pixels_per_frame": width * height,
        "output_pixels_per_frame": output_width * output_height,
        "visual_work_reduction": (
            (width * height) / max(1, output_width * output_height)
        ),
        "resized": resized_image,
        "multiple": multiple,
        "quality": resize_quality,
    }
    if not resized_image:
        return images, report
    return (output[0] if single else output), report


class VLMPerformanceProfile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "profile": (
                    tuple(PERFORMANCE_PROFILES),
                    {"default": "Balanced"},
                )
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = (
        "max_frames",
        "max_megapixels",
        "max_edge",
        "batch_size",
        "unload_after",
        "profile_json",
    )
    FUNCTION = "profile"
    CATEGORY = "VLM Nodes/Performance"
    DESCRIPTION = (
        "Portable speed/quality presets for the sampler, pixel optimizer, "
        "and VLM batch inputs. The profile never changes global runtime state."
    )

    def profile(self, profile):
        values = dict(PERFORMANCE_PROFILES[profile])
        values["profile"] = profile
        return (
            values["max_frames"],
            values["max_megapixels"],
            values["max_edge"],
            values["batch_size"],
            values["unload_after"],
            _json(values),
        )


class VLMImagePixelBudget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 64.0, "step": 0.05},
                ),
                "max_edge": (
                    "INT",
                    {"default": 1344, "min": 32, "max": 16384, "step": 14},
                ),
                "multiple": (
                    ("1", "14", "28", "32"),
                    {
                        "default": "14",
                        "tooltip": (
                            "14/28 suit common VLM vision patches; 32 suits "
                            "many detector backbones. Use 1 for arbitrary sizes."
                        ),
                    },
                ),
                "resize_quality": (
                    RESIZE_QUALITY,
                    {"default": "Fast (area)"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "optimized_images",
        "width",
        "height",
        "optimization_report",
    )
    FUNCTION = "optimize"
    CATEGORY = "VLM Nodes/Performance"
    DESCRIPTION = (
        "Apply one portable pixel budget before any VLM, avoiding repeated "
        "high-resolution visual-token work while preserving aspect ratio."
    )

    def optimize(
        self,
        images,
        max_megapixels,
        max_edge,
        multiple,
        resize_quality,
    ):
        output, report = optimize_image_pixels(
            images,
            max_megapixels=float(max_megapixels),
            max_edge=int(max_edge),
            multiple=int(multiple),
            resize_quality=resize_quality,
        )
        return (
            output,
            report["output_width"],
            report["output_height"],
            _json(report),
        )


NODE_CLASS_MAPPINGS = {
    "VLMPerformanceProfile": VLMPerformanceProfile,
    "VLMImagePixelBudget": VLMImagePixelBudget,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMPerformanceProfile": "VLM Performance Profile",
    "VLMImagePixelBudget": "VLM Image Pixel Budget",
}
