"""Current Moondream 2 node using the model's supported query API.

The pinned checkpoint was authored against Transformers 4.52.4. Loading it
through Transformers 5's ``from_pretrained`` compatibility path can silently
produce an all-EOS model even when every tensor is reported as loaded. The
checkpoint itself is a normal safetensors state dict, so instantiate its
official wrapper and load that state dict directly. This keeps Moondream in
ComfyUI's managed VRAM lifecycle without downgrading Transformers for the rest
of the node pack.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import torch

from .runtime import (
    CachedModelNode,
    ManagedTorchModel,
    batch_text,
    inference_context,
    model_device,
    require_module,
    snapshot_download,
    tensor_batch_to_pil,
    torch_dtype,
)

MODEL_ID = "vikhyatk/moondream2"
MODEL_REVISION = "2025-06-21"
_CHECKPOINT_PACKAGE = "_comfyui_vlm_moondream2_checkpoint"


def _checkpoint_module(model_path: str | Path):
    """Import the checkpoint's relative modules without HF's generated cache.

    Hugging Face's dynamic-module cache can omit transitive relative imports
    for a local snapshot. Giving the snapshot a private package namespace lets
    Python resolve the checkpoint's own ``.config``, ``.vision``, and related
    modules directly and deterministically.
    """

    source = str(Path(model_path).resolve())
    package = sys.modules.get(_CHECKPOINT_PACKAGE)
    if package is None:
        package = ModuleType(_CHECKPOINT_PACKAGE)
        package.__path__ = [source]
        package.__package__ = _CHECKPOINT_PACKAGE
        sys.modules[_CHECKPOINT_PACKAGE] = package
    elif list(getattr(package, "__path__", ())) != [source]:
        raise RuntimeError(
            "Moondream2 checkpoint source changed inside a running process. "
            "Restart ComfyUI before loading a different snapshot."
        )
    return importlib.import_module(f"{_CHECKPOINT_PACKAGE}.hf_moondream")


def _load_native_checkpoint(model_path: str | Path):
    checkpoint = _checkpoint_module(model_path)
    safetensors = require_module("safetensors.torch")
    config = checkpoint.HfConfig.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = checkpoint.HfMoondream(config)
    weights = Path(model_path) / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Moondream2 weights are missing: {weights}")
    missing, unexpected = safetensors.load_model(
        model,
        str(weights),
        strict=True,
    )
    if missing or unexpected:
        raise RuntimeError(
            "Moondream2 checkpoint did not load exactly: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return model.eval()


class Moondream2Predictor:
    def __init__(self):
        model_path = snapshot_download(
            MODEL_ID,
            "moondream2",
            revision=MODEL_REVISION,
            ignore_patterns=["*.bin", "*.gguf"],
        )
        self.dtype = torch_dtype("bfloat16")
        model = _load_native_checkpoint(model_path)
        self.handle = ManagedTorchModel(model)

    def close(self):
        self.handle.close()

    def generate(
        self,
        images,
        question,
        max_tokens=256,
        temperature=0.0,
        top_p=0.3,
        reasoning=False,
    ):
        results = []
        for image in tensor_batch_to_pil(images):
            model = self.handle.ensure_loaded()
            device = model_device(model)
            with torch.inference_mode(), inference_context(device, self.dtype):
                response = model.query(
                    image,
                    question,
                    reasoning=bool(reasoning),
                    settings={
                        "max_tokens": int(max_tokens),
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        # Moondream's encoder indexes this optional key
                        # directly; None selects the base checkpoint.
                        "variant": None,
                    },
                )
            if isinstance(response, dict):
                response = response.get("answer", response)
            if not str(response).strip():
                raise RuntimeError(
                    "Moondream2 returned an empty response. Verify that the "
                    f"{MODEL_REVISION} snapshot is complete, then restart "
                    "ComfyUI so its checkpoint modules are reloaded."
                )
            results.append(str(response))
        return batch_text(results)


class Moondream2model(CachedModelNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text_input": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe this image in detail.",
                    },
                ),
            },
            "optional": {
                "max_tokens": (
                    "INT",
                    {"default": 256, "min": 1, "max": 2048},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "reasoning": ("BOOLEAN", {"default": False}),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "moondream2_generate_predictions"
    CATEGORY = "VLM Nodes/Modern/Edge"

    def moondream2_generate_predictions(
        self,
        image,
        text_input,
        max_tokens=256,
        temperature=0.0,
        top_p=0.3,
        reasoning=False,
        unload_after=False,
    ):
        predictor = self.get_or_create_model(
            (MODEL_ID, MODEL_REVISION), Moondream2Predictor
        )
        try:
            return (
                predictor.generate(
                    image,
                    text_input,
                    max_tokens,
                    temperature,
                    top_p,
                    reasoning,
                ),
            )
        finally:
            self.maybe_clear_model(unload_after)


NODE_CLASS_MAPPINGS = {"Moondream2model": Moondream2model}
NODE_DISPLAY_NAME_MAPPINGS = {"Moondream2model": "Moondream 2"}
