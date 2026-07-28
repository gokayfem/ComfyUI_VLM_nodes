"""Current Moondream 2 node using the model's supported query API."""

from __future__ import annotations

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


class Moondream2Predictor:
    def __init__(self):
        transformers = require_module("transformers")
        model_path = snapshot_download(
            MODEL_ID,
            "moondream2",
            revision=MODEL_REVISION,
            ignore_patterns=["*.bin", "*.gguf"],
        )
        self.dtype = torch_dtype("bfloat16")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).eval()
        self.handle = ManagedTorchModel(model)

    def close(self):
        self.handle.close()

    def generate(self, images, question):
        results = []
        for image in tensor_batch_to_pil(images):
            model = self.handle.ensure_loaded()
            device = model_device(model)
            with torch.inference_mode(), inference_context(device, self.dtype):
                response = model.query(image, question)
            if isinstance(response, dict):
                response = response.get("answer", response)
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
                "unload_after": ("BOOLEAN", {"default": False})
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "moondream2_generate_predictions"
    CATEGORY = "VLM Nodes/Moondream2"

    def moondream2_generate_predictions(
        self, image, text_input, unload_after=False
    ):
        predictor = self.get_or_create_model(
            (MODEL_ID, MODEL_REVISION), Moondream2Predictor
        )
        try:
            return (predictor.generate(image, text_input),)
        finally:
            self.maybe_clear_model(unload_after)


NODE_CLASS_MAPPINGS = {"Moondream2model": Moondream2model}
NODE_DISPLAY_NAME_MAPPINGS = {"Moondream2model": "Moondream 2"}
