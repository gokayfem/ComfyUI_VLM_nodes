"""Kosmos-2 grounding/caption node with lazy, Comfy-managed loading."""

from __future__ import annotations

import torch

from .runtime import (
    CachedModelNode,
    ManagedTorchModel,
    batch_text,
    inference_context,
    model_device,
    move_inputs,
    require_module,
    snapshot_download,
    tensor_batch_to_pil,
    torch_dtype,
)

MODEL_ID = "microsoft/kosmos-2-patch14-224"


class KosmosModelPredictor:
    def __init__(self):
        transformers = require_module("transformers")
        model_path = snapshot_download(
            MODEL_ID, "kosmos2", ignore_patterns=["*.bin"]
        )
        self.dtype = torch_dtype("bfloat16")
        model_class = getattr(
            transformers,
            "Kosmos2ForConditionalGeneration",
            getattr(transformers, "AutoModelForImageTextToText", None),
        )
        if model_class is None:
            raise RuntimeError(
                "This Transformers version does not include Kosmos-2 support."
            )
        model = model_class.from_pretrained(
            model_path, torch_dtype=self.dtype
        ).eval()
        self.processor = transformers.AutoProcessor.from_pretrained(model_path)
        self.handle = ManagedTorchModel(model, processor=self.processor)

    def close(self):
        self.handle.close()
        self.processor = None

    def generate(self, images, text, max_new_tokens):
        results = []
        for image in tensor_batch_to_pil(images):
            prompt = f"<grounding>{text.strip()}"
            inputs = self.processor(
                text=prompt, images=image, return_tensors="pt"
            )
            model = self.handle.ensure_loaded()
            device = model_device(model)
            inputs = move_inputs(inputs, device)
            with torch.inference_mode(), inference_context(device, self.dtype):
                output = model.generate(
                    **inputs,
                    use_cache=True,
                    max_new_tokens=int(max_new_tokens),
                )
            decoded = self.processor.batch_decode(
                output, skip_special_tokens=True
            )[0]
            post_process = getattr(
                self.processor, "post_process_generation", None
            )
            if callable(post_process):
                processed, _entities = post_process(decoded)
            else:
                processed = decoded
            if processed.startswith(text):
                processed = processed[len(text) :].lstrip(": \n")
            results.append(processed.strip())
        return batch_text(results)


class Kosmos2model(CachedModelNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text_input": (
                    "STRING",
                    {"multiline": True, "default": "Describe the image."},
                ),
            },
            "optional": {
                "max_new_tokens": (
                    "INT",
                    {"default": 128, "min": 1, "max": 2048},
                ),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "new_model_generate_predictions"
    CATEGORY = "VLM Nodes/Kosmos-2"

    def new_model_generate_predictions(
        self,
        image,
        text_input,
        max_new_tokens=128,
        unload_after=False,
    ):
        predictor = self.get_or_create_model(
            MODEL_ID, KosmosModelPredictor
        )
        try:
            return (
                predictor.generate(image, text_input, max_new_tokens),
            )
        finally:
            self.maybe_clear_model(unload_after)


NODE_CLASS_MAPPINGS = {"Kosmos2model": Kosmos2model}
NODE_DISPLAY_NAME_MAPPINGS = {"Kosmos2model": "Kosmos-2"}
