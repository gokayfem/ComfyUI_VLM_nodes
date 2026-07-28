"""MC-LLaVA node with in-memory images and ComfyUI-managed weights."""

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

MODEL_ID = "visheratin/MC-LLaVA-3b"


class MCLLaVAModelPredictor:
    def __init__(self):
        transformers = require_module("transformers")
        model_path = snapshot_download(
            MODEL_ID, "mcllava", ignore_patterns=["*.bin"]
        )
        self.dtype = torch_dtype("float16")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).eval()
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.handle = ManagedTorchModel(model, processor=self.processor)

    def close(self):
        self.handle.close()
        self.processor = None

    def generate(
        self,
        images,
        prompt,
        temperature,
        top_p,
        max_crops,
        num_tokens,
        max_new_tokens,
    ):
        results = []
        formatted = (
            "<|im_start|>user\n<image>\n"
            f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        for image in tensor_batch_to_pil(images):
            model = self.handle.ensure_loaded()
            device = model_device(model)
            inputs = self.processor(
                formatted,
                [image],
                model,
                max_crops=int(max_crops),
                num_tokens=int(num_tokens),
            )
            inputs = move_inputs(inputs, device)
            do_sample = float(temperature) > 0.0
            generation = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": do_sample,
                "use_cache": True,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
            }
            if do_sample:
                generation.update(
                    temperature=float(temperature), top_p=float(top_p)
                )
            with torch.inference_mode(), inference_context(device, self.dtype):
                output = model.generate(**inputs, **generation)
            input_length = inputs["input_ids"].shape[-1]
            text = self.processor.tokenizer.decode(
                output[0, input_length:], skip_special_tokens=True
            )
            results.append(text.strip())
        return batch_text(results)


class MCLLaVAModel(CachedModelNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "Describe the image."},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "max_crops": (
                    "INT",
                    {"default": 100, "min": 1, "max": 300, "step": 1},
                ),
                "num_tokens": (
                    "INT",
                    {"default": 728, "min": 1, "max": 4096, "step": 1},
                ),
            },
            "optional": {
                "max_new_tokens": (
                    "INT",
                    {"default": 200, "min": 1, "max": 4096},
                ),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_image_description"
    CATEGORY = "VLM Nodes/MC-LLaVA"

    def generate_image_description(
        self,
        image,
        prompt,
        temperature,
        top_p,
        max_crops,
        num_tokens,
        max_new_tokens=200,
        unload_after=False,
    ):
        predictor = self.get_or_create_model(
            MODEL_ID, MCLLaVAModelPredictor
        )
        try:
            return (
                predictor.generate(
                    image,
                    prompt,
                    temperature,
                    top_p,
                    max_crops,
                    num_tokens,
                    max_new_tokens,
                ),
            )
        finally:
            self.maybe_clear_model(unload_after)


NODE_CLASS_MAPPINGS = {"MCLLaVAModel": MCLLaVAModel}
NODE_DISPLAY_NAME_MAPPINGS = {"MCLLaVAModel": "MC-LLaVA"}
