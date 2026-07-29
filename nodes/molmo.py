"""AllenAI Molmo nodes with batch support and deterministic model ownership."""

from __future__ import annotations

from typing import Any

import torch

from .runtime import (
    CachedModelNode,
    ExternalTorchModel,
    ManagedTorchModel,
    batch_text,
    external_device_map,
    inference_context,
    model_device,
    require_quantization_backend,
    require_module,
    reserve_external_vram,
    snapshot_download,
    tensor_batch_to_pil,
    torch_dtype,
)

MEMORY_MODES = {
    "Full Precision (45GB+ Required)": "managed",
    "8-bit Quantized (25GB+ Required)": "8bit",
    "4-bit Quantized (15GB+ Required)": "4bit",
    "4-bit + CPU Offload (12GB+ Required)": "4bit-offload",
}
MOLMO_MODELS = {
    "MolmoE-1B (Efficient)": "allenai/MolmoE-1B-0924",
    "Molmo-7B-D (Best 7B)": "allenai/Molmo-7B-D-0924",
    "Molmo-7B-O (Alternative 7B)": "allenai/Molmo-7B-O-0924",
}


class MolmoPredictor:
    def __init__(self, model_name, memory_mode, use_autocast):
        transformers = require_module("transformers")
        repo_id = MOLMO_MODELS[model_name]
        mode = MEMORY_MODES[memory_mode]
        external = mode != "managed"
        if external:
            # Validate before downloading a multi-gigabyte checkpoint.
            require_quantization_backend(memory_mode)
        path = snapshot_download(
            repo_id,
            f"molmo/{repo_id.replace('/', '--')}",
            ignore_patterns=["*.bin"],
        )
        self.dtype = torch_dtype("bfloat16")
        self.use_autocast = bool(use_autocast)
        self.processor = transformers.AutoProcessor.from_pretrained(
            path, trust_remote_code=True
        )
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": self.dtype,
        }
        if external:
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_8bit=mode == "8bit",
                load_in_4bit=mode.startswith("4bit"),
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = external_device_map(
                allow_auto_offload=mode == "4bit-offload"
            )
            reserve_external_vram(
                (5 if "1B" in model_name else 12) * 1024**3
            )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path, **kwargs
        ).eval()
        self.handle = (
            ExternalTorchModel(model, processor=self.processor)
            if external
            else ManagedTorchModel(model, processor=self.processor)
        )

    def close(self):
        self.handle.close()
        self.processor = None

    def generate(self, image, prompt, max_new_tokens, temperature, top_p, top_k):
        model = self.handle.ensure_loaded()
        device = model_device(model)
        inputs = self.processor.process(images=[image], text=prompt)
        inputs = {
            key: value.to(device).unsqueeze(0)
            for key, value in inputs.items()
        }
        config = require_module("transformers").GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            do_sample=float(temperature) > 0,
            temperature=max(float(temperature), 1e-5),
            top_p=float(top_p),
            top_k=int(top_k),
            stop_strings="<|endoftext|>",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        context = (
            inference_context(device, self.dtype)
            if self.use_autocast
            else torch.no_grad()
        )
        with torch.inference_mode(), context:
            output = model.generate_from_batch(
                inputs, config, tokenizer=self.processor.tokenizer
            )
        return self.processor.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()


class MolmoNode(CachedModelNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "Describe this image in detail."},
                ),
                "model_name": (list(MOLMO_MODELS),),
                "memory_mode": (
                    list(MEMORY_MODES),
                    {"default": "4-bit Quantized (15GB+ Required)"},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 200, "min": 1, "max": 2048},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100}),
                "use_autocast": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "unload_after": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "VLM Nodes/Legacy/Model Loaders"

    def generate(
        self,
        image,
        prompt,
        model_name,
        memory_mode="4-bit Quantized (15GB+ Required)",
        max_new_tokens=200,
        temperature=0.2,
        top_p=0.9,
        top_k=50,
        use_autocast=True,
        unload_after=False,
    ):
        predictor = self.get_or_create_model(
            (model_name, memory_mode, bool(use_autocast)),
            lambda: MolmoPredictor(model_name, memory_mode, use_autocast),
        )
        try:
            return (
                batch_text(
                    predictor.generate(
                        pil,
                        prompt,
                        max_new_tokens,
                        temperature,
                        top_p,
                        top_k,
                    )
                    for pil in tensor_batch_to_pil(image)
                ),
            )
        finally:
            self.maybe_clear_model(unload_after)


NODE_CLASS_MAPPINGS = {"MolmoNode": MolmoNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MolmoNode": "Molmo Vision-Language Model"}
