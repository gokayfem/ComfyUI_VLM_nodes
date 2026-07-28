"""UForm Gen2 Qwen node with safe lazy loading."""

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

MODEL_ID = "unum-cloud/uform-gen2-qwen-500m"


class UformGen2QwenChat:
    def __init__(self):
        transformers = require_module("transformers")
        model_path = snapshot_download(
            MODEL_ID, "uform-gen2-qwen", ignore_patterns=["*.bin"]
        )
        self.dtype = torch_dtype("float16")
        model = transformers.AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).eval()
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.handle = ManagedTorchModel(model, processor=self.processor)

    def close(self):
        self.handle.close()
        self.processor = None

    def chat(self, images, question, max_new_tokens):
        results = []
        for image in tensor_batch_to_pil(images):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"<image>{question}"},
            ]
            input_ids = self.processor.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            image_tensor = self.processor.feature_extractor(image).unsqueeze(0)
            attention_mask = torch.ones(
                1,
                input_ids.shape[1] + self.processor.num_image_latents - 1,
                dtype=torch.long,
            )
            model = self.handle.ensure_loaded()
            device = model_device(model)
            model_inputs = {
                "input_ids": input_ids.to(device),
                "images": image_tensor.to(device),
                "attention_mask": attention_mask.to(device),
            }
            with torch.inference_mode(), inference_context(device, self.dtype):
                output = model.generate(
                    **model_inputs,
                    max_new_tokens=int(max_new_tokens),
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            generated = output[0, input_ids.shape[-1] :]
            results.append(
                self.processor.tokenizer.decode(
                    generated, skip_special_tokens=True
                ).strip()
            )
        return batch_text(results)


class UformGen2QwenNode(CachedModelNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe this image in detail.",
                    },
                ),
            },
            "optional": {
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 1, "max": 4096},
                ),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "uform_gen2_qwen_chat"
    CATEGORY = "VLM Nodes/UformGen2Qwen"

    def uform_gen2_qwen_chat(
        self, image, question, max_new_tokens=512, unload_after=False
    ):
        predictor = self.get_or_create_model(
            MODEL_ID, UformGen2QwenChat
        )
        try:
            return (predictor.chat(image, question, max_new_tokens),)
        finally:
            self.maybe_clear_model(unload_after)


NODE_CLASS_MAPPINGS = {"UformGen2QwenNode": UformGen2QwenNode}
NODE_DISPLAY_NAME_MAPPINGS = {"UformGen2QwenNode": "UForm Gen2 Qwen"}
