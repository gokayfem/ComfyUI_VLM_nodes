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
        dynamic_modules = require_module("transformers.dynamic_module_utils")
        model_path = snapshot_download(
            MODEL_ID,
            "moondream2",
            revision=MODEL_REVISION,
            ignore_patterns=["*.bin", "*.gguf"],
        )
        self.dtype = torch_dtype("bfloat16")
        config = transformers.AutoConfig.from_pretrained(
            model_path,
            revision=MODEL_REVISION,
            trust_remote_code=True,
        )
        remote_class = dynamic_modules.get_class_from_dynamic_module(
            "hf_moondream.HfMoondream",
            model_path,
            local_files_only=True,
        )

        class Transformers5Moondream(remote_class):
            def __init__(self, model_config):
                super().__init__(model_config)
                # The pinned remote wrapper predates the Transformers 5 model
                # loader and does not declare its tied-weight metadata. Calling
                # the full post_init would reinitialize custom Moondream state.
                self.all_tied_weights_keys = {}

        model = Transformers5Moondream.from_pretrained(
            model_path,
            config=config,
            dtype=self.dtype,
        )
        model.eval()
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
                    "Moondream2 returned an empty response on this "
                    "Torch/Transformers build. Use the Modern VLM node with "
                    "LFM2.5-VL 450M, InternVL 3.5 1B, or Qwen3-VL 2B."
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
    CATEGORY = "VLM Nodes/Moondream2"

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
