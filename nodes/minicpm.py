"""MiniCPM-V 2.6 GGUF node using llama.cpp's native vision handler."""

from __future__ import annotations

from .runtime import (
    CachedModelNode,
    LlamaHandle,
    LlavaClipConfig,
    batch_text,
    default_llama_threads,
    hf_download,
    image_data_uri,
    llama_chat_content,
    llama_runtime_input_types,
    llama_runtime_options,
    tensor_batch_to_pil,
)

MODEL_REPO = "openbmb/MiniCPM-V-2_6-gguf"
GGUF_MODELS = {
    "Q2_K (3GB)": "ggml-model-Q2_K.gguf",
    "Q3_K (3.8GB)": "ggml-model-Q3_K.gguf",
    "Q4_K_M (4.7GB)": "ggml-model-Q4_K_M.gguf",
    "Q5_K_M (5.4GB)": "ggml-model-Q5_K_M.gguf",
    "Q8_0 (8.1GB)": "ggml-model-Q8_0.gguf",
    "F16 (15.2GB)": "ggml-model-f16.gguf",
}


class MiniCPMPredictor:
    def __init__(
        self,
        model_variant,
        context_length,
        gpu_layers,
        n_threads,
        runtime_options=None,
    ):
        model_path = hf_download(
            MODEL_REPO,
            GGUF_MODELS[model_variant],
            "minicpm-v-2_6-gguf",
        )
        projector_path = hf_download(
            MODEL_REPO,
            "mmproj-model-f16.gguf",
            "minicpm-v-2_6-gguf",
        )

        clip = LlavaClipConfig(projector_path, "MiniCPM-V 2.6")

        def create_handler(*, use_gpu=True):
            return clip.create(use_gpu=use_gpu)

        self.handle = LlamaHandle(
            model_path,
            n_ctx=int(context_length),
            n_gpu_layers=int(gpu_layers),
            n_threads=int(n_threads),
            chat_handler_factory=create_handler,
            projector_path=projector_path,
            **dict(runtime_options or {}),
        )

    def close(self):
        self.handle.close()

    def generate(
        self,
        images,
        prompt,
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        max_tokens,
    ):
        llm = self.handle.ensure_loaded()
        results = []
        for image in tensor_batch_to_pil(images):
            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_uri(image)},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repeat_penalty=float(repeat_penalty),
            )
            results.append(llama_chat_content(response))
        return batch_text(results)


class MiniCPMNode(CachedModelNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe this image in detail.",
                    },
                ),
                "model_variant": (list(GGUF_MODELS),),
                "context_length": (
                    "INT",
                    {"default": 4096, "min": 512, "max": 131072},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "top_k": (
                    "INT",
                    {"default": 100, "min": 0, "max": 1000},
                ),
                "repeat_penalty": (
                    "FLOAT",
                    {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
            },
            "optional": {
                "gpu_layers": (
                    "INT",
                    {"default": -1, "min": -1, "max": 1000},
                ),
                "n_threads": (
                    "INT",
                    {
                        "default": default_llama_threads(),
                        "min": 1,
                        "max": 256,
                    },
                ),
                "max_tokens": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192},
                ),
                "unload_after": ("BOOLEAN", {"default": False}),
                **llama_runtime_input_types(),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "VLM Nodes/Legacy/Model Loaders"

    def generate(
        self,
        image,
        prompt,
        model_variant,
        context_length=4096,
        temperature=0.2,
        top_p=0.8,
        top_k=100,
        repeat_penalty=1.05,
        gpu_layers=-1,
        n_threads=None,
        max_tokens=512,
        unload_after=False,
        n_batch=512,
        n_ubatch=512,
        flash_attention="Auto",
        use_mmap=True,
        split_mode="Layer",
        main_gpu=0,
        tensor_split="",
    ):
        n_threads = default_llama_threads() if n_threads is None else int(n_threads)
        options = llama_runtime_options(
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            flash_attention=flash_attention,
            use_mmap=use_mmap,
            split_mode=split_mode,
            main_gpu=main_gpu,
            tensor_split=tensor_split,
        )
        key = (
            model_variant,
            int(context_length),
            int(gpu_layers),
            int(n_threads),
            tuple(sorted(options.items())),
        )
        predictor = self.get_or_create_model(
            key,
            lambda: MiniCPMPredictor(
                model_variant,
                context_length,
                gpu_layers,
                n_threads,
                options,
            ),
        )
        try:
            return (
                predictor.generate(
                    image,
                    prompt,
                    temperature,
                    top_p,
                    top_k,
                    repeat_penalty,
                    max_tokens,
                ),
            )
        finally:
            self.maybe_clear_model(unload_after)


NODE_CLASS_MAPPINGS = {"MiniCPMNode": MiniCPMNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniCPMNode": "MiniCPM-V 2.6 (GGUF)"}
