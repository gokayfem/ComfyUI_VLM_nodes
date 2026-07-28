"""Text, structured-output, API, and music nodes.

llama-cpp-agent was intentionally removed.  llama-cpp-python has native JSON
Schema support, which avoids the dependency's unstable wrapper API while
producing stricter output.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal, Optional

import folder_paths
import torch
from pydantic import BaseModel, Field

from .prompts import system_msg_prompts, system_msg_simple
from .runtime import (
    LlamaHandle,
    close_handle,
    default_llama_threads,
    llama_chat_content,
    llama_runtime_input_types,
    llama_runtime_options,
    require_module,
    resolve_model_path,
    unwrap_llm,
)


class AnyType(str):
    def __ne__(self, value: object) -> bool:
        return False


ANY = AnyType("*")


class Analysis(BaseModel):
    main_character: list[str] = Field(..., description="Main subjects and objects.")
    artform: list[str] = Field(..., description="Art forms present.")
    photo_type: list[str] = Field(..., description="Photographic genres.")
    color_with_objects: list[str] = Field(
        ..., description="Objects paired with their colors."
    )
    digital_artform: list[str] = Field(..., description="Digital art techniques.")
    background: list[str] = Field(..., description="Background details.")
    lighting: list[str] = Field(..., description="Lighting details.")


class PromptGen(BaseModel):
    prompt: str = Field(..., description="A production-ready image prompt.")


class Suggestion(BaseModel):
    suggestion1: str
    suggestion2: str
    suggestion3: str
    suggestion4: str
    suggestion5: str


class ArtisticTechniques(BaseModel):
    preferred: list[str]
    avoided: list[str]


class ImageryTheme(BaseModel):
    core_subject: str
    additional_elements: Optional[list[str]] = None


class VisualStyle(BaseModel):
    desired: list[str]
    undesired: list[str]


class ArtInspirationNarrative(BaseModel):
    description: str


class ArtPromptSpecification(BaseModel):
    techniques: ArtisticTechniques
    theme: ImageryTheme
    style: VisualStyle
    creative_descriptions: list[ArtInspirationNarrative] = Field(default_factory=list)


def _schema(model_class: type[BaseModel]) -> dict[str, Any]:
    if hasattr(model_class, "model_json_schema"):
        return model_class.model_json_schema()
    return model_class.schema()


def _response_content(response: dict[str, Any]) -> str:
    return llama_chat_content(response)


def _chat(
    model: Any,
    *,
    prompt: str,
    system: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    repeat_penalty: float = 1.1,
    seed: int = 42,
    response_format: dict[str, Any] | None = None,
) -> str:
    llm = unwrap_llm(model)
    kwargs: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "repeat_penalty": repeat_penalty,
        "seed": seed,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    return _response_content(llm.create_chat_completion(**kwargs))


def _structured_chat(
    model: Any,
    *,
    prompt: str,
    system: str,
    schema: dict[str, Any],
    temperature: float,
    max_tokens: int = 512,
) -> tuple[str, Any]:
    raw = _chat(
        model,
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object", "schema": schema},
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"The model did not return valid JSON: {raw[:500]}") from exc
    return raw, parsed


API_MODELS = [
    "GPT-5.6 Terra",
    "GPT-5.6 Sol",
    "GPT-5.6 Luna",
    "DeepSeek",
    "Custom / OpenAI-compatible",
    # Kept so saved workflows continue to deserialize without substitutions.
    "ChatGPT-3.5",
    "ChatGPT-4",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-35-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4-0613",
    "gpt-4-1106-preview",
    "glm-4",
]

API_ROUTES = {
    "GPT-5.6 Sol": ("gpt-5.6-sol", None, "Responses"),
    "GPT-5.6 Terra": ("gpt-5.6-terra", None, "Responses"),
    "GPT-5.6 Luna": ("gpt-5.6-luna", None, "Responses"),
    "DeepSeek": ("deepseek-chat", "https://api.deepseek.com/v1", "Chat Completions"),
    "ChatGPT-3.5": ("gpt-3.5-turbo", None, "Chat Completions"),
    "ChatGPT-4": ("gpt-4", None, "Chat Completions"),
    "gpt-35-turbo": ("gpt-35-turbo", None, "Chat Completions"),
    "glm-4": ("glm-4", None, "Chat Completions"),
}


class PromptGenerateAPI:
    def __init__(self):
        self.session_history: list[dict[str, str]] = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (API_MODELS, {"default": "GPT-5.6 Terra"}),
                "chat_type": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Prompt Generator",
                        "label_off": "Simple Chat",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Leave blank to use OPENAI_API_KEY, DEEPSEEK_API_KEY, "
                            "or VLM_API_KEY."
                        ),
                    },
                ),
                "description": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "question": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "context_size": (
                    "INT",
                    {"default": 5, "min": 0, "max": 30, "step": 1},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "base_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1."
                        ),
                    },
                ),
                "model_override": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Exact provider model ID. Overrides the picker.",
                    },
                ),
                "api_mode": (
                    ["Auto", "Responses", "Chat Completions"],
                    {"default": "Auto"},
                ),
                "timeout_seconds": (
                    "FLOAT",
                    {"default": 120.0, "min": 1.0, "max": 1800.0},
                ),
                "reasoning_effort": (
                    ["none", "low", "medium", "high", "xhigh", "max"],
                    {"default": "none"},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompt"
    CATEGORY = "VLM Nodes/LLM"

    def _route(
        self, model_name, model_override, base_url, api_mode
    ) -> tuple[str, str | None, str]:
        route = API_ROUTES.get(model_name)
        if route is None:
            if model_name in API_MODELS and model_name not in {
                "Custom / OpenAI-compatible"
            }:
                route = (model_name, None, "Chat Completions")
            else:
                route = ("", None, "Chat Completions")
        model, route_url, route_mode = route
        model = (model_override or model).strip()
        if not model:
            raise ValueError("A model ID is required for Custom / OpenAI-compatible.")
        effective_url = (base_url or route_url or "").strip() or None
        mode = route_mode if api_mode == "Auto" else api_mode
        return model, effective_url, mode

    def generate_prompt(
        self,
        model_name,
        chat_type,
        api_key,
        description,
        question,
        context_size,
        seed,
        base_url="",
        model_override="",
        api_mode="Auto",
        timeout_seconds=120.0,
        reasoning_effort="none",
    ):
        openai = require_module("openai", "openai")
        model, effective_url, mode = self._route(
            model_name, model_override, base_url, api_mode
        )
        key = (
            api_key.strip()
            or (os.getenv("DEEPSEEK_API_KEY", "") if model_name == "DeepSeek" else "")
            or os.getenv("VLM_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
        )
        if not key:
            raise ValueError(
                "No API key was supplied. Set OPENAI_API_KEY, "
                "DEEPSEEK_API_KEY, or VLM_API_KEY, or enter the key in the node."
            )

        client_kwargs: dict[str, Any] = {
            "api_key": key,
            "timeout": float(timeout_seconds),
            "max_retries": 2,
        }
        if effective_url:
            client_kwargs["base_url"] = effective_url
        client = openai.OpenAI(**client_kwargs)

        system = system_msg_prompts if chat_type else system_msg_simple
        user_message = (
            f"Description:\n{description.strip()}\n\n"
            f"Optional question:\n{question.strip()}"
        ).strip()
        history_limit = max(0, int(context_size)) * 2
        history = self.session_history[-history_limit:] if history_limit else []

        if mode == "Responses":
            response = client.responses.create(
                model=model,
                instructions=system,
                input=history + [{"role": "user", "content": user_message}],
                reasoning={"effort": reasoning_effort},
            )
            result = response.output_text
        else:
            messages = (
                [{"role": "system", "content": system}]
                + history
                + [{"role": "user", "content": user_message}]
            )
            request: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "seed": int(seed),
            }
            if model.startswith("gpt-5.6"):
                request["reasoning_effort"] = reasoning_effort
            completion = client.chat.completions.create(**request)
            result = completion.choices[0].message.content or ""

        self.session_history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": result},
            ]
        )
        return (result,)


class LLMLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"),),
                "max_ctx": (
                    "INT",
                    {"default": 2048, "min": 128, "max": 131072, "step": 64},
                ),
                "gpu_layers": (
                    "INT",
                    {"default": -1, "min": -1, "max": 1000, "step": 1},
                ),
                "n_threads": (
                    "INT",
                    {
                        "default": default_llama_threads(),
                        "min": 1,
                        "max": 256,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "chat_format": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Leave blank to use the chat template embedded in GGUF."
                        ),
                    },
                ),
                **llama_runtime_input_types(),
            },
        }

    RETURN_TYPES = ("CUSTOM",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_llm_checkpoint"
    CATEGORY = "VLM Nodes/LLM"

    def load_llm_checkpoint(
        self,
        ckpt_name,
        max_ctx,
        gpu_layers,
        n_threads,
        chat_format="",
        n_batch=512,
        n_ubatch=512,
        flash_attention="Auto",
        use_mmap=True,
        split_mode="Layer",
        main_gpu=0,
        tensor_split="",
    ):
        return (
            LlamaHandle(
                resolve_model_path(ckpt_name),
                n_ctx=max_ctx,
                n_gpu_layers=gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format.strip() or None,
                **llama_runtime_options(
                    n_batch=n_batch,
                    n_ubatch=n_ubatch,
                    flash_attention=flash_attention,
                    use_mmap=use_mmap,
                    split_mode=split_mode,
                    main_gpu=main_gpu,
                    tensor_split=tensor_split,
                ),
            ),
        )


class LLMPromptGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "model": ("CUSTOM", {"default": ""}),
                "max_tokens": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top_k": ("INT", {"default": 40, "min": 0, "step": 1}),
                "frequency_penalty": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
                "presence_penalty": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
                "repeat_penalty": (
                    "FLOAT",
                    {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text_advanced"
    CATEGORY = "VLM Nodes/LLM"

    def generate_text_advanced(
        self,
        prompt,
        model,
        max_tokens,
        temperature,
        top_p,
        top_k,
        frequency_penalty,
        presence_penalty,
        repeat_penalty,
    ):
        return (
            _chat(
                model,
                prompt=prompt,
                system=system_msg_prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
            ),
        )


class LLMSampler:
    @classmethod
    def INPUT_TYPES(cls):
        # Keep this order stable: Comfy serializes widget values by position.
        return {
            "required": {
                "system_msg": (
                    "STRING",
                    {
                        "default": "You are a helpful and accurate assistant.",
                        "multiline": True,
                    },
                ),
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "model": ("CUSTOM", {"default": ""}),
                "max_tokens": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top_k": ("INT", {"default": 40, "min": 0, "step": 1}),
                "frequency_penalty": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
                "presence_penalty": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
                "repeat_penalty": (
                    "FLOAT",
                    {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "seed": ("INT", {"default": 42, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text_advanced"
    CATEGORY = "VLM Nodes/LLM"

    def generate_text_advanced(
        self,
        system_msg,
        prompt,
        model,
        max_tokens,
        temperature,
        top_p,
        top_k,
        frequency_penalty,
        presence_penalty,
        repeat_penalty,
        seed,
    ):
        return (
            _chat(
                model,
                prompt=prompt,
                system=system_msg,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
                seed=seed,
            ),
        )


class ChatMusician:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "model": ("CUSTOM", {"default": ""}),
                "max_tokens": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top_k": ("INT", {"default": 40, "min": 0, "step": 1}),
                "frequency_penalty": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
                "presence_penalty": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
                "repeat_penalty": (
                    "FLOAT",
                    {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "seed": ("INT", {"default": 42, "step": 1}),
                "sample_rate": (
                    "INT",
                    {"default": 44100, "min": 8000, "max": 192000},
                ),
            }
        }

    RETURN_NAMES = (
        "response",
        "wave_form (legacy)",
        "sample_rate (legacy)",
        "audio",
    )
    RETURN_TYPES = ("STRING", ANY, "INT", "AUDIO")
    FUNCTION = "chat_musician"
    CATEGORY = "VLM Nodes/Audio"
    OUTPUT_NODE = True

    def chat_musician(
        self,
        prompt,
        model,
        max_tokens,
        temperature,
        top_p,
        top_k,
        frequency_penalty,
        presence_penalty,
        repeat_penalty,
        seed,
        sample_rate,
    ):
        response = _chat(
            model,
            prompt=(
                "Write exactly one complete ABC notation tune. Begin with X: "
                "and include headers and body.\n\n" + prompt
            ),
            system=(
                "You are a composer. Return valid ABC notation without a "
                "Markdown code fence."
            ),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            seed=seed,
        )
        match = re.search(r"(?ms)^X:\s*\d+.*", response)
        if match is None:
            raise RuntimeError(
                "The model response did not contain ABC notation beginning with X:."
            )
        abc = match.group(0).strip()

        symusic = require_module("symusic", "symusic")
        score = symusic.Score.from_abc(abc)
        rendered = symusic.Synthesizer(sample_rate=int(sample_rate)).render(
            score, stereo=True
        )
        waveform = torch.as_tensor(rendered, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise RuntimeError(
                f"Unexpected synthesizer waveform shape: {tuple(waveform.shape)}"
            )
        # Comfy AUDIO is [batch, channels, samples].
        audio = {
            "waveform": waveform.unsqueeze(0),
            "sample_rate": int(sample_rate),
        }
        # soundfile-compatible legacy output is [samples, channels].
        legacy = waveform.transpose(0, 1).contiguous().cpu().numpy()
        return (abc, legacy, int(sample_rate), audio)


class KeywordExtraction:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "model": ("CUSTOM", {"default": ""}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "keyword_extract"
    CATEGORY = "VLM Nodes/LLM"

    def keyword_extract(self, prompt, model, temperature):
        raw, _ = _structured_chat(
            model,
            prompt=prompt,
            system="Analyze the input and return only the requested JSON object.",
            schema=_schema(Analysis),
            temperature=temperature,
        )
        return (raw,)


class LLavaPromptGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return KeywordExtraction.INPUT_TYPES()

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompts"
    CATEGORY = "VLM Nodes/LLM"

    def generate_prompts(self, prompt, model, temperature):
        _, parsed = _structured_chat(
            model,
            prompt=prompt,
            system=(
                "Create one production-ready image-generation prompt and "
                "return only the requested JSON object."
            ),
            schema=_schema(PromptGen),
            temperature=temperature,
        )
        return (str(parsed["prompt"]),)


class CreativeArtPromptGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return KeywordExtraction.INPUT_TYPES()

    RETURN_TYPES = ("STRING",)
    FUNCTION = "create_creative_art_prompts"
    CATEGORY = "VLM Nodes/LLM"

    def create_creative_art_prompts(self, prompt, model, temperature):
        _, parsed = _structured_chat(
            model,
            prompt=prompt,
            system=(
                "Develop a coherent visual concept and return only the "
                "requested JSON object."
            ),
            schema=_schema(ArtPromptSpecification),
            temperature=temperature,
        )
        descriptions = parsed.get("creative_descriptions") or []
        if descriptions:
            return (str(descriptions[0]["description"]),)
        techniques = ", ".join(parsed["techniques"]["preferred"])
        theme = parsed["theme"]["core_subject"]
        styles = ", ".join(parsed["style"]["desired"])
        return (f"{theme}. Techniques: {techniques}. Visual style: {styles}.",)


class Suggester:
    @classmethod
    def INPUT_TYPES(cls):
        base = KeywordExtraction.INPUT_TYPES()["required"].copy()
        base["randomize"] = (
            "BOOLEAN",
            {
                "default": True,
                "label_on": "Similar",
                "label_off": "Different",
            },
        )
        return {"required": base}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_suggestions"
    CATEGORY = "VLM Nodes/LLM"

    def generate_suggestions(self, prompt, model, temperature, randomize):
        instruction = (
            "Create five close, useful variations of the input."
            if randomize
            else "Create five deliberately different but production-ready ideas."
        )
        raw, _ = _structured_chat(
            model,
            prompt=prompt,
            system=instruction + " Return only the requested JSON object.",
            schema=_schema(Suggestion),
            temperature=temperature,
        )
        return (raw,)


class StructuredOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "model": ("CUSTOM", {"default": ""}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "attribute_name": ("STRING", {"default": "result"}),
                "attribute_type": (
                    ["str", "int", "float", "bool", "Category"],
                    {"default": "str"},
                ),
                "attribute_description": (
                    "STRING",
                    {"default": ""},
                ),
                "categories": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma-separated values for Category.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "keyword_extract"
    CATEGORY = "VLM Nodes/LLM"

    def keyword_extract(
        self,
        prompt,
        model,
        temperature,
        attribute_name,
        attribute_type,
        attribute_description,
        categories,
    ):
        name = attribute_name.strip()
        if not name:
            raise ValueError("attribute_name cannot be empty.")
        types = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
        }
        property_schema: dict[str, Any] = {"description": attribute_description.strip()}
        if attribute_type == "Category":
            values = [value.strip() for value in categories.split(",") if value.strip()]
            if not values:
                raise ValueError(
                    "Category requires at least one comma-separated value."
                )
            property_schema.update({"type": "string", "enum": values})
        else:
            property_schema["type"] = types[attribute_type]
        schema = {
            "type": "object",
            "properties": {name: property_schema},
            "required": [name],
            "additionalProperties": False,
        }
        _, parsed = _structured_chat(
            model,
            prompt=prompt,
            system="Extract the requested value and return only valid JSON.",
            schema=schema,
            temperature=temperature,
        )
        value = parsed[name]
        return (value if isinstance(value, str) else json.dumps(value),)


class _CachedLLMBase:
    def __init__(self):
        self._handle = None
        self._key = None

    def _model(
        self,
        ckpt_name,
        max_ctx,
        gpu_layers,
        n_threads,
        seed=42,
        chat_format="",
        **runtime_options,
    ):
        key = (
            ckpt_name,
            int(max_ctx),
            int(gpu_layers),
            int(n_threads),
            int(seed),
            chat_format.strip(),
            tuple(sorted(runtime_options.items())),
        )
        if self._handle is None or self._key != key:
            close_handle(self._handle)
            self._handle = LlamaHandle(
                resolve_model_path(ckpt_name),
                n_ctx=max_ctx,
                n_gpu_layers=gpu_layers,
                n_threads=n_threads,
                seed=seed,
                chat_format=chat_format.strip() or None,
                **runtime_options,
            )
            self._key = key
        return self._handle

    def _maybe_unload(self, unload):
        if unload:
            close_handle(self._handle)
            self._handle = None
            self._key = None


class LLMOptionalMemoryFreeSimple(_CachedLLMBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"),),
                "max_ctx": (
                    "INT",
                    {"default": 4096, "min": 128, "max": 131072, "step": 64},
                ),
                "gpu_layers": (
                    "INT",
                    {"default": -1, "min": -1, "max": 1000, "step": 1},
                ),
                "n_threads": (
                    "INT",
                    {
                        "default": default_llama_threads(),
                        "min": 1,
                        "max": 256,
                        "step": 1,
                    },
                ),
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "unload": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "chat_format": (
                    "STRING",
                    {"default": ""},
                ),
                **llama_runtime_input_types(),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "VLM Nodes/LLM"

    def generate_text(
        self,
        ckpt_name,
        max_ctx,
        gpu_layers,
        n_threads,
        prompt,
        temperature,
        unload,
        chat_format="",
        n_batch=512,
        n_ubatch=512,
        flash_attention="Auto",
        use_mmap=True,
        split_mode="Layer",
        main_gpu=0,
        tensor_split="",
    ):
        options = llama_runtime_options(
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            flash_attention=flash_attention,
            use_mmap=use_mmap,
            split_mode=split_mode,
            main_gpu=main_gpu,
            tensor_split=tensor_split,
        )
        model = self._model(
            ckpt_name,
            max_ctx,
            gpu_layers,
            n_threads,
            chat_format=chat_format,
            **options,
        )
        try:
            return (
                _chat(
                    model,
                    prompt=prompt,
                    system="You are a helpful AI assistant.",
                    temperature=temperature,
                ),
            )
        finally:
            self._maybe_unload(unload)


class LLMOptionalMemoryFreeAdvanced(_CachedLLMBase):
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"),),
            "max_ctx": (
                "INT",
                {"default": 4096, "min": 128, "max": 131072, "step": 64},
            ),
            "gpu_layers": (
                "INT",
                {"default": -1, "min": -1, "max": 1000, "step": 1},
            ),
            "n_threads": (
                "INT",
                {
                    "default": default_llama_threads(),
                    "min": 1,
                    "max": 256,
                    "step": 1,
                },
            ),
            "system_msg": (
                "STRING",
                {
                    "default": "You are a helpful AI assistant.",
                    "multiline": True,
                },
            ),
            "prompt": (
                "STRING",
                {"default": "", "multiline": True},
            ),
            "max_tokens": (
                "INT",
                {"default": 512, "min": 1, "max": 8192, "step": 1},
            ),
            "temperature": (
                "FLOAT",
                {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01},
            ),
            "top_p": (
                "FLOAT",
                {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
            ),
            "top_k": ("INT", {"default": 40, "min": 0, "step": 1}),
            "frequency_penalty": (
                "FLOAT",
                {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
            ),
            "presence_penalty": (
                "FLOAT",
                {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
            ),
            "repeat_penalty": (
                "FLOAT",
                {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.01},
            ),
            "seed": ("INT", {"default": 42, "step": 1}),
            "unload": ("BOOLEAN", {"default": False}),
        }
        return {
            "required": required,
            "optional": {
                "chat_format": (
                    "STRING",
                    {"default": ""},
                ),
                **llama_runtime_input_types(),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text_advanced"
    CATEGORY = "VLM Nodes/LLM"

    def generate_text_advanced(
        self,
        ckpt_name,
        max_ctx,
        gpu_layers,
        n_threads,
        system_msg,
        prompt,
        max_tokens,
        temperature,
        top_p,
        top_k,
        frequency_penalty,
        presence_penalty,
        repeat_penalty,
        seed,
        unload,
        chat_format="",
        n_batch=512,
        n_ubatch=512,
        flash_attention="Auto",
        use_mmap=True,
        split_mode="Layer",
        main_gpu=0,
        tensor_split="",
    ):
        options = llama_runtime_options(
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            flash_attention=flash_attention,
            use_mmap=use_mmap,
            split_mode=split_mode,
            main_gpu=main_gpu,
            tensor_split=tensor_split,
        )
        model = self._model(
            ckpt_name,
            max_ctx,
            gpu_layers,
            n_threads,
            seed,
            chat_format,
            **options,
        )
        try:
            return (
                _chat(
                    model,
                    prompt=prompt,
                    system=system_msg,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    repeat_penalty=repeat_penalty,
                    seed=seed,
                ),
            )
        finally:
            self._maybe_unload(unload)


NODE_CLASS_MAPPINGS = {
    "LLMLoader": LLMLoader,
    "LLMSampler": LLMSampler,
    "LLMPromptGenerator": LLMPromptGenerator,
    "KeywordExtraction": KeywordExtraction,
    "LLavaPromptGenerator": LLavaPromptGenerator,
    "Suggester": Suggester,
    "PromptGenerateAPI": PromptGenerateAPI,
    "CreativeArtPromptGenerator": CreativeArtPromptGenerator,
    "ChatMusician": ChatMusician,
    "StructuredOutput": StructuredOutput,
    "LLMOptionalMemoryFreeSimple": LLMOptionalMemoryFreeSimple,
    "LLMOptionalMemoryFreeAdvanced": LLMOptionalMemoryFreeAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMLoader": "LLM Loader (GGUF)",
    "LLMSampler": "LLM Sampler",
    "LLMPromptGenerator": "LLM Prompt Generator",
    "KeywordExtraction": "Structured Keyword Extraction",
    "LLavaPromptGenerator": "Structured Prompt Generator",
    "Suggester": "Prompt Suggester",
    "PromptGenerateAPI": "OpenAI-Compatible Prompt API",
    "CreativeArtPromptGenerator": "Creative Art Prompt Generator",
    "ChatMusician": "Chat Musician",
    "StructuredOutput": "Structured Output",
    "LLMOptionalMemoryFreeSimple": "LLM (Managed Cache)",
    "LLMOptionalMemoryFreeAdvanced": "LLM (Managed Cache, Advanced)",
}
