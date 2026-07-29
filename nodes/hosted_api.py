"""Secure hosted LLM and VLM API nodes.

Credentials are deliberately server-side only.  Workflows select a provider,
never an API key or an arbitrary environment variable.  Built-in provider
credentials are also pinned to built-in HTTPS endpoints so a downloaded
workflow cannot redirect (for example) ``OPENAI_API_KEY`` to another host.
Custom endpoints can read only ``CUSTOM_API_KEY``.
"""

from __future__ import annotations

import ipaddress
import io
import json
import os
import re
from dataclasses import dataclass, replace
from typing import Any, Callable
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit

import torch
from PIL import Image

from .prompts import system_msg_prompts, system_msg_simple
from .runtime import require_module, tensor_to_pil

PROVIDER_CREDENTIAL = "Provider environment variable"
LOCAL_NO_KEY = "No key (loopback custom endpoint only)"
CREDENTIAL_SOURCES = (PROVIDER_CREDENTIAL, LOCAL_NO_KEY)
API_MODES = ("Auto", "Responses", "Chat Completions")
REASONING_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
IMAGE_DETAILS = ("auto", "low", "high")
OUTPUT_FORMATS = ("Text", "JSON object", "JSON Schema")
SCHEMA_API_STYLES = (
    "Auto (provider native)",
    "OpenAI JSON Schema",
    "llama.cpp JSON Schema",
    "JSON object + local validation",
)

MAX_ERROR_CHARS = 800
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_MEDIA_BYTES = 24 * 1024 * 1024
MAX_PROMPT_CHARS = 200_000
MAX_SYSTEM_PROMPT_CHARS = 32_000
MAX_SCHEMA_CHARS = 64_000
MAX_SCHEMA_DEPTH = 32
MAX_SCHEMA_NODES = 4096


@dataclass(frozen=True)
class ProviderProfile:
    label: str
    provider: str
    model: str
    api_key_env: str
    base_url: str | None
    api_mode: str = "Chat Completions"
    vision: bool = False
    responses: bool = False
    custom_endpoint: bool = False
    supports_seed: bool = False
    max_images: int = 8
    web_search: str = "none"
    schema_api_style: str = "openai"


_PROFILES = (
    ProviderProfile(
        "OpenAI — GPT-5.6 Terra",
        "OpenAI",
        "gpt-5.6-terra",
        "OPENAI_API_KEY",
        None,
        api_mode="Responses",
        vision=True,
        responses=True,
        max_images=32,
        web_search="responses",
    ),
    ProviderProfile(
        "OpenAI — GPT-5.6 Sol",
        "OpenAI",
        "gpt-5.6-sol",
        "OPENAI_API_KEY",
        None,
        api_mode="Responses",
        vision=True,
        responses=True,
        max_images=32,
        web_search="responses",
    ),
    ProviderProfile(
        "OpenAI — GPT-5.6 Luna",
        "OpenAI",
        "gpt-5.6-luna",
        "OPENAI_API_KEY",
        None,
        api_mode="Responses",
        vision=True,
        responses=True,
        max_images=32,
        web_search="responses",
    ),
    ProviderProfile(
        "Google — Gemini 3.6 Flash",
        "Google Gemini",
        "gemini-3.6-flash",
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        vision=True,
        max_images=32,
        web_search="gemini",
    ),
    ProviderProfile(
        "Google — Gemini 3.5 Flash",
        "Google Gemini",
        "gemini-3.5-flash",
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        vision=True,
        max_images=32,
        web_search="gemini",
    ),
    ProviderProfile(
        "Google — Gemini 3.5 Flash-Lite",
        "Google Gemini",
        "gemini-3.5-flash-lite",
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        vision=True,
        max_images=32,
        web_search="gemini",
    ),
    ProviderProfile(
        "Anthropic — Claude Fable 5",
        "Anthropic",
        "claude-fable-5",
        "ANTHROPIC_API_KEY",
        "https://api.anthropic.com/v1/",
        api_mode="Anthropic Messages",
        vision=True,
        max_images=20,
        web_search="anthropic",
    ),
    ProviderProfile(
        "Anthropic — Claude Opus 5",
        "Anthropic",
        "claude-opus-5",
        "ANTHROPIC_API_KEY",
        "https://api.anthropic.com/v1/",
        api_mode="Anthropic Messages",
        vision=True,
        max_images=20,
        web_search="anthropic",
    ),
    ProviderProfile(
        "Anthropic — Claude Sonnet 5",
        "Anthropic",
        "claude-sonnet-5",
        "ANTHROPIC_API_KEY",
        "https://api.anthropic.com/v1/",
        api_mode="Anthropic Messages",
        vision=True,
        max_images=20,
        web_search="anthropic",
    ),
    ProviderProfile(
        "Anthropic — Claude Haiku 4.5",
        "Anthropic",
        "claude-haiku-4-5",
        "ANTHROPIC_API_KEY",
        "https://api.anthropic.com/v1/",
        api_mode="Anthropic Messages",
        vision=True,
        max_images=20,
        web_search="anthropic",
    ),
    ProviderProfile(
        "xAI — Grok 4.5",
        "xAI",
        "grok-4.5",
        "XAI_API_KEY",
        "https://api.x.ai/v1",
        api_mode="Responses",
        vision=True,
        responses=True,
        max_images=10,
        web_search="responses",
    ),
    ProviderProfile(
        "DeepSeek — V4 Flash",
        "DeepSeek",
        "deepseek-v4-flash",
        "DEEPSEEK_API_KEY",
        "https://api.deepseek.com/v1",
        schema_api_style="json_object",
    ),
    ProviderProfile(
        "DeepSeek — V4 Pro",
        "DeepSeek",
        "deepseek-v4-pro",
        "DEEPSEEK_API_KEY",
        "https://api.deepseek.com/v1",
        schema_api_style="json_object",
    ),
    ProviderProfile(
        "Groq — Qwen 3.6 27B (vision)",
        "Groq",
        "qwen/qwen3.6-27b",
        "GROQ_API_KEY",
        "https://api.groq.com/openai/v1",
        api_mode="Responses",
        vision=True,
        responses=True,
        max_images=5,
        schema_api_style="json_object",
    ),
    ProviderProfile(
        "Groq — GPT-OSS 20B",
        "Groq",
        "openai/gpt-oss-20b",
        "GROQ_API_KEY",
        "https://api.groq.com/openai/v1",
        api_mode="Responses",
        responses=True,
        schema_api_style="openai",
    ),
    ProviderProfile(
        "Mistral — Mistral Large (latest)",
        "Mistral",
        "mistral-large-latest",
        "MISTRAL_API_KEY",
        "https://api.mistral.ai/v1",
        vision=True,
    ),
    ProviderProfile(
        "Mistral — Mistral Small (latest)",
        "Mistral",
        "mistral-small-latest",
        "MISTRAL_API_KEY",
        "https://api.mistral.ai/v1",
        vision=True,
    ),
    ProviderProfile(
        "Mistral — Ministral 14B (latest)",
        "Mistral",
        "ministral-14b-latest",
        "MISTRAL_API_KEY",
        "https://api.mistral.ai/v1",
        vision=True,
    ),
    ProviderProfile(
        "Together — Kimi K2.5 (vision)",
        "Together AI",
        "moonshotai/Kimi-K2.5",
        "TOGETHER_API_KEY",
        "https://api.together.ai/v1",
        vision=True,
    ),
    ProviderProfile(
        "Together — Qwen 3.5 9B (vision)",
        "Together AI",
        "Qwen/Qwen3.5-9B",
        "TOGETHER_API_KEY",
        "https://api.together.ai/v1",
        vision=True,
    ),
    ProviderProfile(
        "OpenRouter — Custom model",
        "OpenRouter",
        "",
        "OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1",
        vision=True,
        web_search="openrouter",
    ),
    ProviderProfile(
        "Custom / Local — OpenAI compatible",
        "Custom / Local",
        "",
        "CUSTOM_API_KEY",
        None,
        vision=True,
        responses=True,
        custom_endpoint=True,
        supports_seed=True,
        max_images=32,
    ),
)

PROVIDER_PROFILES = {profile.label: profile for profile in _PROFILES}

# Old labels remain valid so existing workflows keep their selected model.  The
# third legacy widget is scrubbed in the frontend before graph configuration.
_LEGACY_ALIASES = {
    "GPT-5.6 Terra": "OpenAI — GPT-5.6 Terra",
    "GPT-5.6 Sol": "OpenAI — GPT-5.6 Sol",
    "GPT-5.6 Luna": "OpenAI — GPT-5.6 Luna",
    "DeepSeek": "DeepSeek — V4 Flash",
    "Custom / OpenAI-compatible": "Custom / Local — OpenAI compatible",
    "ChatGPT-3.5": "OpenAI — GPT-5.6 Luna",
    "ChatGPT-4": "OpenAI — GPT-5.6 Terra",
    "gpt-3.5-turbo": "OpenAI — GPT-5.6 Luna",
    "gpt-3.5-turbo-0125": "OpenAI — GPT-5.6 Luna",
    "gpt-35-turbo": "OpenAI — GPT-5.6 Luna",
    "gpt-3.5-turbo-16k": "OpenAI — GPT-5.6 Luna",
    "gpt-3.5-turbo-16k-0613": "OpenAI — GPT-5.6 Luna",
    "gpt-4-0613": "OpenAI — GPT-5.6 Terra",
    "gpt-4-1106-preview": "OpenAI — GPT-5.6 Terra",
    "glm-4": "Custom / Local — OpenAI compatible",
}

API_MODELS = list(PROVIDER_PROFILES) + list(_LEGACY_ALIASES)
VLM_API_MODELS = [
    label for label, profile in PROVIDER_PROFILES.items() if profile.vision
]


def provider_profile(label: str) -> ProviderProfile:
    canonical = _LEGACY_ALIASES.get(label, label)
    try:
        profile = PROVIDER_PROFILES[canonical]
    except KeyError:
        raise ValueError(
            "Unknown hosted API provider/model selection. Refresh the node "
            "definition and choose a current provider."
        ) from None

    # Preserve the exact legacy model ID only when it represented a real API ID.
    if label in {
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-35-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0613",
        "gpt-4-1106-preview",
    }:
        return replace(profile, model=label, api_mode="Chat Completions")
    return profile


def _is_loopback_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    lowered = hostname.rstrip(".").lower()
    if lowered == "localhost" or lowered.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(lowered).is_loopback
    except ValueError:
        return False


def validate_custom_base_url(value: str) -> tuple[str, bool]:
    """Validate and normalize a user-controlled OpenAI-compatible endpoint."""

    raw = (value or "").strip()
    if not raw:
        raise ValueError(
            "Custom / Local requires a base_url. Use HTTPS for remote services "
            "or an HTTP loopback URL such as http://127.0.0.1:8000/v1."
        )
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"https", "http"} or not parsed.hostname:
        raise ValueError("base_url must be an absolute HTTP(S) URL.")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not contain embedded credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query string or fragment.")

    loopback = _is_loopback_host(parsed.hostname)
    if parsed.scheme.lower() != "https" and not loopback:
        raise ValueError("Remote API endpoints must use HTTPS.")

    normalized = urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc,
            parsed.path.rstrip("/") or "",
            "",
            "",
        )
    )
    return normalized, loopback


def resolve_endpoint(
    profile: ProviderProfile, base_url: str
) -> tuple[str | None, bool]:
    if profile.custom_endpoint:
        return validate_custom_base_url(base_url)
    if (base_url or "").strip():
        raise ValueError(
            "For credential safety, base_url overrides are accepted only by "
            "Custom / Local. Built-in provider keys are pinned to official hosts."
        )
    return profile.base_url, False


def resolve_api_key(
    profile: ProviderProfile,
    credential_source: str,
    *,
    loopback: bool,
) -> str:
    if credential_source not in CREDENTIAL_SOURCES:
        raise ValueError(
            "A legacy plaintext API key was removed from this workflow. "
            f"Set {profile.api_key_env} in the ComfyUI server environment and "
            "select 'Provider environment variable'."
        )
    if credential_source == LOCAL_NO_KEY:
        if not (profile.custom_endpoint and loopback):
            raise ValueError(
                "'No key' is allowed only for a loopback Custom / Local endpoint."
            )
        return "local-no-key"

    value = os.getenv(profile.api_key_env, "").strip()
    if value:
        return value
    if profile.custom_endpoint and loopback:
        return "local-no-key"
    raise ValueError(
        f"No credential is configured for {profile.provider}. Set "
        f"{profile.api_key_env} in the environment that starts ComfyUI, then "
        "restart the server. API keys are intentionally not accepted by nodes."
    )


_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|authorization|x-api-key)\b"
        r"\s*[=:]\s*[\"']?[^\s,\"'}]{4,}"
    ),
    re.compile(r"\b(?:sk-ant-|sk-|xai-|gsk_|AIza)[A-Za-z0-9._-]{8,}"),
    re.compile(r"(?i)(https?://)[^/@\s:]+:[^/@\s]+@"),
)


def redact_sensitive(value: object, secrets: tuple[str, ...] = ()) -> str:
    """Return a bounded, key-safe error string."""

    text = str(value)
    for secret in secrets:
        if not secret or secret == "local-no-key":
            continue
        for variant in {secret, quote(secret, safe=""), quote_plus(secret)}:
            if variant:
                text = text.replace(variant, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    text = " ".join(text.split())
    if len(text) > MAX_ERROR_CHARS:
        text = f"{text[:MAX_ERROR_CHARS]}…"
    return text


def safe_api_error(
    exc: Exception,
    profile: ProviderProfile,
    *,
    secret: str,
) -> RuntimeError:
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(getattr(exc, "response", None), "status_code", None)
    status_text = f" (HTTP {status})" if isinstance(status, int) else ""
    detail = redact_sensitive(exc, (secret,))
    if not detail:
        detail = type(exc).__name__
    return RuntimeError(f"{profile.provider} API request failed{status_text}: {detail}")


def _progress_text_sender(node_id: str | None) -> Callable[[str], None] | None:
    if node_id is None:
        return None
    try:
        from server import PromptServer

        server = PromptServer.instance
    except (ImportError, AttributeError):
        return None

    def send(text: str) -> None:
        try:
            server.send_progress_text(text, str(node_id), server.client_id)
        except Exception:
            return

    return send


def _uniform_indices(total: int, limit: int) -> list[int]:
    if total <= 0:
        return []
    count = min(total, max(1, int(limit)))
    if count == 1:
        return [0]
    return sorted(
        {
            round(index * (total - 1) / (count - 1))
            for index in range(count)
        }
    )


def _fit_image(image: Image.Image, max_edge: int) -> Image.Image:
    value = image.convert("RGB")
    edge = max(value.size)
    if edge <= max_edge:
        return value
    scale = max_edge / edge
    size = (
        max(1, round(value.width * scale)),
        max(1, round(value.height * scale)),
    )
    return value.resize(size, Image.Resampling.LANCZOS)


def _jpeg_data_uri(
    image: Image.Image,
    *,
    max_edge: int,
    quality: int,
) -> tuple[str, int]:
    value = _fit_image(image, max_edge)
    current_quality = max(45, min(95, int(quality)))
    while True:
        buffer = io.BytesIO()
        value.save(
            buffer,
            format="JPEG",
            quality=current_quality,
            optimize=True,
            progressive=True,
        )
        payload = buffer.getvalue()
        if len(payload) <= MAX_IMAGE_BYTES:
            break
        if max(value.size) <= 512:
            raise ValueError(
                "A sampled frame could not be compressed below the 4 MiB safety limit."
            )
        value = _fit_image(value, max(512, round(max(value.size) * 0.8)))
        current_quality = max(45, current_quality - 10)

    import base64

    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", len(payload)


def encode_image_batch(
    images: torch.Tensor,
    *,
    max_frames: int,
    max_image_edge: int,
    jpeg_quality: int,
) -> list[str]:
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a ComfyUI IMAGE tensor.")
    if images.ndim == 3:
        total = 1
    elif images.ndim == 4:
        total = int(images.shape[0])
    else:
        raise ValueError(f"Expected an HWC/BHWC IMAGE batch, got {tuple(images.shape)}.")

    output: list[str] = []
    total_bytes = 0
    for index in _uniform_indices(total, max_frames):
        pil_image = tensor_to_pil(images, index if images.ndim == 4 else 0)
        data_uri, payload_bytes = _jpeg_data_uri(
            pil_image,
            max_edge=max_image_edge,
            quality=jpeg_quality,
        )
        if total_bytes + payload_bytes > MAX_TOTAL_MEDIA_BYTES:
            break
        output.append(data_uri)
        total_bytes += payload_bytes
    if not output:
        raise ValueError("No image frames fit within the 24 MiB request safety limit.")
    return output


def _chat_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {
                "text",
                "output_text",
            }:
                parts.append(str(item.get("text", "")))
            elif getattr(item, "text", None):
                parts.append(str(item.text))
        return "".join(parts)
    return str(content or "")


def _bounded_text(name: str, value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        raise ValueError(
            f"{name} is {len(text):,} characters; the safety limit is {limit:,}."
        )
    return text


def _walk_json_schema(value: Any, *, depth: int = 0) -> int:
    """Bound a user schema and reject references that could resolve off-server."""

    if depth > MAX_SCHEMA_DEPTH:
        raise ValueError(
            f"json_schema exceeds the maximum nesting depth of {MAX_SCHEMA_DEPTH}."
        )
    nodes = 1
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("json_schema object keys must be strings.")
            if key in {"$ref", "$dynamicRef", "$recursiveRef"} and (
                not isinstance(item, str) or not item.startswith("#")
            ):
                raise ValueError(
                    "json_schema allows only local fragment reference values; "
                    "remote and file references are blocked."
                )
            nodes += _walk_json_schema(item, depth=depth + 1)
    elif isinstance(value, list):
        for item in value:
            nodes += _walk_json_schema(item, depth=depth + 1)
    if nodes > MAX_SCHEMA_NODES:
        raise ValueError(
            f"json_schema exceeds the {MAX_SCHEMA_NODES:,}-node safety limit."
        )
    return nodes


def parse_json_schema(output_format: str, raw_schema: object) -> dict[str, Any] | None:
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}.")
    if output_format != "JSON Schema":
        return None

    text = _bounded_text("json_schema", raw_schema, MAX_SCHEMA_CHARS)
    if not text:
        raise ValueError("JSON Schema output requires a json_schema value.")
    try:
        schema = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"json_schema is not valid JSON (line {exc.lineno}, column {exc.colno})."
        ) from None
    if not isinstance(schema, dict):
        raise ValueError("json_schema must be a JSON object.")
    _walk_json_schema(schema)

    jsonschema = require_module("jsonschema", "jsonschema>=4.22,<5")
    try:
        validator_class = jsonschema.validators.validator_for(schema)
        validator_class.check_schema(schema)
    except Exception:
        raise ValueError(
            "json_schema is not valid for its declared JSON Schema draft."
        ) from None
    return schema


def _schema_path(path: Any) -> str:
    output = "$"
    for item in path:
        if isinstance(item, int):
            output += f"[{item}]"
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(item)):
            output += f".{item}"
        else:
            output += f"[{json.dumps(str(item), ensure_ascii=True)}]"
    return output


def validate_structured_output(
    result: str,
    output_format: str,
    schema: dict[str, Any] | None,
) -> str:
    """Parse, locally validate, and normalize completed structured output."""

    if output_format == "Text":
        return result
    candidate = result.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        first_newline = candidate.find("\n")
        if first_newline >= 0:
            candidate = candidate[first_newline + 1 : -3].strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The provider did not return valid JSON "
            f"(line {exc.lineno}, column {exc.colno})."
        ) from None
    if output_format == "JSON object" and not isinstance(parsed, dict):
        raise RuntimeError("The provider returned JSON, but it was not an object.")
    if schema is not None:
        jsonschema = require_module("jsonschema", "jsonschema>=4.22,<5")
        validator_class = jsonschema.validators.validator_for(schema)
        validator = validator_class(schema)
        error = next(validator.iter_errors(parsed), None)
        if error is not None:
            path = _schema_path(error.absolute_path)
            rule = str(error.validator or "schema")
            raise RuntimeError(
                f"Structured output failed local validation at {path} "
                f"({rule} constraint)."
            ) from None
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def _structured_prompt(
    system_prompt: str,
    output_format: str,
    schema: dict[str, Any] | None,
) -> str:
    if output_format == "Text":
        return system_prompt
    if output_format == "JSON object":
        guidance = (
            "Return only one valid JSON object. Do not wrap it in Markdown or "
            "include commentary outside the JSON."
        )
    else:
        guidance = (
            "Return only JSON matching this schema exactly. Do not wrap it in "
            "Markdown or include commentary outside the JSON.\nJSON Schema:\n"
            f"{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
        )
    return f"{system_prompt.rstrip()}\n\n{guidance}".strip()


def _schema_style(profile: ProviderProfile, requested: str) -> str:
    if requested not in SCHEMA_API_STYLES:
        raise ValueError(
            f"schema_api_style must be one of {SCHEMA_API_STYLES}."
        )
    if requested == "Auto (provider native)":
        return profile.schema_api_style
    if not profile.custom_endpoint:
        raise ValueError(
            "schema_api_style overrides are available only for Custom / Local; "
            "built-in providers use their verified native contract."
        )
    return {
        "OpenAI JSON Schema": "openai",
        "llama.cpp JSON Schema": "llama_cpp",
        "JSON object + local validation": "json_object",
    }[requested]


def _chat_response_format(
    output_format: str,
    schema: dict[str, Any] | None,
    schema_style: str,
) -> dict[str, Any] | None:
    if output_format == "Text":
        return None
    if output_format == "JSON object" or schema_style == "json_object":
        return {"type": "json_object"}
    if schema_style == "llama_cpp":
        return {"type": "json_schema", "schema": schema}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "comfyui_output",
            "strict": True,
            "schema": schema,
        },
    }


def _stream_responses(client: Any, request: dict[str, Any], sender) -> str:
    stream = client.responses.create(**request, stream=True)
    chunks: list[str] = []
    try:
        for event in stream:
            if getattr(event, "type", "") != "response.output_text.delta":
                continue
            delta = str(getattr(event, "delta", "") or "")
            if not delta:
                continue
            chunks.append(delta)
            if sender is not None:
                sender("".join(chunks))
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    return "".join(chunks).strip()


def _stream_chat(client: Any, request: dict[str, Any], sender) -> str:
    stream = client.chat.completions.create(**request, stream=True)
    chunks: list[str] = []
    try:
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = _chat_text(getattr(choices[0].delta, "content", ""))
            if not delta:
                continue
            chunks.append(delta)
            if sender is not None:
                sender("".join(chunks))
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    return "".join(chunks).strip()


def _data_uri_parts(data_uri: str) -> tuple[str, str]:
    try:
        header, data = data_uri.split(",", 1)
        media_type = header.split(";", 1)[0].split(":", 1)[1]
    except (IndexError, ValueError):
        raise ValueError("Invalid image data URI.") from None
    if media_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise ValueError(f"Unsupported image media type: {media_type}.")
    return media_type, data


def _anthropic_image_source(data_uri: str) -> dict[str, Any]:
    media_type, data = _data_uri_parts(data_uri)
    return {
        "type": "base64",
        "media_type": media_type,
        "data": data,
    }


def _anthropic_response_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


def _gemini_response_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    content = candidates[0].get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    return "".join(
        str(part.get("text", ""))
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )


def _call_gemini_api(
    *,
    profile: ProviderProfile,
    model: str,
    api_key: str,
    system_prompt: str,
    prompt: str,
    image_data: list[str],
    timeout_seconds: float,
    max_output_tokens: int,
    stream_output: bool,
    use_system_proxy: bool,
    unique_id: str | None,
    web_search: bool,
    output_format: str,
    output_schema: dict[str, Any] | None,
) -> str:
    """Use Gemini's native multimodal API for search and structured output."""

    httpx = require_module("httpx", "httpx")
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for data_uri in image_data:
        media_type, data = _data_uri_parts(data_uri)
        parts.append(
            {
                "inlineData": {
                    "mimeType": media_type,
                    "data": data,
                }
            }
        )
    generation_config: dict[str, Any] = {
        "maxOutputTokens": int(max_output_tokens),
    }
    if output_format != "Text":
        text_format: dict[str, Any] = {"mimeType": "application/json"}
        if output_schema is not None:
            text_format["schema"] = output_schema
        generation_config["responseFormat"] = {"text": text_format}

    request: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
        "store": False,
    }
    if web_search:
        request["tools"] = [{"google_search": {}}]

    headers = {
        "x-goog-api-key": api_key,
        "content-type": "application/json",
    }
    sender = _progress_text_sender(unique_id) if stream_output else None
    if sender is not None:
        sender("Connecting securely…")
    method = "streamGenerateContent" if stream_output else "generateContent"
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{quote(model, safe='')}:{method}"
    )
    if stream_output:
        url += "?alt=sse"

    client = httpx.Client(
        timeout=float(timeout_seconds),
        follow_redirects=False,
        trust_env=bool(use_system_proxy),
    )
    try:
        if stream_output:
            chunks: list[str] = []
            with client.stream(
                "POST",
                url,
                headers=headers,
                json=request,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        delta = _gemini_response_text(json.loads(raw))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                    if not delta:
                        continue
                    chunks.append(delta)
                    if sender is not None:
                        sender("".join(chunks))
            result = "".join(chunks).strip()
        else:
            response = client.post(
                url,
                headers=headers,
                json=request,
            )
            response.raise_for_status()
            result = _gemini_response_text(response.json()).strip()
        if not result:
            raise RuntimeError("Gemini returned an empty text response.")
        if sender is not None:
            sender(result)
        return result
    except Exception as exc:
        raise safe_api_error(exc, profile, secret=api_key) from None
    finally:
        client.close()


def _call_anthropic_api(
    *,
    profile: ProviderProfile,
    model: str,
    endpoint: str,
    api_key: str,
    system_prompt: str,
    prompt: str,
    image_data: list[str],
    timeout_seconds: float,
    max_output_tokens: int,
    stream_output: bool,
    use_system_proxy: bool,
    unique_id: str | None,
    web_search: bool,
    output_format: str,
    output_schema: dict[str, Any] | None,
) -> str:
    """Use Anthropic's native Messages contract, including native vision."""

    httpx = require_module("httpx", "httpx")
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(
        {
            "type": "image",
            "source": _anthropic_image_source(data_uri),
        }
        for data_uri in image_data
    )
    request = {
        "model": model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": int(max_output_tokens),
        "stream": bool(stream_output),
    }
    if web_search:
        request["tools"] = [
            {
                "type": "web_search_20260318",
                "name": "web_search",
                "allowed_callers": ["direct"],
                "response_inclusion": "excluded",
                "max_uses": 5,
            }
        ]
    # Anthropic citation blocks and native structured output cannot be mixed.
    # In that combination, JSON-object prompting plus local validation remains
    # enabled without asking the API for an incompatible output_config.
    if output_schema is not None and not web_search:
        request["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": output_schema,
            }
        }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    sender = _progress_text_sender(unique_id) if stream_output else None
    if sender is not None:
        sender("Connecting securely…")

    client = httpx.Client(
        timeout=float(timeout_seconds),
        follow_redirects=False,
        trust_env=bool(use_system_proxy),
    )
    try:
        url = f"{endpoint.rstrip('/')}/messages"
        if stream_output:
            chunks: list[str] = []
            with client.stream(
                "POST",
                url,
                headers=headers,
                json=request,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    delta = event.get("delta", {})
                    if (
                        event.get("type") != "content_block_delta"
                        or delta.get("type") != "text_delta"
                    ):
                        continue
                    text = str(delta.get("text", "") or "")
                    if not text:
                        continue
                    chunks.append(text)
                    if sender is not None:
                        sender("".join(chunks))
            result = "".join(chunks).strip()
        else:
            response = client.post(
                url,
                headers=headers,
                json=request,
            )
            response.raise_for_status()
            result = _anthropic_response_text(response.json())
        if not result:
            raise RuntimeError("Anthropic returned an empty text response.")
        if sender is not None:
            sender(result)
        return result
    except Exception as exc:
        raise safe_api_error(exc, profile, secret=api_key) from None
    finally:
        client.close()


def _call_hosted_api(
    *,
    profile: ProviderProfile,
    model: str,
    endpoint: str | None,
    api_key: str,
    mode: str,
    system_prompt: str,
    prompt: str,
    image_data: list[str],
    image_detail: str,
    timeout_seconds: float,
    max_output_tokens: int,
    reasoning_effort: str,
    seed: int,
    stream_output: bool,
    use_system_proxy: bool,
    unique_id: str | None,
    web_search: bool,
    output_format: str,
    output_schema: dict[str, Any] | None,
    schema_api_style: str,
) -> str:
    if image_detail not in IMAGE_DETAILS:
        raise ValueError(f"image_detail must be one of {IMAGE_DETAILS}.")
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}.")
    if web_search and profile.web_search == "none":
        raise ValueError(
            f"{profile.label} does not expose native web search through this API. "
            "Choose OpenAI, Gemini, Anthropic, xAI, or route the model through "
            "OpenRouter."
        )
    if profile.web_search == "gemini" and (
        web_search or output_format != "Text"
    ):
        return validate_structured_output(
            _call_gemini_api(
                profile=profile,
                model=model,
                api_key=api_key,
                system_prompt=system_prompt,
                prompt=prompt,
                image_data=image_data,
                timeout_seconds=timeout_seconds,
                max_output_tokens=max_output_tokens,
                stream_output=stream_output,
                use_system_proxy=use_system_proxy,
                unique_id=unique_id,
                web_search=web_search,
                output_format=output_format,
                output_schema=output_schema,
            ),
            output_format,
            output_schema,
        )
    if mode == "Anthropic Messages":
        if not endpoint:
            raise ValueError("Anthropic Messages requires its fixed API endpoint.")
        return validate_structured_output(
            _call_anthropic_api(
                profile=profile,
                model=model,
                endpoint=endpoint,
                api_key=api_key,
                system_prompt=system_prompt,
                prompt=prompt,
                image_data=image_data,
                timeout_seconds=timeout_seconds,
                max_output_tokens=max_output_tokens,
                stream_output=stream_output,
                use_system_proxy=use_system_proxy,
                unique_id=unique_id,
                web_search=web_search,
                output_format=output_format,
                output_schema=output_schema,
            ),
            output_format,
            output_schema,
        )
    openai = require_module("openai", "openai>=2,<3")
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": float(timeout_seconds),
        "max_retries": 2,
    }
    if endpoint:
        client_kwargs["base_url"] = endpoint

    http_client_class = getattr(openai, "DefaultHttpxClient", None)
    if http_client_class is not None:
        client_kwargs["http_client"] = http_client_class(
            follow_redirects=False,
            trust_env=bool(use_system_proxy),
            timeout=float(timeout_seconds),
        )

    client = None
    sender = _progress_text_sender(unique_id) if stream_output else None
    if sender is not None:
        sender("Connecting securely…")
    try:
        client = openai.OpenAI(**client_kwargs)
        if mode == "Responses":
            content: list[dict[str, Any]] = [
                {"type": "input_text", "text": prompt}
            ]
            content.extend(
                {
                    "type": "input_image",
                    "image_url": data_uri,
                    "detail": image_detail,
                }
                for data_uri in image_data
            )
            request: dict[str, Any] = {
                "model": model,
                "instructions": system_prompt,
                "input": [{"role": "user", "content": content}],
                "max_output_tokens": int(max_output_tokens),
            }
            if profile.provider == "OpenAI":
                request["store"] = False
                if reasoning_effort != "none":
                    request["reasoning"] = {"effort": reasoning_effort}
            if web_search:
                request["tools"] = [{"type": "web_search"}]
            if output_format == "JSON object":
                request["text"] = {"format": {"type": "json_object"}}
            elif output_schema is not None:
                request["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "comfyui_output",
                        "strict": True,
                        "schema": output_schema,
                    }
                }
            if stream_output:
                result = _stream_responses(client, request, sender)
            else:
                response = client.responses.create(**request)
                result = str(getattr(response, "output_text", "") or "").strip()
        else:
            if image_data:
                user_content: str | list[dict[str, Any]] = [
                    {"type": "text", "text": prompt},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri,
                                "detail": image_detail,
                            },
                        }
                        for data_uri in image_data
                    ],
                ]
            else:
                user_content = prompt
            request = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": int(max_output_tokens),
            }
            if profile.supports_seed and int(seed):
                request["seed"] = int(seed)
            response_format = _chat_response_format(
                output_format,
                output_schema,
                schema_api_style,
            )
            if response_format is not None:
                request["response_format"] = response_format
            if web_search and profile.web_search == "openrouter":
                request["tools"] = [{"type": "openrouter:web_search"}]
            if stream_output:
                result = _stream_chat(client, request, sender)
            else:
                completion = client.chat.completions.create(**request)
                choices = getattr(completion, "choices", None) or []
                result = (
                    _chat_text(getattr(choices[0].message, "content", "")).strip()
                    if choices
                    else ""
                )
        if not result:
            raise RuntimeError("The provider returned an empty text response.")
        result = validate_structured_output(
            result,
            output_format,
            output_schema,
        )
        if sender is not None:
            sender(result)
        return result
    except Exception as exc:
        raise safe_api_error(exc, profile, secret=api_key) from None
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def execute_hosted(
    *,
    model_name: str,
    credential_source: str,
    prompt: str,
    system_prompt: str,
    base_url: str,
    model_override: str,
    api_mode: str,
    timeout_seconds: float,
    max_output_tokens: int,
    reasoning_effort: str,
    seed: int,
    stream_output: bool,
    use_system_proxy: bool,
    unique_id: str | None,
    image_data: list[str] | None = None,
    image_detail: str = "auto",
    web_search: bool = False,
    output_format: str = "Text",
    json_schema: str = "",
    schema_api_style: str = "Auto (provider native)",
) -> tuple[str, str, ProviderProfile]:
    profile = provider_profile(model_name)
    model = (model_override or profile.model).strip()
    if not model:
        raise ValueError(
            f"A model_override is required for {profile.label}."
        )
    images = image_data or []
    if images and not profile.vision:
        raise ValueError(f"{profile.label} is not configured for image input.")

    endpoint, loopback = resolve_endpoint(profile, base_url)
    api_key = resolve_api_key(
        profile,
        credential_source,
        loopback=loopback,
    )
    mode = profile.api_mode if api_mode == "Auto" else api_mode
    if (
        api_mode == "Auto"
        and profile.provider == "Groq"
        and output_format != "Text"
    ):
        # Groq documents structured output on Chat Completions. Keep Responses
        # as the fast default for ordinary text/vision generation.
        mode = "Chat Completions"
    if mode not in {"Responses", "Chat Completions", "Anthropic Messages"}:
        raise ValueError("api_mode must be Auto, Responses, or Chat Completions.")
    if mode == "Responses" and not profile.responses:
        raise ValueError(
            f"{profile.provider} is configured for Chat Completions. "
            "Use Auto unless its official compatibility layer adds Responses."
        )
    safe_prompt = _bounded_text("prompt", prompt, MAX_PROMPT_CHARS)
    safe_system_prompt = _bounded_text(
        "system_prompt",
        system_prompt,
        MAX_SYSTEM_PROMPT_CHARS,
    )
    output_schema = parse_json_schema(output_format, json_schema)
    safe_system_prompt = _structured_prompt(
        safe_system_prompt,
        output_format,
        output_schema,
    )
    effective_schema_style = (
        _schema_style(profile, schema_api_style)
        if output_format == "JSON Schema"
        else profile.schema_api_style
    )
    result = _call_hosted_api(
        profile=profile,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        mode=mode,
        system_prompt=safe_system_prompt,
        prompt=safe_prompt,
        image_data=images,
        image_detail=image_detail,
        timeout_seconds=max(1.0, min(1800.0, float(timeout_seconds))),
        max_output_tokens=max(1, min(131072, int(max_output_tokens))),
        reasoning_effort=reasoning_effort,
        seed=seed,
        stream_output=bool(stream_output),
        use_system_proxy=bool(use_system_proxy),
        unique_id=unique_id,
        web_search=bool(web_search),
        output_format=output_format,
        output_schema=output_schema,
        schema_api_style=effective_schema_style,
    )
    return result, model, profile


class PromptGenerateAPI:
    """Backwards-compatible class name for the secure hosted LLM node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    API_MODELS,
                    {"default": "OpenAI — GPT-5.6 Sol"},
                ),
                "chat_type": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Prompt Generator",
                        "label_off": "Simple Chat",
                    },
                ),
                # Kept in the old api_key widget position. It never accepts a key.
                "credential_source": (
                    CREDENTIAL_SOURCES,
                    {
                        "default": PROVIDER_CREDENTIAL,
                        "tooltip": (
                            "Keys stay in the ComfyUI server environment and are "
                            "never serialized into the workflow."
                        ),
                    },
                ),
                "description": ("STRING", {"multiline": True, "default": ""}),
                "question": ("STRING", {"multiline": True, "default": ""}),
                "context_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 30,
                        "step": 1,
                        "tooltip": (
                            "Legacy slot retained for workflow compatibility. "
                            "Hosted calls are stateless for privacy."
                        ),
                    },
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
                            "Custom / Local only. Remote URLs require HTTPS; "
                            "HTTP is allowed only for loopback."
                        ),
                    },
                ),
                "model_override": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Exact provider model ID. Overrides the preset.",
                    },
                ),
                "api_mode": (API_MODES, {"default": "Auto"}),
                "timeout_seconds": (
                    "FLOAT",
                    {"default": 120.0, "min": 1.0, "max": 1800.0},
                ),
                "reasoning_effort": (
                    REASONING_EFFORTS,
                    {"default": "none"},
                ),
                "max_output_tokens": (
                    "INT",
                    {"default": 4096, "min": 1, "max": 131072},
                ),
                "web_search": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Enable the provider's native/server-side web search. "
                            "Search calls may have additional cost and data terms."
                        ),
                    },
                ),
                "output_format": (
                    OUTPUT_FORMATS,
                    {
                        "default": "Text",
                        "tooltip": (
                            "JSON modes are parsed and validated locally before "
                            "the node succeeds."
                        ),
                    },
                ),
                "json_schema": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "JSON Schema object used when output_format is "
                            "JSON Schema."
                        ),
                    },
                ),
                "schema_api_style": (
                    SCHEMA_API_STYLES,
                    {
                        "default": "Auto (provider native)",
                        "tooltip": (
                            "Custom / Local compatibility override. llama.cpp "
                            "uses a different response_format schema shape."
                        ),
                    },
                ),
                "stream_output": ("BOOLEAN", {"default": True}),
                "use_system_proxy": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Opt in to HTTP(S)_PROXY from the server environment."
                        ),
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_prompt"
    CATEGORY = "VLM Nodes/API"
    DESCRIPTION = (
        "Stateless hosted LLM access with provider-bound environment credentials, "
        "redacted errors, HTTPS enforcement, and optional live text streaming."
    )

    def generate_prompt(
        self,
        model_name,
        chat_type,
        credential_source,
        description,
        question,
        context_size,
        seed,
        base_url="",
        model_override="",
        api_mode="Auto",
        timeout_seconds=120.0,
        reasoning_effort="none",
        max_output_tokens=4096,
        web_search=False,
        output_format="Text",
        json_schema="",
        schema_api_style="Auto (provider native)",
        stream_output=True,
        use_system_proxy=False,
        unique_id=None,
    ):
        del context_size
        prompt = (
            f"Description:\n{str(description).strip()}\n\n"
            f"Optional question:\n{str(question).strip()}"
        ).strip()
        result, _model, _profile = execute_hosted(
            model_name=model_name,
            credential_source=credential_source,
            prompt=prompt,
            system_prompt=system_msg_prompts if chat_type else system_msg_simple,
            base_url=base_url,
            model_override=model_override,
            api_mode=api_mode,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            seed=seed,
            stream_output=stream_output,
            use_system_proxy=use_system_proxy,
            unique_id=unique_id,
            web_search=web_search,
            output_format=output_format,
            json_schema=json_schema,
            schema_api_style=schema_api_style,
        )
        return (result,)


class HostedVLMAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    VLM_API_MODELS,
                    {"default": "OpenAI — GPT-5.6 Sol"},
                ),
                "credential_source": (
                    CREDENTIAL_SOURCES,
                    {"default": PROVIDER_CREDENTIAL},
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe the important visual details.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "You are a precise visual assistant. Distinguish "
                            "observations from uncertain inferences."
                        ),
                    },
                ),
                "max_frames": (
                    "INT",
                    {"default": 8, "min": 1, "max": 32, "step": 1},
                ),
                "max_image_edge": (
                    "INT",
                    {"default": 1536, "min": 256, "max": 2048, "step": 64},
                ),
                "jpeg_quality": (
                    "INT",
                    {"default": 88, "min": 45, "max": 95, "step": 1},
                ),
                "image_detail": (IMAGE_DETAILS, {"default": "auto"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "base_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Custom / Local only. Remote URLs require HTTPS."
                        ),
                    },
                ),
                "model_override": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Required for OpenRouter/Custom; optional for presets."
                        ),
                    },
                ),
                "api_mode": (API_MODES, {"default": "Auto"}),
                "timeout_seconds": (
                    "FLOAT",
                    {"default": 180.0, "min": 1.0, "max": 1800.0},
                ),
                "max_output_tokens": (
                    "INT",
                    {"default": 4096, "min": 1, "max": 131072},
                ),
                "reasoning_effort": (
                    REASONING_EFFORTS,
                    {"default": "none"},
                ),
                "web_search": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Enable native/server-side search where the selected "
                            "provider supports it."
                        ),
                    },
                ),
                "output_format": (
                    OUTPUT_FORMATS,
                    {"default": "Text"},
                ),
                "json_schema": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "For JSON Schema mode, including open-source VLMs "
                            "served by llama.cpp, vLLM, or Ollama."
                        ),
                    },
                ),
                "schema_api_style": (
                    SCHEMA_API_STYLES,
                    {
                        "default": "Auto (provider native)",
                        "tooltip": (
                            "Custom / Local only. Select llama.cpp for its direct "
                            "schema response_format dialect."
                        ),
                    },
                ),
                "stream_output": ("BOOLEAN", {"default": True}),
                "use_system_proxy": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("text", "model_used", "frames_sent")
    FUNCTION = "analyze"
    CATEGORY = "VLM Nodes/API"
    DESCRIPTION = (
        "Secure hosted image and sampled-video-frame understanding. Local images "
        "are resized, JPEG-compressed, and bounded before upload."
    )

    def analyze(
        self,
        model_name,
        credential_source,
        prompt,
        system_prompt,
        max_frames,
        max_image_edge,
        jpeg_quality,
        image_detail,
        images=None,
        base_url="",
        model_override="",
        api_mode="Auto",
        timeout_seconds=180.0,
        max_output_tokens=4096,
        reasoning_effort="none",
        web_search=False,
        output_format="Text",
        json_schema="",
        schema_api_style="Auto (provider native)",
        stream_output=True,
        use_system_proxy=False,
        unique_id=None,
    ):
        profile = provider_profile(model_name)
        image_data = (
            encode_image_batch(
                images,
                max_frames=min(int(max_frames), profile.max_images),
                max_image_edge=max_image_edge,
                jpeg_quality=jpeg_quality,
            )
            if images is not None
            else []
        )
        result, model, _profile = execute_hosted(
            model_name=model_name,
            credential_source=credential_source,
            prompt=prompt,
            system_prompt=system_prompt,
            base_url=base_url,
            model_override=model_override,
            api_mode=api_mode,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            seed=0,
            stream_output=stream_output,
            use_system_proxy=use_system_proxy,
            unique_id=unique_id,
            image_data=image_data,
            image_detail=image_detail,
            web_search=web_search,
            output_format=output_format,
            json_schema=json_schema,
            schema_api_style=schema_api_style,
        )
        return result, model, len(image_data)


NODE_CLASS_MAPPINGS = {
    "PromptGenerateAPI": PromptGenerateAPI,
    "HostedVLMAPI": HostedVLMAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptGenerateAPI": "Hosted LLM API (Secure)",
    "HostedVLMAPI": "Hosted VLM API (Secure)",
}
