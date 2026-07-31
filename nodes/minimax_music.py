"""MiniMax music generation and cover support with fixed regional routing."""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import torch

from .audioldm2 import ANY
from .hosted_api import redact_sensitive
from .runtime import require_module

API_KEY_ENV = "MINIMAX_API_KEY"
REGION_ENDPOINTS = {
    "global_en": "https://api.minimax.io/v1/music_generation",
    "cn_zh": "https://api.minimaxi.com/v1/music_generation",
}
GENERATION_MODELS = (
    "music-3.0",
    "music-2.6",
    "music-3.0-free",
    "music-2.6-free",
)
COVER_MODELS = ("music-cover", "music-cover-free")
MUSIC_MODELS = GENERATION_MODELS + COVER_MODELS
DEFAULT_MODEL = "music-3.0"
REQUEST_FIELDS = frozenset(
    {
        "model",
        "prompt",
        "lyrics",
        "stream",
        "output_format",
        "audio_setting",
        "lyrics_optimizer",
        "is_instrumental",
        "audio_url",
        "audio_base64",
        "cover_feature_id",
    }
)
OUTPUT_FORMATS = ("url", "hex")
STREAM_OUTPUT_FORMATS = ("hex",)
AUDIO_FORMATS = ("mp3", "wav", "pcm")
SAMPLE_RATES = (16000, 24000, 32000, 44100)
BITRATES = (32000, 64000, 128000, 256000)
REGIONAL_FIELDS = {"global_en": (), "cn_zh": ("aigc_watermark",)}
STATUS_IN_PROGRESS = 1
STATUS_COMPLETED = 2
MAX_COVER_BYTES = 50 * 1024 * 1024
MAX_AUDIO_BYTES = 128 * 1024 * 1024


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _validate_cover_base64(value: str) -> None:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError, TypeError):
        raise ValueError("audio_base64 must contain valid base64 data.") from None
    if len(decoded) > MAX_COVER_BYTES:
        raise ValueError("audio_base64 exceeds the 50 MiB cover input limit.")


def build_music_request(
    *,
    region: str,
    model: str,
    prompt: str,
    lyrics: str,
    stream: bool,
    output_format: str,
    audio_format: str,
    sample_rate: int,
    bitrate: int,
    lyrics_optimizer: bool,
    is_instrumental: bool,
    aigc_watermark: bool,
    audio_url: str = "",
    audio_base64: str = "",
    cover_feature_id: str = "",
) -> dict[str, Any]:
    """Validate node inputs and build the documented JSON request body."""

    if region not in REGION_ENDPOINTS:
        raise ValueError(f"region must be one of {tuple(REGION_ENDPOINTS)}.")
    if model not in MUSIC_MODELS:
        raise ValueError(f"model must be one of {MUSIC_MODELS}.")
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}.")
    if bool(stream) and output_format not in STREAM_OUTPUT_FORMATS:
        raise ValueError("Streaming music responses require output_format='hex'.")
    if audio_format not in AUDIO_FORMATS:
        raise ValueError(f"audio_format must be one of {AUDIO_FORMATS}.")
    if int(sample_rate) not in SAMPLE_RATES:
        raise ValueError(f"sample_rate must be one of {SAMPLE_RATES}.")
    if int(bitrate) not in BITRATES:
        raise ValueError(f"bitrate must be one of {BITRATES}.")

    clean_prompt = _clean_text(prompt)
    clean_lyrics = _clean_text(lyrics)
    clean_audio_url = _clean_text(audio_url)
    clean_audio_base64 = _clean_text(audio_base64)
    clean_cover_feature_id = _clean_text(cover_feature_id)
    if len(clean_prompt) > 2000:
        raise ValueError("prompt exceeds the 2,000-character music API limit.")

    payload: dict[str, Any] = {
        "model": model,
        "stream": bool(stream),
        "output_format": output_format,
        "audio_setting": {
            "sample_rate": int(sample_rate),
            "bitrate": int(bitrate),
            "format": audio_format,
        },
    }
    if clean_prompt:
        payload["prompt"] = clean_prompt
    if clean_lyrics:
        payload["lyrics"] = clean_lyrics

    if model in COVER_MODELS:
        if not 10 <= len(clean_prompt) <= 300:
            raise ValueError("Cover generation requires a 10-300 character prompt.")
        sources = (clean_audio_url, clean_audio_base64, clean_cover_feature_id)
        if sum(bool(value) for value in sources) != 1:
            raise ValueError(
                "Cover generation requires exactly one of audio_url, "
                "audio_base64, or cover_feature_id."
            )
        if clean_audio_base64:
            _validate_cover_base64(clean_audio_base64)
            payload["audio_base64"] = clean_audio_base64
        elif clean_audio_url:
            payload["audio_url"] = clean_audio_url
        else:
            if not 10 <= len(clean_lyrics) <= 1000:
                raise ValueError(
                    "cover_feature_id requires lyrics between 10 and 1,000 characters."
                )
            payload["cover_feature_id"] = clean_cover_feature_id
        if clean_lyrics and not 10 <= len(clean_lyrics) <= 1000:
            raise ValueError("Cover lyrics must be between 10 and 1,000 characters.")
    else:
        if any((clean_audio_url, clean_audio_base64, clean_cover_feature_id)):
            raise ValueError("Cover audio fields require a cover model.")
        if len(clean_lyrics) > 3500:
            raise ValueError("lyrics exceeds the 3,500-character music API limit.")
        if bool(is_instrumental) and not clean_prompt:
            raise ValueError("Instrumental generation requires a prompt.")
        if not bool(is_instrumental) and not clean_lyrics and not bool(lyrics_optimizer):
            raise ValueError(
                "Non-instrumental generation requires lyrics or lyrics_optimizer."
            )
        payload["lyrics_optimizer"] = bool(lyrics_optimizer)
        payload["is_instrumental"] = bool(is_instrumental)

    if region == "cn_zh":
        payload["aigc_watermark"] = bool(aigc_watermark)
    return payload


def _response_parts(payload: object) -> tuple[str, int, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise RuntimeError("MiniMax returned a non-object music response.")
    base_response = payload.get("base_resp")
    if not isinstance(base_response, dict):
        raise RuntimeError("MiniMax returned no base_resp status.")
    try:
        success_code = int(base_response.get("status_code"))
    except (TypeError, ValueError):
        raise RuntimeError("MiniMax returned an invalid base_resp status code.") from None
    if success_code != 0:
        message = _clean_text(base_response.get("status_msg")) or "unknown API error"
        raise RuntimeError(f"MiniMax music API error {success_code}: {message}")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("MiniMax returned no music data object.")
    try:
        status = int(data.get("status"))
    except (TypeError, ValueError):
        raise RuntimeError("MiniMax returned an invalid music status.") from None
    if status not in {STATUS_IN_PROGRESS, STATUS_COMPLETED}:
        raise RuntimeError(f"MiniMax returned unsupported music status {status}.")
    audio = data.get("audio", "")
    if not isinstance(audio, str):
        raise RuntimeError("MiniMax returned a non-string audio value.")
    extra_info = payload.get("extra_info")
    return audio.strip(), status, extra_info if isinstance(extra_info, dict) else {}


def _stream_audio(response: Any) -> tuple[str, dict[str, Any]]:
    audio = ""
    extra_info: dict[str, Any] = {}
    completed = False
    saw_payload = False
    for line in response.iter_lines():
        raw = line.decode("utf-8") if isinstance(line, bytes) else str(line)
        raw = raw.strip()
        if raw.startswith("data:"):
            raw = raw[5:].strip()
        if not raw or raw == "[DONE]" or raw.startswith("event:"):
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise RuntimeError("MiniMax returned invalid streaming JSON.") from None
        chunk, status, metadata = _response_parts(payload)
        saw_payload = True
        if chunk:
            if chunk.startswith(audio):
                audio = chunk
            elif not audio.startswith(chunk):
                audio += chunk
        if metadata:
            extra_info = metadata
        completed = completed or status == STATUS_COMPLETED
    if not saw_payload:
        raise RuntimeError("MiniMax returned an empty streaming response.")
    if not completed:
        raise RuntimeError("MiniMax streaming ended before music generation completed.")
    if not audio:
        raise RuntimeError("MiniMax returned no audio data.")
    return audio, extra_info


def _request_audio_value(
    client: Any,
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if payload["stream"]:
        with client.stream("POST", endpoint, headers=headers, json=payload) as response:
            response.raise_for_status()
            return _stream_audio(response)

    response = client.post(endpoint, headers=headers, json=payload)
    response.raise_for_status()
    audio, status, extra_info = _response_parts(response.json())
    if status != STATUS_COMPLETED:
        raise RuntimeError(
            "MiniMax music generation is still in progress and has no query endpoint."
        )
    if not audio:
        raise RuntimeError("MiniMax returned no audio data.")
    return audio, extra_info


def _download_audio(client: Any, url: str) -> bytes:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RuntimeError("MiniMax returned an invalid HTTPS audio URL.")
    chunks: list[bytes] = []
    total = 0
    with client.stream("GET", url) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > MAX_AUDIO_BYTES:
                raise RuntimeError("MiniMax audio download exceeds 128 MiB.")
            chunks.append(chunk)
    return b"".join(chunks)


def _audio_bytes(client: Any, value: str, output_format: str) -> bytes:
    if output_format == "url":
        return _download_audio(client, value)
    try:
        return bytes.fromhex("".join(value.split()))
    except ValueError:
        raise RuntimeError("MiniMax returned invalid hexadecimal audio data.") from None


def _metadata_integer(metadata: dict[str, Any], name: str, default: int) -> int:
    try:
        value = int(metadata.get(name, default))
    except (TypeError, ValueError):
        return int(default)
    return value if value > 0 else int(default)


def _decode_audio(
    content: bytes,
    audio_format: str,
    requested_sample_rate: int,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, int]:
    if not content:
        raise RuntimeError("MiniMax returned an empty audio payload.")
    if audio_format == "pcm":
        if len(content) % 2:
            raise RuntimeError("MiniMax returned an odd-length PCM payload.")
        channels = _metadata_integer(metadata, "music_channel", 1)
        raw = np.frombuffer(content, dtype="<i2")
        if raw.size % channels:
            raise RuntimeError("MiniMax PCM samples do not align with the channel count.")
        samples = raw.astype(np.float32).reshape(-1, channels) / 32768.0
        sample_rate = _metadata_integer(
            metadata,
            "music_sample_rate",
            requested_sample_rate,
        )
    else:
        soundfile = require_module("soundfile", "soundfile>=0.12")
        try:
            samples, sample_rate = soundfile.read(
                io.BytesIO(content),
                dtype="float32",
                always_2d=True,
            )
        except Exception as exc:
            detail = redact_sensitive(exc)
            raise RuntimeError(f"Could not decode MiniMax {audio_format} audio: {detail}") from None
        samples = np.asarray(samples, dtype=np.float32)
        sample_rate = int(sample_rate)
    if samples.ndim != 2 or not samples.size:
        raise RuntimeError("MiniMax decoded to an empty audio array.")
    if not np.isfinite(samples).all():
        raise RuntimeError("MiniMax decoded audio contains non-finite samples.")
    return np.ascontiguousarray(samples), int(sample_rate)


class MiniMaxMusicNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "region": (tuple(REGION_ENDPOINTS), {"default": "global_en"}),
                "model": (MUSIC_MODELS, {"default": DEFAULT_MODEL}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "lyrics": ("STRING", {"default": "", "multiline": True}),
                "stream": ("BOOLEAN", {"default": False}),
                "output_format": (OUTPUT_FORMATS, {"default": "hex"}),
                "audio_format": (AUDIO_FORMATS, {"default": "mp3"}),
                "sample_rate": (SAMPLE_RATES, {"default": 44100}),
                "bitrate": (BITRATES, {"default": 256000}),
                "lyrics_optimizer": ("BOOLEAN", {"default": False}),
                "is_instrumental": ("BOOLEAN", {"default": False}),
                "aigc_watermark": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Sent only to the cn_zh endpoint.",
                    },
                ),
            },
            "optional": {
                "audio_url": ("STRING", {"default": ""}),
                "audio_base64": ("STRING", {"default": "", "multiline": True}),
                "cover_feature_id": ("STRING", {"default": ""}),
                "timeout_seconds": (
                    "FLOAT",
                    {"default": 600.0, "min": 1.0, "max": 1800.0},
                ),
                "use_system_proxy": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_NAMES = ("wave_form", "sample_rate", "audio")
    RETURN_TYPES = (ANY, "INT", "AUDIO")
    OUTPUT_NODE = True
    FUNCTION = "generate_music"
    CATEGORY = "VLM Nodes/Audio"
    DESCRIPTION = (
        "Generate music or covers through fixed MiniMax regional endpoints. "
        f"The API key is read only from {API_KEY_ENV}."
    )

    def generate_music(
        self,
        region,
        model,
        prompt,
        lyrics,
        stream,
        output_format,
        audio_format,
        sample_rate,
        bitrate,
        lyrics_optimizer,
        is_instrumental,
        aigc_watermark,
        audio_url="",
        audio_base64="",
        cover_feature_id="",
        timeout_seconds=600.0,
        use_system_proxy=False,
    ):
        payload = build_music_request(
            region=region,
            model=model,
            prompt=prompt,
            lyrics=lyrics,
            stream=stream,
            output_format=output_format,
            audio_format=audio_format,
            sample_rate=sample_rate,
            bitrate=bitrate,
            lyrics_optimizer=lyrics_optimizer,
            is_instrumental=is_instrumental,
            aigc_watermark=aigc_watermark,
            audio_url=audio_url,
            audio_base64=audio_base64,
            cover_feature_id=cover_feature_id,
        )
        api_key = os.getenv(API_KEY_ENV, "").strip()
        if not api_key:
            raise ValueError(
                f"Set {API_KEY_ENV} in the environment that starts ComfyUI, "
                "then restart the server."
            )

        httpx = require_module("httpx", "httpx>=0.27,<1")
        client = httpx.Client(
            timeout=max(1.0, min(1800.0, float(timeout_seconds))),
            follow_redirects=False,
            trust_env=bool(use_system_proxy),
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            value, metadata = _request_audio_value(
                client,
                REGION_ENDPOINTS[region],
                headers,
                payload,
            )
            content = _audio_bytes(client, value, output_format)
            samples, actual_rate = _decode_audio(
                content,
                audio_format,
                int(sample_rate),
                metadata,
            )
            legacy = samples[:, 0] if samples.shape[1] == 1 else samples
            audio = {
                "waveform": torch.from_numpy(samples.T.copy()).unsqueeze(0),
                "sample_rate": actual_rate,
            }
            return (legacy.tolist(), actual_rate, audio)
        except Exception as exc:
            detail = redact_sensitive(exc, (api_key,))
            raise RuntimeError(f"MiniMax music request failed: {detail}") from None
        finally:
            try:
                client.close()
            except Exception:
                pass


NODE_CLASS_MAPPINGS = {"MiniMaxMusicNode": MiniMaxMusicNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxMusicNode": "MiniMax Music"}


__all__ = [
    "MiniMaxMusicNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
