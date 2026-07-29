"""Isolated Moondream 3.1 Photon worker.

This file is launched directly by the ComfyUI process with the dedicated
Moondream virtual environment.  It intentionally has no imports from ComfyUI
or this package: Moondream pins a Pillow version that is incompatible with
current ComfyUI releases, so sharing one Python environment is unsafe.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from multiprocessing.connection import Client
from typing import Any

from PIL import Image


def _honor_do_not_track() -> bool:
    """Disable anonymous Photon reporting when the sidecar requests privacy.

    Kestrel 0.4.2 does not currently inspect the conventional DO_NOT_TRACK
    environment variable. Base-model inference does not need its reporter, so
    keep validation local, skip the telemetry loop, and still close the HTTP
    client during engine shutdown. Finetune inference retains upstream auth
    and reporting behavior because it explicitly receives an API key.
    """

    if os.environ.get("DO_NOT_TRACK") != "1":
        return False
    if os.environ.get("MOONDREAM_API_KEY", "").strip():
        return False

    from kestrel.photon import PhotonReporter

    async def validate_api_key(self) -> bool:
        return False

    def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        await self._client.aclose()

    PhotonReporter.validate_api_key = validate_api_key
    PhotonReporter.start = start
    PhotonReporter.shutdown = shutdown
    return True


def _register_moondream31_if_needed(model_name: str) -> bool:
    """Bridge the official model-card ID on runtimes released before the ID.

    Moondream 3.1 uses the same MD3 Photon runtime/checkpoint format as the
    preview. Stable moondream 1.3.0 / kestrel 0.4.2 shipped the safetensors
    loader but omitted the new registry entry published by the later model
    card. Prefer an upstream entry whenever present; otherwise clone only the
    runtime metadata and point it at the official 3.1 weights.
    """

    if model_name != "moondream3.1-9B-A2B":
        return False
    from kestrel.models import get_spec, register

    try:
        get_spec(model_name)
        return False
    except ValueError:
        preview = get_spec("moondream3-preview")
        register(
            replace(
                preview,
                name=model_name,
                repo_id="moondream/moondream3.1-9B-A2B",
                filename="model.safetensors",
                checkpoint_format="md3",
            )
        )
        return True


def _base_model_name(value: str) -> str:
    return str(value).split("/", 1)[0]


def _model_skills(model_name: str) -> frozenset[str]:
    base_model = _base_model_name(model_name)
    if base_model == "moondream3.1-9B-A2B":
        # Source of truth: the final 3.1 model card. Segment remains a skill
        # of the 3 Preview and cloud API, not the final local 3.1 checkpoint.
        return frozenset(("caption", "query", "detect", "point"))
    from kestrel.models import get_spec

    spec = get_spec(base_model)
    templates = spec.default_config.get("tokenizer", {}).get("templates", {})
    return frozenset(
        name for name, template in templates.items() if template is not None
    )


def _image(value: bytes) -> Image.Image:
    if not isinstance(value, bytes):
        raise TypeError("Worker image payloads must be bytes.")
    with Image.open(BytesIO(value)) as source:
        return source.convert("RGB")


def _parallel(
    images: list[bytes],
    operation: Callable[[Image.Image], dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    if not images:
        return []
    worker_count = max(1, min(int(workers), len(images)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        return list(pool.map(lambda value: operation(_image(value)), images))


def _private_shutdown(model: Any) -> None:
    """Best-effort graceful Photon shutdown before the process exits.

    The public moondream package currently has no close method.  Process
    isolation remains the hard guarantee: the parent terminates this exact
    process if this best-effort private cleanup ever changes or stalls.
    """

    engine = getattr(model, "_engine", None)
    loop = getattr(model, "_loop", None)
    thread = getattr(model, "_thread", None)
    if engine is not None and loop is not None:
        try:
            import asyncio

            asyncio.run_coroutine_threadsafe(engine.shutdown(), loop).result(timeout=20)
        except Exception:  # noqa: BLE001,S110 - private best-effort cleanup.
            pass
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:  # noqa: BLE001,S110 - private best-effort cleanup.
            pass
    if thread is not None:
        try:
            thread.join(timeout=5)
        except Exception:  # noqa: BLE001,S110 - private best-effort cleanup.
            pass


def _request(
    model: Any,
    request: dict[str, Any],
    send: Callable[[dict[str, Any]], None],
    max_batch_size: int,
    supported_skills: frozenset[str],
) -> bool:
    request_id = request.get("id")
    operation = request.get("operation")
    if operation == "shutdown":
        send({"id": request_id, "type": "result", "result": {"closed": True}})
        return False
    if operation not in supported_skills:
        raise ValueError(
            f"Model does not support the {operation!r} skill. "
            f"Available skills: {', '.join(sorted(supported_skills))}."
        )

    started = time.perf_counter()
    settings = {"max_tokens": int(request.get("max_tokens", 512))}
    if operation in {"query", "caption"}:
        image_payload = request.get("image")
        image = _image(image_payload) if image_payload is not None else None
        if operation == "query":
            output = model.query(
                image=image,
                question=str(request["question"]),
                stream=bool(request.get("stream", True)),
                settings=settings,
                reasoning=bool(request.get("reasoning", False)),
            )
            key = "answer"
        else:
            if image is None:
                raise ValueError("Caption requires an image.")
            output = model.caption(
                image=image,
                length=str(request.get("length", "normal")),
                stream=bool(request.get("stream", True)),
                settings=settings,
            )
            key = "caption"

        value = output[key]
        if isinstance(value, str):
            text = value
        else:
            chunks = []
            for chunk in value:
                chunk_text = str(chunk)
                chunks.append(chunk_text)
                send(
                    {
                        "id": request_id,
                        "type": "chunk",
                        "text": chunk_text,
                    }
                )
            text = "".join(chunks)
        result = {
            key: text,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if operation == "query" and output.get("reasoning") is not None:
            result["reasoning"] = output["reasoning"]
        send({"id": request_id, "type": "result", "result": result})
        return True

    images = request.get("images")
    if not isinstance(images, list):
        raise TypeError(f"{operation} requires an image list.")
    workers = min(
        max_batch_size,
        max(1, int(request.get("parallel_requests", max_batch_size))),
    )
    object_prompt = str(request.get("object", "")).strip()
    if not object_prompt:
        raise ValueError(f"{operation} requires a non-empty object prompt.")

    if operation == "detect":
        results = _parallel(
            images,
            lambda image: model.detect(image, object_prompt, settings=settings),
            workers,
        )
    elif operation == "point":
        results = _parallel(
            images,
            lambda image: model.point(image, object_prompt, settings=settings),
            workers,
        )
    elif operation == "segment":
        spatial_refs = request.get("spatial_refs") or None
        results = _parallel(
            images,
            lambda image: model.segment(
                image,
                object_prompt,
                spatial_refs=spatial_refs,
                stream=False,
                settings=settings,
            ),
            workers,
        )
    else:
        raise ValueError(f"Unknown worker operation {operation!r}.")

    send(
        {
            "id": request_id,
            "type": "result",
            "result": {
                "items": results,
                "processed_frames": len(images),
                "parallel_requests": workers,
                "elapsed_seconds": time.perf_counter() - started,
            },
        }
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--auth-key")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-batch-size", type=int, required=True)
    parser.add_argument("--kv-cache-pages", type=int, default=0)
    args = parser.parse_args()

    auth_key = args.auth_key or os.environ.pop("MOONDREAM_WORKER_AUTH", "")
    if not auth_key:
        parser.error("worker authentication is missing")
    connection = Client(
        (args.host, args.port),
        authkey=bytes.fromhex(auth_key),
    )

    def send(value: dict[str, Any]) -> None:
        connection.send(value)

    send(
        {
            "type": "status",
            "status": "loading",
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pid": os.getpid(),
        }
    )

    model = None
    try:
        import moondream as md

        base_model = _base_model_name(args.model)
        compatibility_registration = _register_moondream31_if_needed(base_model)
        telemetry_disabled = _honor_do_not_track()
        supported_skills = _model_skills(args.model)
        kwargs: dict[str, Any] = {
            "local": True,
            "model": args.model,
            "device": args.device,
            "max_batch_size": args.max_batch_size,
        }
        if args.kv_cache_pages > 0:
            kwargs["kv_cache_pages"] = args.kv_cache_pages
        model = md.vl(**kwargs)
        try:
            package_version = version("moondream")
        except PackageNotFoundError:
            package_version = "unknown"
        send(
            {
                "type": "status",
                "status": "ready",
                "moondream_version": package_version,
                "compatibility_registration": compatibility_registration,
                "telemetry_disabled": telemetry_disabled,
                "skills": sorted(supported_skills),
                "pid": os.getpid(),
            }
        )

        running = True
        while running:
            request = connection.recv()
            request_id = request.get("id") if isinstance(request, dict) else None
            try:
                if not isinstance(request, dict):
                    raise TypeError("Worker requests must be dictionaries.")
                running = _request(
                    model,
                    request,
                    send,
                    args.max_batch_size,
                    supported_skills,
                )
            except Exception as exc:  # noqa: BLE001 - report request failures over IPC.
                send(
                    {
                        "id": request_id,
                        "type": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=12),
                    }
                )
    except Exception as exc:  # noqa: BLE001 - report startup failures over IPC.
        send(
            {
                "type": "fatal",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=20),
            }
        )
        return 1
    finally:
        if model is not None:
            _private_shutdown(model)
        try:
            connection.close()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
