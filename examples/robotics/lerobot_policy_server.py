#!/usr/bin/env python
"""Isolated LeRobot policy server for the ComfyUI VLA HTTP node.

Run this file in a dedicated environment that contains LeRobot and the
policy-specific dependencies.  Do not install LeRobot's full dependency stack
into ComfyUI merely to use this bridge.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import io
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch
from PIL import Image

MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_CAMERAS = 16
MAX_FRAMES_PER_CAMERA = 256
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16 * 1024 * 1024
MAX_STATE_DIM = 2_048
MAX_ACTION_DIM = 2_048
MAX_TASK_CHARS = 16_384


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _decode_image(frame: dict[str, Any]) -> np.ndarray:
    if frame.get("encoding") != "base64-jpeg":
        raise ValueError("Only base64-jpeg camera frames are supported.")
    raw = base64.b64decode(frame["data"], validate=True)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Encoded camera frame exceeds the 8 MiB safety limit.")
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("Decoded camera frame exceeds the pixel safety limit.")
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _decode_observation(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "comfyui-vlm/robot-observation":
        raise ValueError("Unsupported observation schema.")
    if int(payload.get("version", 0)) != 1:
        raise ValueError("Unsupported observation schema version.")
    cameras = payload.get("cameras")
    if not isinstance(cameras, dict) or not 1 <= len(cameras) <= MAX_CAMERAS:
        raise ValueError("cameras must contain between 1 and 16 entries.")
    observation: dict[str, Any] = {}
    for key, encoded_frames in cameras.items():
        key = str(key).strip()
        if not key or len(key) > 256 or any(ord(char) < 32 for char in key):
            raise ValueError("Camera names must contain 1 to 256 printable characters.")
        if not isinstance(encoded_frames, list) or not (
            1 <= len(encoded_frames) <= MAX_FRAMES_PER_CAMERA
        ):
            raise ValueError(f"Camera {key!r} has an invalid history.")
        # Current LeRobot policy processors accept one current observation.
        # ComfyUI may send history for servers/models that use it; this generic
        # bridge deliberately selects the latest frame.
        array = _decode_image(encoded_frames[-1])
        tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32) / 255.0
        observation[key] = tensor.unsqueeze(0)
    state = np.asarray(payload.get("state"), dtype=np.float32)
    if (
        state.ndim != 1
        or not 1 <= state.size <= MAX_STATE_DIM
        or not np.isfinite(state).all()
    ):
        raise ValueError(f"state must contain 1 to {MAX_STATE_DIM} finite values.")
    observation["observation.state"] = torch.from_numpy(state).unsqueeze(0)
    task = str(payload.get("task", "")).strip()
    if not task or len(task) > MAX_TASK_CHARS:
        raise ValueError(f"task must contain 1 to {MAX_TASK_CHARS} characters.")
    observation["task"] = task
    return observation


def _postprocess_chunk(postprocessor, action: torch.Tensor) -> torch.Tensor:
    if action.ndim == 1:
        action = action.unsqueeze(0)
    if action.ndim == 2:
        # select_action normally returns [batch, dim].
        processed = postprocessor(action)
        if processed.ndim == 1:
            processed = processed.unsqueeze(0)
        return processed.unsqueeze(1) if processed.ndim == 2 else processed
    if action.ndim != 3:
        raise ValueError(f"Policy returned unsupported action shape {tuple(action.shape)}.")
    processed_steps = [postprocessor(action[:, index, :]) for index in range(action.shape[1])]
    return torch.stack(processed_steps, dim=1)


def _feature_metadata(features: Any) -> dict[str, dict[str, Any]]:
    """Return the portable part of a LeRobot policy feature contract."""

    result: dict[str, dict[str, Any]] = {}
    for key, feature in (features or {}).items():
        if isinstance(feature, dict):
            feature_type = feature.get("type")
            shape = feature.get("shape", ())
        else:
            feature_type = getattr(feature, "type", None)
            shape = getattr(feature, "shape", ())
        feature_type = getattr(feature_type, "value", feature_type)
        dimensions: list[int | str | None] = []
        for dimension in shape or ():
            if dimension is None:
                dimensions.append(None)
                continue
            try:
                dimensions.append(int(dimension))
            except (TypeError, ValueError):
                dimensions.append(str(dimension))
        result[str(key)] = {
            "type": str(feature_type) if feature_type is not None else "UNKNOWN",
            "shape": dimensions,
        }
    return result


def _optional_config_int(config: Any, name: str) -> int | None:
    value = getattr(config, name, None)
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


class PolicyRuntime:
    def __init__(
        self,
        *,
        policy_type: str,
        policy_path: str,
        revision: str | None,
        device: str,
        actions_per_chunk: int,
        idle_offload_seconds: float,
    ):
        self.policy_type = policy_type
        self.policy_path = policy_path
        self.revision = revision
        self.device = _device(device)
        self.actions_per_chunk = actions_per_chunk
        self.idle_offload_seconds = idle_offload_seconds
        self.lock = threading.Lock()
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.resident_device = "unloaded"
        self.last_request = 0.0
        self.load_seconds = 0.0
        self._load()
        if idle_offload_seconds > 0 and self.device != "cpu":
            threading.Thread(target=self._idle_worker, daemon=True).start()

    def _load(self) -> None:
        from lerobot.policies import get_policy_class, make_pre_post_processors

        started = time.perf_counter()
        policy_class = get_policy_class(self.policy_type)
        kwargs = {}
        if self.revision:
            kwargs["revision"] = self.revision
        self.policy = policy_class.from_pretrained(self.policy_path, **kwargs)
        self.policy.eval()
        self.policy.to(self.device)
        overrides = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.policy_path,
            pretrained_revision=self.revision,
            preprocessor_overrides={"device_processor": overrides},
            postprocessor_overrides={"device_processor": overrides},
        )
        self.resident_device = self.device
        self.last_request = time.monotonic()
        self.load_seconds = time.perf_counter() - started

    def _ensure_resident(self) -> None:
        if self.resident_device != self.device:
            self.policy.to(self.device)
            self.resident_device = self.device

    def _idle_worker(self) -> None:
        interval = min(max(self.idle_offload_seconds / 4, 1.0), 30.0)
        while True:
            time.sleep(interval)
            if time.monotonic() - self.last_request < self.idle_offload_seconds:
                continue
            if not self.lock.acquire(blocking=False):
                continue
            try:
                if (
                    self.resident_device != "cpu"
                    and time.monotonic() - self.last_request >= self.idle_offload_seconds
                ):
                    self.policy.to("cpu")
                    self.resident_device = "cpu"
            finally:
                self.lock.release()

    def metadata(self) -> dict[str, Any]:
        config = self.policy.config
        return {
            "protocol": "comfyui-vla-http-v1",
            "policy_type": self.policy_type,
            "policy_path": self.policy_path,
            "revision": self.revision,
            "configured_device": self.device,
            "resident_device": self.resident_device,
            "actions_per_chunk": self.actions_per_chunk,
            "idle_offload_seconds": self.idle_offload_seconds,
            "load_seconds": self.load_seconds,
            "policy_contract": {
                "input_features": _feature_metadata(
                    getattr(config, "input_features", None)
                ),
                "output_features": _feature_metadata(
                    getattr(config, "output_features", None)
                ),
                "observation_steps": _optional_config_int(config, "n_obs_steps"),
                "native_chunk_size": _optional_config_int(config, "chunk_size"),
                "native_action_steps": _optional_config_int(config, "n_action_steps"),
            },
        }

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        observation = _decode_observation(payload)
        with self.lock:
            self._ensure_resident()
            started = time.perf_counter()
            processed = self.preprocessor(observation)
            preprocess_ms = (time.perf_counter() - started) * 1000
            started_inference = time.perf_counter()
            with torch.inference_mode():
                predictor = getattr(self.policy, "predict_action_chunk", None)
                if callable(predictor):
                    action = predictor(processed)
                else:
                    action = self.policy.select_action(processed)
            inference_ms = (time.perf_counter() - started_inference) * 1000
            started_postprocess = time.perf_counter()
            action = _postprocess_chunk(self.postprocessor, action)
            if action.ndim == 3:
                if action.shape[0] != 1:
                    raise ValueError("Only policy batch size 1 is supported.")
                action = action[0]
            elif action.ndim == 1:
                action = action.unsqueeze(0)
            if action.ndim != 2:
                raise ValueError(f"Unexpected final action shape {tuple(action.shape)}.")
            action = action[: self.actions_per_chunk].detach().to("cpu", torch.float32)
            if not 1 <= int(action.shape[1]) <= MAX_ACTION_DIM:
                raise ValueError(
                    f"Policy action dimension must be in [1, {MAX_ACTION_DIM}]."
                )
            if not torch.isfinite(action).all():
                # Preserve the response for ComfyUI's safety node, but do not
                # serialize non-standard JSON numbers.
                raise ValueError("Policy returned NaN or infinite action values.")
            postprocess_ms = (time.perf_counter() - started_postprocess) * 1000
            self.last_request = time.monotonic()
        return {
            "actions": action.tolist(),
            "server_timing": {
                "preprocess_ms": preprocess_ms,
                "infer_ms": inference_ms,
                "postprocess_ms": postprocess_ms,
            },
            "policy": {
                "type": self.policy_type,
                "path": self.policy_path,
                "device": self.device,
            },
        }


class PolicyHandler(BaseHTTPRequestHandler):
    server_version = "ComfyUI-VLA-Policy/1"

    def log_message(self, format_string: str, *args: Any) -> None:
        # The request path is safe to log. Headers and bodies may contain
        # credentials or camera/state data and are intentionally excluded.
        print(f"{self.address_string()} - {format_string % args}")

    @property
    def runtime(self) -> PolicyRuntime:
        return self.server.runtime

    @property
    def token(self) -> str:
        return self.server.token

    def _authorized(self) -> bool:
        if not self.token:
            return True
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.token}"
        return hmac.compare_digest(supplied, expected)

    def _send(self, status: HTTPStatus, value: Any) -> None:
        body = _json_bytes(value)
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/healthz", "/v1/metadata"}:
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if not self._authorized():
            self._send(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        if self.path == "/healthz":
            self._send(HTTPStatus.OK, {"status": "ok"})
        else:
            self._send(HTTPStatus.OK, self.runtime.metadata())

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/infer":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if not self._authorized():
            self._send(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= content_length <= MAX_REQUEST_BYTES:
                raise ValueError("Request body size is invalid.")
            body = self.rfile.read(content_length)
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            result = self.runtime.infer(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)[:1000]})
            return
        except Exception as exc:
            # Do not return tracebacks, request data, environment variables, or
            # authorization headers across the network.
            self._send(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(exc).__name__}: {str(exc)[:800]}"},
            )
            return
        self._send(HTTPStatus.OK, result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve one LeRobot policy through the ComfyUI VLA HTTP protocol."
    )
    parser.add_argument("--policy-type", required=True, help="LeRobot policy type, e.g. smolvla")
    parser.add_argument("--policy-path", required=True, help="Hub repo id or local checkpoint")
    parser.add_argument("--revision", default=None, help="Optional immutable Hub revision")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, xpu, or cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--actions-per-chunk", type=int, default=16)
    parser.add_argument(
        "--idle-offload-seconds",
        type=float,
        default=0.0,
        help="Move the policy to CPU after this idle period; 0 keeps it resident.",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65_535:
        parser.error("--port must be in [1, 65535]")
    if not 1 <= args.actions_per_chunk <= 4096:
        parser.error("--actions-per-chunk must be in [1, 4096]")
    if args.idle_offload_seconds < 0:
        parser.error("--idle-offload-seconds must be non-negative")

    runtime = PolicyRuntime(
        policy_type=args.policy_type,
        policy_path=args.policy_path,
        revision=args.revision,
        device=args.device,
        actions_per_chunk=args.actions_per_chunk,
        idle_offload_seconds=args.idle_offload_seconds,
    )
    token = os.environ.get("VLA_POLICY_TOKEN", "").strip()
    server = ThreadingHTTPServer((args.host, args.port), PolicyHandler)
    server.runtime = runtime
    server.token = token
    print(
        f"Policy ready at http://{args.host}:{args.port}/v1/infer "
        f"(type={args.policy_type}, device={runtime.device}, auth={'on' if token else 'off'})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
