import json
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ComfyUI_VLM_nodes.examples.robotics import lerobot_policy_server
from ComfyUI_VLM_nodes.nodes import robotics
from ComfyUI_VLM_nodes.nodes.robotics import (
    EMBODIMENT_SCHEMA,
    VLA_MODEL_CATALOG,
    RobotActions,
    actions_from_json,
    actions_from_response,
    blend_action_chunks,
    build_observation,
    make_embodiment,
    observation_to_groot_payload,
    observation_to_http_payload,
    observation_to_openpi_payload,
    render_action_preview,
    validate_action_trajectory,
    validate_policy_url,
    validate_ws_url,
    validate_zmq_host,
)


def _image(frames=1, height=24, width=32):
    return torch.linspace(0, 1, frames * height * width * 3).reshape(
        frames, height, width, 3
    )


def _joint_profile():
    return make_embodiment("Generic 7-DoF joint + gripper")


def _observation(profile=None):
    profile = profile or _joint_profile()
    return build_observation(
        task="Pick up the blue cube.",
        state_json=json.dumps([0.0] * profile.state_dim),
        primary_image=_image(frames=3),
        primary_camera=profile.camera_names[0],
        history_fps=15,
        timestamp=12.5,
        embodiment=profile,
    )


def test_embodiment_presets_are_explicit_roundtrippable_contracts():
    for preset in robotics.EMBODIMENT_PRESETS:
        profile = make_embodiment(preset)
        encoded = profile.to_dict()
        assert encoded["schema"] == EMBODIMENT_SCHEMA
        assert len(encoded["action_names"]) == len(encoded["action_min"])
        assert len(encoded["action_names"]) == len(encoded["action_max"])
        assert len(encoded["action_names"]) == len(encoded["max_delta_per_step"])
        assert robotics.RobotEmbodiment.from_dict(encoded) == profile
        assert "Template limits" in profile.notes


def test_embodiment_rejects_invalid_bounds_and_mismatched_overrides():
    with pytest.raises(ValueError, match="smaller"):
        robotics.RobotEmbodiment(
            name="bad",
            state_names=("s",),
            action_names=("a",),
            action_min=(1,),
            action_max=(0,),
            max_delta=(0.1,),
            control_hz=10,
            action_mode="absolute",
            camera_names=("front",),
        )
    with pytest.raises(ValueError, match="must contain 8"):
        make_embodiment(
            "Generic 7-DoF joint + gripper",
            action_min=[-1],
        )


def test_observation_preserves_history_and_validates_state_and_cameras():
    profile = _joint_profile()
    observation = _observation(profile)
    summary = observation.summary()
    assert summary["history_frames"] == 3
    assert summary["cameras"][profile.camera_names[0]]["width"] == 32
    assert summary["state"] == [0.0] * 8
    assert summary["task"] == "Pick up the blue cube."

    with pytest.raises(ValueError, match="State dimension"):
        build_observation(
            task="move",
            state_json="[0]",
            primary_image=_image(),
            primary_camera=profile.camera_names[0],
            history_fps=10,
            timestamp=0,
            embodiment=profile,
        )
    with pytest.raises(ValueError, match="not declared"):
        build_observation(
            task="move",
            state_json=json.dumps([0] * profile.state_dim),
            primary_image=_image(),
            primary_camera="unknown",
            history_fps=10,
            timestamp=0,
            embodiment=profile,
        )


def test_http_payload_is_bounded_and_contains_no_tensor_details():
    payload = observation_to_http_payload(_observation(), include_history=True)
    assert payload["schema"] == "comfyui-vlm/robot-observation"
    camera_frames = next(iter(payload["cameras"].values()))
    assert len(camera_frames) == 3
    assert all(item["encoding"] == "base64-jpeg" for item in camera_frames)
    assert "device" not in json.dumps(payload)


def test_openpi_payload_supports_flat_and_aloha_shapes():
    observation = _observation()
    flat = observation_to_openpi_payload(
        observation,
        layout="Flat keys (DROID / LIBERO)",
        state_key="observation/state",
        prompt_key="prompt",
    )
    camera_key = observation.images[0][0]
    assert flat[camera_key].shape == (24, 32, 3)
    assert flat[camera_key].dtype == np.uint8
    assert flat["observation/state"].dtype == np.float32

    nested = observation_to_openpi_payload(
        observation,
        layout="Nested images (ALOHA)",
        state_key="state",
        prompt_key="prompt",
    )
    assert nested["images"][camera_key].shape == (3, 24, 32)


def test_openpi_array_codec_roundtrips_and_rejects_object_arrays():
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    encoded = robotics._openpi_pack_array(source)
    decoded = robotics._openpi_unpack_array(encoded)
    assert np.array_equal(decoded, source)
    with pytest.raises(ValueError, match="does not support dtype"):
        robotics._openpi_pack_array(np.array([object()], dtype=object))
    forged = {
        b"__ndarray__": True,
        b"data": b"",
        b"dtype": "|O",
        b"shape": (0,),
    }
    with pytest.raises(ValueError, match="unsafe"):
        robotics._openpi_unpack_array(forged)


def test_openpi_client_protocol_and_token_redaction(monkeypatch):
    sent = []

    class Connection:
        responses = [b"metadata", b"actions"]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def recv(self, timeout):
            assert timeout == 5
            return self.responses.pop(0)

        def send(self, value):
            sent.append(value)

    connect_calls = []

    def connect(uri, **kwargs):
        connect_calls.append((uri, kwargs))
        return Connection()

    fake_ws = SimpleNamespace(connect=connect)

    class FakeMsgpack:
        @staticmethod
        def packb(value, default):
            assert value["prompt"] == "Pick up the blue cube."
            assert callable(default)
            return b"encoded-request"

        @staticmethod
        def unpackb(value, object_hook):
            assert callable(object_hook)
            if value == b"metadata":
                return {"model": "pi-test"}
            return {
                "actions": np.zeros((2, 8), dtype=np.float32),
                "server_timing": {"infer_ms": 12.5},
            }

    def fake_require(name, *_args):
        return fake_ws if name == "websockets.sync.client" else FakeMsgpack

    monkeypatch.setattr(robotics, "require_module", fake_require)
    monkeypatch.setenv("OPENPI_API_KEY", "openpi-secret")
    actions, report = robotics.call_openpi_policy(
        _observation(),
        endpoint="ws://127.0.0.1:8000",
        timeout_seconds=5,
        allow_remote=False,
        layout="Flat keys (DROID / LIBERO)",
        state_key="observation/state",
        prompt_key="prompt",
    )
    assert actions.values.shape == (2, 8)
    assert sent == [b"encoded-request"]
    assert connect_calls[0][1]["additional_headers"] == {
        "Authorization": "Api-Key openpi-secret"
    }
    assert report["server_metadata"] == {"model": "pi-test"}
    assert "openpi-secret" not in json.dumps(report)

    def broken_connect(*_args, **_kwargs):
        raise RuntimeError("token=openpi-secret")

    fake_ws.connect = broken_connect
    with pytest.raises(RuntimeError, match=r"token=\[REDACTED\]") as exc:
        robotics.call_openpi_policy(
            _observation(),
            endpoint="ws://127.0.0.1:8000",
            timeout_seconds=5,
            allow_remote=False,
            layout="Flat keys (DROID / LIBERO)",
            state_key="observation/state",
            prompt_key="prompt",
        )
    assert "openpi-secret" not in str(exc.value)


def test_groot_payload_uses_official_nested_batch_time_contract():
    observation = _observation()
    payload = observation_to_groot_payload(observation)
    camera = next(iter(payload["video"].values()))
    assert camera.shape == (1, 3, 24, 32, 3)
    assert camera.dtype == np.uint8
    assert payload["state"]["state"].shape == (1, 3, 8)
    assert payload["state"]["state"].dtype == np.float32
    assert payload["language"]["task"] == [["Pick up the blue cube."]]


def test_groot_array_codec_and_native_client(monkeypatch):
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    encoded = robotics._groot_encode(array)
    decoded = robotics._groot_decode(encoded)
    assert np.array_equal(decoded, array)
    with pytest.raises(TypeError, match="object/void"):
        robotics._groot_encode(np.array([object()], dtype=object))
    with pytest.raises(ValueError, match="object/void"):
        robotics._groot_decode({"nd": True, "kind": "O", "type": "|O"})

    sockets = []

    class Socket:
        def __init__(self):
            self.options = []
            self.connected = None
            self.request = None
            self.closed = False

        def setsockopt(self, key, value):
            self.options.append((key, value))

        def connect(self, value):
            self.connected = value

        def send(self, value):
            self.request = value

        def recv(self):
            return b"response"

        def close(self, linger):
            assert linger == 0
            self.closed = True

    class Context:
        terminated = False

        def socket(self, kind):
            assert kind == 1
            socket = Socket()
            sockets.append(socket)
            return socket

        def term(self):
            self.terminated = True

    fake_zmq = SimpleNamespace(
        REQ=1,
        RCVTIMEO=2,
        SNDTIMEO=3,
        LINGER=4,
        Context=Context,
    )
    packed_requests = []

    class FakeMsgpack:
        @staticmethod
        def packb(value, default):
            packed_requests.append(value)
            assert callable(default)
            return b"request"

        @staticmethod
        def unpackb(value, object_hook, raw):
            assert value == b"response"
            assert callable(object_hook)
            assert raw is False
            return [
                {
                    "arm": np.zeros((1, 3, 7), dtype=np.float32),
                    "gripper": np.ones((1, 3, 1), dtype=np.float32),
                },
                {"server": "ok"},
            ]

    def fake_require(name, *_args):
        return fake_zmq if name == "zmq" else FakeMsgpack

    monkeypatch.setattr(robotics, "require_module", fake_require)
    monkeypatch.setenv("GROOT_API_TOKEN", "groot-secret")
    actions, report = robotics.call_groot_policy(
        _observation(),
        host="127.0.0.1",
        port=5555,
        timeout_seconds=3,
        allow_remote=False,
    )
    assert actions.values.shape == (3, 8)
    assert actions.stream_slices == (("arm", 0, 7), ("gripper", 7, 8))
    assert packed_requests[0]["api_token"] == "groot-secret"
    assert sockets[0].connected == "tcp://127.0.0.1:5555"
    assert sockets[0].closed is True
    assert report["policy_info"] == {"server": "ok"}
    assert "groot-secret" not in json.dumps(report)


def test_action_response_parses_arrays_and_named_streams():
    single = actions_from_response(
        {"actions": [[[1, 2], [3, 4]]]},
        source="test",
    )
    assert single.values.shape == (2, 2)
    assert single.stream_slices == (("actions", 0, 2),)

    streams = actions_from_response(
        {
            "arm": np.zeros((1, 4, 7), dtype=np.float32),
            "gripper": np.ones((1, 4, 1), dtype=np.float32),
            "info": {"ignored": True},
        },
        source="groot",
    )
    assert streams.values.shape == (4, 8)
    assert streams.stream_slices == (("arm", 0, 7), ("gripper", 7, 8))


def test_actions_json_roundtrip_and_chunk_replanning():
    profile = _joint_profile()
    previous = RobotActions(
        torch.zeros((5, 8)),
        profile.action_names,
        "previous",
    )
    new = RobotActions(
        torch.ones((6, 8)),
        profile.action_names,
        "new",
    )
    parsed = actions_from_json(json.dumps(new.to_dict()))
    assert torch.equal(parsed.values, new.values)
    assert parsed.action_names == new.action_names

    replanned, report = blend_action_chunks(
        previous,
        new,
        executed_steps=2,
        transition_steps=2,
        max_horizon=4,
    )
    assert replanned.values.shape == (4, 8)
    assert torch.allclose(replanned.values[0], torch.full((8,), 1 / 3))
    assert torch.allclose(replanned.values[1], torch.full((8,), 2 / 3))
    assert torch.equal(replanned.values[2:], torch.ones((2, 8)))
    assert report["transition_steps_applied"] == 2
    with pytest.raises(ValueError, match="outside"):
        blend_action_chunks(
            previous,
            new,
            executed_steps=99,
            transition_steps=2,
            max_horizon=4,
        )


def test_action_safety_clamps_bounds_deltas_and_horizon():
    profile = _joint_profile()
    values = torch.tensor(
        [
            [4.0, 0, 0, 0, 0, 0, 0, 2.0],
            [-4.0, 0, 0, 0, 0, 0, 0, -2.0],
            [0.0, 0, 0, 0, 0, 0, 0, 0.5],
        ]
    )
    actions = RobotActions(
        values=values,
        action_names=profile.action_names,
        source="unit",
    )
    safe, report = validate_action_trajectory(
        actions,
        profile,
        mode="Clamp safely",
        execution_horizon=2,
        previous_action_json=json.dumps([0.0] * 8),
    )
    assert safe.horizon == 2
    assert torch.all(safe.values <= torch.tensor(profile.action_max))
    assert torch.all(safe.values >= torch.tensor(profile.action_min))
    limits = torch.tensor(profile.max_delta)
    previous = torch.zeros(8)
    for step in safe.values:
        assert torch.all((step - previous).abs() <= limits + 1.0e-6)
        previous = step
    assert report["violations"]["total"] > 0
    assert report["changed"] is True
    assert report["safe_for_handoff"] is True


def test_action_safety_blocks_or_holds_nonfinite_actions():
    profile = _joint_profile()
    values = torch.zeros((2, 8))
    values[0, 2] = float("nan")
    actions = RobotActions(values, profile.action_names, "unit")
    with pytest.raises(ValueError, match="blocked"):
        validate_action_trajectory(
            actions,
            profile,
            mode="Block unsafe",
            execution_horizon=2,
        )
    held, report = validate_action_trajectory(
        actions,
        profile,
        mode="Hold position on unsafe",
        execution_horizon=2,
        previous_action_json=json.dumps([0.25] * 8),
    )
    assert torch.allclose(held.values, torch.full((2, 8), 0.25))
    assert report["safe_for_handoff"] is True
    with pytest.raises(ValueError, match="requires previous_action_json"):
        validate_action_trajectory(
            actions,
            profile,
            mode="Hold position on unsafe",
            execution_horizon=2,
        )


def test_action_json_rejects_nonfinite_before_serialization():
    actions = RobotActions(
        torch.tensor([[float("inf")]]),
        ("action",),
        "unit",
    )
    with pytest.raises(ValueError, match="NaN or infinity"):
        actions.to_dict()


def test_policy_endpoint_security_defaults():
    assert (
        validate_policy_url("http://127.0.0.1:8787", allow_remote=False)
        == "http://127.0.0.1:8787/v1/infer"
    )
    assert validate_policy_url(
        "https://policy.example/v1/infer",
        allow_remote=True,
    ) == "https://policy.example/v1/infer"
    with pytest.raises(ValueError, match="HTTPS"):
        validate_policy_url("http://policy.example", allow_remote=True)
    with pytest.raises(ValueError, match="disabled"):
        validate_policy_url("https://policy.example", allow_remote=False)
    with pytest.raises(ValueError, match="embedded credentials"):
        validate_policy_url("https://secret@example.com", allow_remote=True)

    assert validate_ws_url("127.0.0.1:8000", allow_remote=False) == "ws://127.0.0.1:8000"
    with pytest.raises(ValueError, match="WSS"):
        validate_ws_url("ws://policy.example", allow_remote=True)
    assert validate_zmq_host("localhost", allow_remote=False) == "localhost"
    with pytest.raises(ValueError, match="disabled"):
        validate_zmq_host("policy.example", allow_remote=False)


class _PolicyHandler(BaseHTTPRequestHandler):
    observed_auth = None
    observed_payload = None

    def log_message(self, *_args):
        pass

    def do_POST(self):  # noqa: N802
        type(self).observed_auth = self.headers.get("Authorization")
        length = int(self.headers["Content-Length"])
        type(self).observed_payload = json.loads(self.rfile.read(length))
        body = json.dumps(
            {
                "actions": [[0.0] * 8, [0.1] * 8],
                "action_names": [f"a{index}" for index in range(8)],
            }
        ).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_http_policy_real_loopback_request_uses_env_token_without_leaking(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PolicyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("VLA_POLICY_TOKEN", "top-secret-value")
    try:
        actions, report = robotics.call_http_policy(
            _observation(),
            endpoint=f"http://127.0.0.1:{server.server_port}",
            timeout_seconds=5,
            allow_remote=False,
            include_history=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert actions.values.shape == (2, 8)
    assert _PolicyHandler.observed_auth == "Bearer top-secret-value"
    assert _PolicyHandler.observed_payload["task"] == "Pick up the blue cube."
    assert report["authenticated"] is True
    assert "top-secret-value" not in json.dumps(report)
    assert "top-secret-value" not in repr(actions)


def test_http_policy_error_redacts_server_token(monkeypatch):
    class BrokenClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def stream(self, *_args, **_kwargs):
            raise RuntimeError("Authorization: Bearer secret-http-token")

    monkeypatch.setenv("VLA_POLICY_TOKEN", "secret-http-token")
    monkeypatch.setattr(
        robotics,
        "require_module",
        lambda *_args: SimpleNamespace(Client=BrokenClient),
    )
    with pytest.raises(RuntimeError) as exc:
        robotics.call_http_policy(
            _observation(),
            endpoint="http://127.0.0.1:8787",
            timeout_seconds=5,
            allow_remote=False,
            include_history=False,
        )
    assert "secret-http-token" not in str(exc.value)
    assert "[REDACTED]" in str(exc.value)


def test_http_policy_rejects_declared_oversized_response(monkeypatch):
    class Response:
        headers = {
            "content-length": str(robotics.MAX_HTTP_RESPONSE_BYTES + 1),
        }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def raise_for_status(self):
            pass

        def iter_bytes(self):
            raise AssertionError("Oversized response body must not be read.")

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def stream(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(
        robotics,
        "require_module",
        lambda *_args: SimpleNamespace(Client=Client),
    )
    with pytest.raises(RuntimeError, match="32 MiB"):
        robotics.call_http_policy(
            _observation(),
            endpoint="http://127.0.0.1:8787",
            timeout_seconds=5,
            allow_remote=False,
            include_history=False,
        )


def test_catalog_has_unique_official_entries_and_clear_readiness():
    assert 12 <= len(VLA_MODEL_CATALOG) <= 30
    checkpoints = []
    for label, info in VLA_MODEL_CATALOG.items():
        assert label == info.label
        assert info.official_url.startswith("https://")
        assert info.backend
        assert info.status
        if info.checkpoint:
            checkpoints.append(info.checkpoint)
    assert len(checkpoints) == len(set(checkpoints))
    assert any(info.family == "SmolVLA" for info in VLA_MODEL_CATALOG.values())
    assert any(info.family == "Isaac GR00T N1.7" for info in VLA_MODEL_CATALOG.values())
    assert any(info.family == "OpenVLA-OFT" for info in VLA_MODEL_CATALOG.values())


def test_trajectory_preview_is_a_comfy_image():
    profile = _joint_profile()
    actions = RobotActions(
        torch.linspace(-0.5, 0.5, 5 * 8).reshape(5, 8),
        profile.action_names,
        "preview",
    )
    preview = render_action_preview(actions, embodiment=profile, width=640, height=320)
    assert preview.shape == (1, 320, 640, 3)
    assert preview.dtype == torch.float32
    assert 0 <= float(preview.min()) <= float(preview.max()) <= 1


def test_robotics_nodes_are_registered_and_have_safe_categories():
    expected = {
        "VLAEmbodimentProfile",
        "VLAObservationBuilder",
        "VLAHTTPPolicy",
        "VLAOpenPIWebSocketPolicy",
        "VLAGr00tZMQPolicy",
        "VLAActionSafety",
        "VLAActionsFromJSON",
        "VLAActionChunkReplan",
        "VLAActionInspect",
        "VLATrajectoryPreview",
        "VLAModelCatalog",
    }
    assert expected == set(robotics.NODE_CLASS_MAPPINGS)
    for node in robotics.NODE_CLASS_MAPPINGS.values():
        assert node.CATEGORY.startswith("VLM Nodes/Robotics")
        assert "forceInput" not in repr(node.INPUT_TYPES())


def test_lerobot_sidecar_is_packaged_and_avoids_pickle_transport():
    root = Path(robotics.__file__).parents[1]
    server = root / "examples" / "robotics" / "lerobot_policy_server.py"
    source = server.read_text(encoding="utf-8")
    compile(source, str(server), "exec")
    assert "pickle.loads" not in source
    assert "VLA_POLICY_TOKEN" in source
    assert "ThreadingHTTPServer" in source


def test_lerobot_sidecar_exposes_checkpoint_feature_contract(monkeypatch):
    visual = SimpleNamespace(
        type=SimpleNamespace(value="VISUAL"),
        shape=(3, 256, 256),
    )
    action = {"type": "ACTION", "shape": (6,)}
    assert lerobot_policy_server._feature_metadata(
        {"observation.images.camera1": visual, "action": action}
    ) == {
        "observation.images.camera1": {
            "type": "VISUAL",
            "shape": [3, 256, 256],
        },
        "action": {"type": "ACTION", "shape": [6]},
    }
    assert lerobot_policy_server._optional_config_int(
        SimpleNamespace(chunk_size=50), "chunk_size"
    ) == 50

    monkeypatch.setattr(
        lerobot_policy_server,
        "_decode_image",
        lambda _frame: np.zeros((1, 1, 3), dtype=np.uint8),
    )
    oversized_task = {
        "schema": "comfyui-vlm/robot-observation",
        "version": 1,
        "cameras": {
            "observation.images.front": [
                {"encoding": "base64-jpeg", "data": "unused"}
            ]
        },
        "state": [0.0],
        "task": "x" * (lerobot_policy_server.MAX_TASK_CHARS + 1),
    }
    with pytest.raises(ValueError, match="task must contain"):
        lerobot_policy_server._decode_observation(oversized_task)


def test_robotics_client_requirements_match_optional_extra():
    root = Path(robotics.__file__).parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"^robotics-client\s*=\s*\[(.*?)^\]", pyproject, re.M | re.S)
    assert match is not None
    optional_extra = set(re.findall(r'"([^"]+)"', match.group(1)))
    requirement_file = {
        line.strip()
        for line in (root / "requirements-robotics-client.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert optional_extra == requirement_file
