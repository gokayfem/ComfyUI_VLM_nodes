import asyncio
import importlib
import inspect
import json
import sys
import types
from dataclasses import dataclass

import pytest
import torch

PACKAGE = __package__.split(".")[0] if __package__ else "ComfyUI_VLM_nodes"
module = importlib.import_module(f"{PACKAGE}.nodes.moondream31")
worker = importlib.import_module(f"{PACKAGE}.nodes.moondream31_worker")

Moondream31Detect = module.Moondream31Detect
Moondream31Loader = module.Moondream31Loader
Moondream31Model = module.Moondream31Model
Moondream31Segment = module.Moondream31Segment
svg_path_to_mask = module.svg_path_to_mask


def _fake_model(handler, model_name=module.MODEL_ID):
    model = object.__new__(Moondream31Model)
    model.config = module.Moondream31Config(
        model=model_name,
        device="cuda",
        max_batch_size=4,
        kv_cache_pages=8192,
    )
    model.request = handler
    model.close = lambda: None
    return model


def test_svg_path_is_transformed_from_bbox_space_to_image_pixels():
    mask, polygon, contours = svg_path_to_mask(
        "M 0 0 H 1 V 1 H 0 Z",
        {"x_min": 0.25, "y_min": 0.25, "x_max": 0.75, "y_max": 0.75},
        100,
        80,
        supersample=4,
    )
    assert mask.shape == (80, 100)
    assert mask[40, 50] > 0.99
    assert mask[5, 5] == 0
    assert mask.sum().item() == pytest.approx(2000, rel=0.06)
    assert len(polygon) >= 4
    assert len(contours) == 1
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    assert min(xs) == pytest.approx(25)
    assert max(xs) == pytest.approx(75)
    assert min(ys) == pytest.approx(20)
    assert max(ys) == pytest.approx(60)


def test_svg_curves_and_evenodd_holes_are_preserved():
    path = "M 0 0 H 1 V 1 H 0 Z M .25 .25 C .4 .1 .6 .1 .75 .25 V .75 H .25 Z"
    mask, polygon, contours = svg_path_to_mask(
        path,
        {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
        128,
        128,
        supersample=4,
        precision_px=0.5,
    )
    assert len(contours) == 2
    assert len(polygon) >= 4
    assert mask[8, 8] > 0.99
    assert mask[64, 64] < 0.01


@pytest.mark.parametrize(
    ("path", "bbox", "message"),
    [
        ("", {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}, "empty"),
        (
            "M 0 0 L nan 1 Z",
            {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
            "invalid",
        ),
        (
            "M 0 0 H 1 V 1 Z",
            {"x_min": 0.7, "y_min": 0, "x_max": 0.2, "y_max": 1},
            "positive",
        ),
    ],
)
def test_svg_rejects_malformed_or_unsafe_geometry(path, bbox, message):
    with pytest.raises((TypeError, ValueError), match=message):
        svg_path_to_mask(path, bbox, 64, 64)


def test_video_detect_uses_stride_parallelism_and_reports_measured_fps():
    observed = {}

    def request(operation, **payload):
        observed["operation"] = operation
        observed.update(payload)
        return {
            "items": [
                {
                    "objects": [
                        {
                            "x_min": 0.1,
                            "y_min": 0.2,
                            "x_max": 0.4,
                            "y_max": 0.6,
                        }
                    ]
                },
                {"objects": []},
            ],
            "elapsed_seconds": 0.1,
            "parallel_requests": 2,
        }

    images = torch.zeros((4, 48, 64, 3), dtype=torch.float32)
    outputs = Moondream31Detect().detect(
        _fake_model(request),
        images,
        "person",
        30.0,
        2,
        2,
        20,
        False,
    )
    sequence = outputs[0]
    performance = json.loads(outputs[-1])
    assert observed["operation"] == "detect"
    assert len(observed["images"]) == 2
    assert observed["parallel_requests"] == 2
    assert sequence.frame_count == 4
    assert [frame.frame_index for frame in sequence.frames] == [0, 2]
    assert sequence.frames[0].detections[0].bbox_xyxy == pytest.approx(
        (6.4, 9.6, 25.6, 28.8)
    )
    assert outputs[2].shape == images.shape
    assert outputs[3].shape == (4, 48, 64)
    assert performance["processed_frames"] == 2
    assert performance["worker_fps"] == pytest.approx(20)
    assert performance["target_processed_fps"] == pytest.approx(15)
    assert performance["parallel_requests"] == 2


def test_segment_exposes_svg_mask_cutout_overlay_and_structured_detection():
    def request(operation, **payload):
        assert operation == "segment"
        assert payload["spatial_refs"] == [[0.5, 0.5]]
        return {
            "items": [
                {
                    "path": "M 0 0 H 1 V 1 H 0 Z",
                    "bbox": {
                        "x_min": 0.25,
                        "y_min": 0.25,
                        "x_max": 0.75,
                        "y_max": 0.75,
                    },
                }
            ],
            "elapsed_seconds": 0.2,
            "parallel_requests": 1,
        }

    image = torch.ones((1, 32, 40, 3), dtype=torch.float32)
    outputs = Moondream31Segment().segment(
        _fake_model(request, module.PREVIEW_MODEL_ID),
        image,
        "object",
        1.0,
        1,
        1,
        4,
        False,
        spatial_refs_json="[[0.5, 0.5]]",
    )
    sequence = outputs[0]
    native = json.loads(outputs[2])
    mask = outputs[3]
    mask_image = outputs[4]
    cutout = outputs[5]
    overlay = outputs[6]
    detection = sequence.frames[0].detections[0]
    assert native[0]["path"].startswith("M 0 0")
    assert mask.shape == (1, 32, 40)
    assert mask_image.shape == (1, 32, 40, 3)
    assert cutout.shape == image.shape
    assert overlay.shape == image.shape
    assert mask[0, 16, 20] > 0.99
    assert mask[0, 2, 2] == 0
    assert cutout[0, 16, 20].min() > 0.99
    assert cutout[0, 2, 2].max() == 0
    assert detection.mask is not None
    assert detection.polygon is not None
    assert detection.metadata["native_svg_path"].startswith("M 0 0")


def test_license_gate_and_node_registration():
    with pytest.raises(ValueError, match="License"):
        Moondream31Loader().load(
            False,
            "Auto",
            4,
            "Balanced (8K pages)",
        )
    assert set(module.NODE_CLASS_MAPPINGS) == {
        "Moondream31Loader",
        "Moondream31Query",
        "Moondream31Caption",
        "Moondream31Detect",
        "Moondream31Point",
        "Moondream31Segment",
    }
    assert all(
        node.CATEGORY == "VLM Nodes/Moondream 3"
        for node in module.NODE_CLASS_MAPPINGS.values()
    )


def test_final_31_model_does_not_claim_preview_svg_segment():
    with pytest.raises(ValueError, match="3 Preview"):
        Moondream31Segment().segment(
            _fake_model(lambda *_args, **_kwargs: {}),
            torch.zeros((1, 16, 16, 3)),
            "object",
            1.0,
            1,
            1,
            1,
            False,
        )


def test_worker_auth_is_not_exposed_in_process_arguments_and_logs_are_redacted(
    tmp_path,
    monkeypatch,
):
    source = inspect.getsource(Moondream31Model.ensure_started)
    assert '"--auth-key"' not in source
    assert "MOONDREAM_WORKER_AUTH" in inspect.getsource(
        module._worker_environment
    )
    log = tmp_path / "worker.log"
    log.write_text(
        "api_key=secret-value\nAuthorization: bearer-value\nCUDA error",
        encoding="utf-8",
    )
    tail = module._safe_log_tail(log)
    assert "secret-value" not in tail
    assert "bearer-value" not in tail
    assert "CUDA error" in tail

    monkeypatch.setenv("PATH", "/runtime/bin")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross")
    monkeypatch.setenv("HF_TOKEN", "hf-server-side")
    monkeypatch.setenv("MOONDREAM_API_KEY", "adapter-only")
    monkeypatch.setenv("HTTPS_PROXY", "https://user:password@example.test")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    base_environment = module._worker_environment(
        tmp_path,
        b"\x01" * 32,
        module.MODEL_ID,
    )
    assert base_environment["PATH"] == "/runtime/bin"
    assert base_environment["HF_TOKEN"] == "hf-server-side"
    assert "OPENAI_API_KEY" not in base_environment
    assert "MOONDREAM_API_KEY" not in base_environment
    assert "HTTPS_PROXY" not in base_environment
    assert "PYTORCH_ALLOC_CONF" not in base_environment
    assert "PYTORCH_CUDA_ALLOC_CONF" not in base_environment
    assert base_environment["MOONDREAM_WORKER_AUTH"] == "01" * 32

    adapter_environment = module._worker_environment(
        tmp_path,
        b"\x02" * 32,
        f"{module.MODEL_ID}/adapter@step",
    )
    assert adapter_environment["MOONDREAM_API_KEY"] == "adapter-only"


def test_runtime_python_preserves_virtualenv_symlink(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    binary = tmp_path / "base-python"
    binary.write_text("", encoding="utf-8")
    venv_python = root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    try:
        venv_python.symlink_to(binary)
    except OSError:
        pytest.skip("This filesystem cannot create symlinks.")
    monkeypatch.delenv("MOONDREAM_PYTHON", raising=False)
    selected = module._runtime_python(root)
    assert selected == venv_python.absolute()
    assert selected != binary.resolve()


def test_worker_registers_official_31_id_only_when_upstream_is_missing(
    monkeypatch,
):
    @dataclass(frozen=True)
    class Spec:
        name: str
        repo_id: str
        filename: str
        checkpoint_format: str

    registry = {
        "moondream3-preview": Spec(
            "moondream3-preview",
            "moondream/moondream3-preview",
            "model_fp8.pt",
            "md3",
        )
    }
    fake = types.ModuleType("kestrel.models")
    fake.get_spec = lambda name: (
        registry[name] if name in registry else (_ for _ in ()).throw(ValueError(name))
    )
    fake.register = lambda spec: registry.__setitem__(spec.name, spec)
    monkeypatch.setitem(sys.modules, "kestrel.models", fake)

    assert worker._register_moondream31_if_needed("moondream3.1-9B-A2B")
    registered = registry["moondream3.1-9B-A2B"]
    assert registered.repo_id == "moondream/moondream3.1-9B-A2B"
    assert registered.filename == "model.safetensors"
    assert registered.checkpoint_format == "md3"
    assert not worker._register_moondream31_if_needed("moondream3.1-9B-A2B")
    assert not worker._register_moondream31_if_needed("custom-model")


def test_worker_honors_do_not_track_for_base_models(monkeypatch):
    class SimpleClient:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    class Reporter:
        def __init__(self):
            self._client = SimpleClient()

    fake = types.ModuleType("kestrel.photon")
    fake.PhotonReporter = Reporter
    monkeypatch.setitem(sys.modules, "kestrel.photon", fake)
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.delenv("MOONDREAM_API_KEY", raising=False)

    assert worker._honor_do_not_track()
    reporter = Reporter()
    assert asyncio.run(reporter.validate_api_key()) is False
    assert reporter.start() is None
    asyncio.run(reporter.shutdown())
    assert reporter._client.closed

    monkeypatch.setenv("MOONDREAM_API_KEY", "finetune-key")
    assert not worker._honor_do_not_track()
