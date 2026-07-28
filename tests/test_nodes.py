import base64
import inspect
import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import ComfyUI_VLM_nodes as package
from ComfyUI_VLM_nodes.nodes import (
    audioldm2,
    florence2,
    modern_vlm,
    paligemma,
    qwen2vl,
)
from ComfyUI_VLM_nodes.nodes.runtime import (
    accelerator_backend,
    external_device_map,
    image_data_uri,
    pil_mask_to_tensor,
    pil_to_tensor,
    runtime_diagnostics,
    tensor_batch_to_pil,
    torch_dtype,
)


def test_every_module_imports_and_expected_nodes_exist():
    assert package.IMPORT_ERRORS == {}
    expected = {
        "ModernVLM",
        "VLMRuntimeDiagnostics",
        "Florence2",
        "Paligemma",
        "MolmoNode",
        "Qwen2VLNode",
        "Moondream2model",
        "MiniCPMNode",
    }
    assert expected <= package.NODE_CLASS_MAPPINGS.keys()


def test_node_schemas_do_not_use_force_input():
    for node_class in package.NODE_CLASS_MAPPINGS.values():
        schema = node_class.INPUT_TYPES()
        assert "forceInput" not in repr(schema)


def test_source_has_no_runtime_installer_or_direct_cuda_cache():
    root = Path(package.__file__).parent
    source = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (root / "nodes").rglob("*.py")
    )
    assert "torch.cuda.empty_cache" not in source
    assert "subprocess.run" not in source
    assert "pip install" not in source


def test_portable_device_dtype_and_backend_contracts(monkeypatch):
    assert torch_dtype("float16", torch.device("cpu")) == torch.float32
    assert torch_dtype("float16", torch.device("mps")) == torch.float16
    assert torch_dtype("float16", torch.device("xpu")) == torch.float16
    assert accelerator_backend(torch.device("mps")) == "apple-metal"
    assert accelerator_backend(torch.device("xpu")) == "intel-xpu"

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert accelerator_backend(torch.device("cuda")) == "nvidia-cuda"
    monkeypatch.setattr(torch.version, "hip", "7.2", raising=False)
    assert accelerator_backend(torch.device("cuda")) == "amd-rocm"


def test_runtime_report_and_device_map_are_supportable():
    report = runtime_diagnostics()
    assert {
        "platform",
        "machine",
        "python",
        "torch",
        "device",
        "backend",
        "bf16",
        "torch_cuda",
        "torch_hip",
        "packages",
    } <= report.keys()
    device_map = external_device_map()
    assert set(device_map) == {""}
    assert device_map[""] == report["device"]


def test_dependency_metadata_matches_installer_requirements():
    try:
        import tomllib
    except ModuleNotFoundError:
        pytest.skip("tomllib is built into Python 3.11+")
    from packaging.requirements import Requirement

    root = Path(package.__file__).parent
    metadata = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))
    project_requirements = {
        str(Requirement(value)) for value in metadata["project"]["dependencies"]
    }
    installer_requirements = {
        str(Requirement(line))
        for line in (root / "requirements.txt").read_text("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert project_requirements == installer_requirements

    bitsandbytes = next(
        Requirement(value)
        for value in metadata["project"]["dependencies"]
        if Requirement(value).name == "bitsandbytes"
    )
    assert bitsandbytes.marker is not None
    supported = (
        ("linux", "x86_64"),
        ("linux", "aarch64"),
        ("win32", "AMD64"),
        ("win32", "ARM64"),
        ("darwin", "arm64"),
    )
    unsupported = (
        ("darwin", "x86_64"),
        ("linux", "ppc64le"),
    )
    for system, machine in supported:
        assert bitsandbytes.marker.evaluate(
            {"sys_platform": system, "platform_machine": machine}
        )
    for system, machine in unsupported:
        assert not bitsandbytes.marker.evaluate(
            {"sys_platform": system, "platform_machine": machine}
        )


def test_image_roundtrip_and_png_data_uri():
    tensor = torch.tensor(
        [[[[0.0, 0.5, 1.0], [1.0, float("nan"), 0.0]]]],
        dtype=torch.float32,
    )
    images = tensor_batch_to_pil(tensor)
    assert images[0].size == (2, 1)
    uri = image_data_uri(images[0])
    payload = base64.b64decode(uri.split(",", 1)[1])
    assert Image.open(io.BytesIO(payload)).format == "PNG"
    assert pil_to_tensor(images[0]).shape == (1, 1, 2, 3)
    assert pil_mask_to_tensor(Image.new("L", (2, 3))).shape == (1, 3, 2)


def test_paligemma_parser_uses_normalized_boxes_and_16_codes():
    codes = "".join(f"<seg{index:03d}>" for index in range(16))
    parsed = paligemma.parse_segments(
        f"<loc0100><loc0200><loc0900><loc0800>{codes} cat"
    )
    assert len(parsed) == 1
    box, values, label = parsed[0]
    assert box == pytest.approx((100 / 1024, 200 / 1024, 900 / 1024, 800 / 1024))
    assert values == list(range(16))
    assert label == "cat"


def test_florence_rendering_supports_boxes_quads_and_nested_polygons():
    image = Image.new("RGB", (32, 24), "black")
    parsed = {
        "<TASK>": {
            "bboxes": [[1, 1, 10, 10]],
            "labels": ["box"],
            "quad_boxes": [[2, 2, 8, 2, 8, 8, 2, 8]],
            "polygons": [[[4, 4, 20, 4, 20, 20, 4, 20]]],
        }
    }
    mask, visual = florence2._visualize(image, parsed)
    assert np.asarray(mask).max() == 255
    assert visual.size == image.size


def test_modern_catalog_has_current_quality_and_low_vram_tiers():
    repositories = {spec.repo_id for spec in modern_vlm.MODEL_CATALOG.values()}
    small_fast = [
        spec for spec in modern_vlm.MODEL_CATALOG.values() if spec.small_fast
    ]
    assert 10 <= len(small_fast) <= 20
    assert all(
        not spec.trust_remote_code
        for spec in modern_vlm.MODEL_CATALOG.values()
        if spec.family != "Custom"
    )
    assert modern_vlm.MODEL_CATALOG[
        "Custom Hugging Face model"
    ].trust_remote_code
    assert "Qwen/Qwen3.5-4B" in repositories
    assert "Qwen/Qwen3.5-35B-A3B" in repositories
    assert "Qwen/Qwen3.6-27B" in repositories
    assert "Qwen/Qwen3-VL-8B-Instruct" in repositories
    assert "Qwen/Qwen2.5-VL-3B-Instruct" in repositories
    assert "google/gemma-3-4b-it" in repositories
    assert "HuggingFaceTB/SmolVLM2-256M-Video-Instruct" in repositories
    assert "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" in repositories
    assert "LiquidAI/LFM2.5-VL-450M" in repositories
    assert "LiquidAI/LFM2.5-VL-1.6B" in repositories
    assert "OpenGVLab/InternVL3_5-1B-HF" in repositories
    assert "OpenGVLab/InternVL3_5-2B-HF" in repositories
    assert "ibm-granite/granite-vision-3.3-2b" in repositories
    assert "ibm-granite/granite-vision-4.1-4b" in repositories


def test_modern_video_is_primary_input_and_thinking_is_explicit():
    assert "image" in modern_vlm.ModernVLM.INPUT_TYPES()["optional"]
    assert "image" in qwen2vl.Qwen2VLNode.INPUT_TYPES()["optional"]
    predictor = modern_vlm.ModernVLMPredictor.__new__(
        modern_vlm.ModernVLMPredictor
    )
    predictor.spec = modern_vlm.ModelSpec(
        "test/model", "Qwen 3.5", 1.0, video=True
    )
    captured = {}

    def capture(messages, enable_thinking=False, **kwargs):
        captured["messages"] = messages
        captured["enable_thinking"] = enable_thinking
        captured.update(kwargs)
        raise RuntimeError("captured before inference")

    predictor._inputs = capture
    frames = torch.zeros((4, 8, 8, 3), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="captured before inference"):
        predictor.generate(
            None,
            "What moves?",
            "",
            8,
            0.0,
            0.9,
            frames,
            2.0,
            True,
        )

    content = captured["messages"][-1]["content"]
    assert [part["type"] for part in content] == ["video", "text"]
    assert len(content[0]["video"]) == 4
    assert "2 FPS" in content[1]["text"]
    assert captured["enable_thinking"] is True
    assert captured["video_metadata"]["fps"] == 2.0
    assert captured["video_metadata"]["frames_indices"] == [0, 1, 2, 3]


def test_internvl_video_uses_an_even_vision_patch_grid():
    predictor = modern_vlm.ModernVLMPredictor.__new__(
        modern_vlm.ModernVLMPredictor
    )
    predictor.spec = modern_vlm.ModelSpec(
        "test/model", "InternVL 3.5", 1.0, video=True
    )
    captured = {}

    class ImageProcessor:
        size = {"height": 448, "width": 448}

    class Processor:
        image_processor = ImageProcessor()

        def apply_chat_template(self, _messages, **kwargs):
            captured.update(kwargs)
            return {"input_ids": torch.ones((1, 1), dtype=torch.long)}

    predictor.processor = Processor()
    predictor._inputs(
        [{"role": "user", "content": [{"type": "text", "text": "test"}]}],
        video_metadata={"fps": 2.0},
    )
    assert captured["processor_kwargs"]["size"] == {
        "height": 448,
        "width": 448,
    }


def test_qwen2_legacy_quantized_labels_use_maintained_backends():
    assert list(qwen2vl.QWEN2_VL_CHOICES) == [
        "Qwen2-VL-2B",
        "Qwen2-VL-7B",
    ]
    assert qwen2vl.LEGACY_QUANTIZED_ALIASES[
        "Qwen2-VL-7B-GPTQ-Int8"
    ] == ("Qwen2-VL-7B", "Balanced (8-bit)")
    assert qwen2vl.LEGACY_QUANTIZED_ALIASES[
        "Qwen2-VL-7B-AWQ"
    ] == ("Qwen2-VL-7B", "Maximum Savings (4-bit)")


def test_audioldm_keeps_legacy_outputs_and_adds_standard_audio(monkeypatch):
    class FakePredictor:
        def generate(self, *_args):
            return np.zeros((2, 16), dtype=np.float32), 16000

    node = audioldm2.AudioLDM2Node()
    monkeypatch.setattr(node, "get_or_create_model", lambda *_args: FakePredictor())
    result = node.generate_audio_final(
        "rain", "", 1, 3.5, 16000, 42, 2, "wav"
    )
    assert len(result) == 3
    assert result[1] == 16000
    assert result[2]["waveform"].shape == (2, 1, 16)


def test_node_functions_accept_every_declared_input_name():
    for node_class in package.NODE_CLASS_MAPPINGS.values():
        function = getattr(node_class, node_class.FUNCTION)
        signature = inspect.signature(function)
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            continue
        declared = {
            name
            for group in node_class.INPUT_TYPES().values()
            if isinstance(group, dict)
            for name in group
        }
        accepted = set(signature.parameters)
        assert declared <= accepted, (
            node_class.__name__,
            declared - accepted,
        )
