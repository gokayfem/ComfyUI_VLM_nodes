import base64
import inspect
import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import ComfyUI_VLM_nodes as package
from ComfyUI_VLM_nodes.nodes import audioldm2, florence2, modern_vlm, paligemma
from ComfyUI_VLM_nodes.nodes.runtime import (
    image_data_uri,
    pil_mask_to_tensor,
    pil_to_tensor,
    tensor_batch_to_pil,
)


def test_every_module_imports_and_expected_nodes_exist():
    assert package.IMPORT_ERRORS == {}
    expected = {
        "ModernVLM",
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
    assert "Qwen/Qwen3.5-4B" in repositories
    assert "Qwen/Qwen3-VL-8B-Instruct" in repositories
    assert "google/gemma-3-4b-it" in repositories
    assert "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" in repositories


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
