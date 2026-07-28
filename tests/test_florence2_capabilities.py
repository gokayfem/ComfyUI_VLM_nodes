from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from ComfyUI_VLM_nodes.nodes import florence2
from PIL import Image

EXPECTED_TASKS = {
    "Caption": ("<CAPTION>", "none"),
    "Detailed caption": ("<DETAILED_CAPTION>", "none"),
    "More detailed caption": ("<MORE_DETAILED_CAPTION>", "none"),
    "OCR": ("<OCR>", "none"),
    "OCR with regions": ("<OCR_WITH_REGION>", "none"),
    "Object detection": ("<OD>", "none"),
    "Dense region caption": ("<DENSE_REGION_CAPTION>", "none"),
    "Caption to phrase grounding": ("<CAPTION_TO_PHRASE_GROUNDING>", "text"),
    "Referring expression segmentation": (
        "<REFERRING_EXPRESSION_SEGMENTATION>",
        "text",
    ),
    "Region to segmentation": ("<REGION_TO_SEGMENTATION>", "region"),
    "Open vocabulary detection": ("<OPEN_VOCABULARY_DETECTION>", "text"),
    "Region to category": ("<REGION_TO_CATEGORY>", "region"),
    "Region to description": ("<REGION_TO_DESCRIPTION>", "region"),
    "Region to OCR": ("<REGION_TO_OCR>", "region"),
    "Region proposals": ("<REGION_PROPOSAL>", "none"),
}


def test_registry_covers_all_official_transformers_tasks():
    assert len(florence2.TASKS) == 15
    assert {
        name: (spec.token, spec.input_kind) for name, spec in florence2.TASKS.items()
    } == EXPECTED_TASKS
    assert {spec.output_kind for spec in florence2.TASKS.values()} == {
        "text",
        "boxes",
        "quad_boxes",
        "polygons",
        "mixed",
    }


def test_node_contract_preserves_outputs_and_adds_core_region_input():
    schema = florence2.Florence2.INPUT_TYPES()
    assert florence2.NODE_CLASS_MAPPINGS["Florence2"] is florence2.Florence2
    assert florence2.Florence2.RETURN_TYPES[:4] == (
        "STRING",
        "STRING",
        "MASK",
        "IMAGE",
    )
    assert florence2.Florence2.RETURN_NAMES[:4] == (
        "text",
        "structured_json",
        "mask",
        "visualization",
    )
    assert schema["optional"]["region"][0] == "BOUNDING_BOX"
    assert "forceInput" not in repr(schema["optional"]["region"])


def test_region_encoding_uses_core_xywh_and_florence_location_bins():
    region = {"x": 10, "y": 20, "width": 40, "height": 100}
    assert florence2._encode_region(region, (100, 200)) == (
        "<loc_100><loc_100><loc_500><loc_600>"
    )

    clamped = {"x": -10, "y": -20, "width": 200, "height": 300}
    assert florence2._encode_region(clamped, (100, 200)) == (
        "<loc_0><loc_0><loc_999><loc_999>"
    )

    with pytest.raises(ValueError, match="greater than zero"):
        florence2._encode_region(
            {"x": 0, "y": 0, "width": 0, "height": 10},
            (100, 100),
        )
    with pytest.raises(ValueError, match="does not overlap"):
        florence2._encode_region(
            {"x": 200, "y": 200, "width": 10, "height": 10},
            (100, 100),
        )


def test_task_inputs_are_validated_before_inference():
    image_size = (100, 100)
    region = {"x": 10, "y": 10, "width": 20, "height": 20}

    assert florence2._task_extra_input("Caption", "", None, image_size) == ""
    with pytest.raises(ValueError, match="does not accept text"):
        florence2._task_extra_input("Caption", "unexpected", None, image_size)
    with pytest.raises(ValueError, match="requires text"):
        florence2._task_extra_input("Open vocabulary detection", "", None, image_size)
    assert (
        florence2._task_extra_input(
            "Open vocabulary detection", "red car", None, image_size
        )
        == "red car"
    )
    with pytest.raises(ValueError, match="requires a connected BOUNDING_BOX"):
        florence2._task_extra_input("Region to OCR", "", None, image_size)
    assert (
        florence2._task_extra_input("Region to OCR", "", region, image_size)
        == "<loc_100><loc_100><loc_300><loc_300>"
    )
    with pytest.raises(ValueError, match="does not accept text"):
        florence2._task_extra_input("Region to OCR", "also text", region, image_size)


def test_region_selection_accepts_core_and_batched_detector_shapes():
    first = {"x": 1, "y": 2, "width": 3, "height": 4}
    second = {"x": 5, "y": 6, "width": 7, "height": 8}
    assert florence2._select_region(first, 0, 2) is first
    assert florence2._select_region([first, second], 1, 2) is second
    assert florence2._select_region([[first], [second]], 0, 2) is first

    with pytest.raises(ValueError, match="exactly one"):
        florence2._select_region([[first, second]], 0, 1)


def test_visualization_is_deterministic_and_masks_every_spatial_shape():
    image = Image.new("RGB", (48, 36), "black")
    parsed = {
        "<OPEN_VOCABULARY_DETECTION>": {
            "bboxes": [[1, 1, 10, 10]],
            "bboxes_labels": ["box"],
            "quad_boxes": [[14, 1, 22, 1, 22, 10, 14, 10]],
            "labels": ["ocr"],
            "polygons": [[[26, 1, 40, 1, 40, 12, 26, 12]]],
            "polygons_labels": ["polygon"],
        }
    }

    mask_a, visual_a = florence2._visualize(image, parsed)
    mask_b, visual_b = florence2._visualize(image, parsed)
    mask = np.asarray(mask_a)

    assert mask[5, 5] == 255
    assert mask[5, 18] == 255
    assert mask[5, 30] == 255
    assert mask_a.tobytes() == mask_b.tobytes()
    assert visual_a.tobytes() == visual_b.tobytes()


def test_predictor_generation_is_deterministic_without_downloads():
    calls = {}

    class FakeProcessor:
        def __call__(self, text, images, return_tensors):
            calls["prompt"] = text
            assert images.size == (8, 8)
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor([[1]], dtype=torch.long),
                "pixel_values": torch.zeros((1, 3, 8, 8)),
            }

        def batch_decode(self, generated, skip_special_tokens):
            assert skip_special_tokens is False
            return ["<s>answer</s>"]

        def post_process_generation(self, raw, task, image_size):
            return {task: "answer"}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def generate(self, **kwargs):
            calls["generation"] = kwargs
            return torch.tensor([[2]], dtype=torch.long)

    class FakeHandle:
        def __init__(self):
            self.model = FakeModel()

        def ensure_loaded(self):
            return self.model

    predictor = object.__new__(florence2.FlorencePredictor)
    predictor.dtype = torch.float32
    predictor.processor = FakeProcessor()
    predictor.handle = FakeHandle()

    raw, parsed = predictor.run(
        Image.new("RGB", (8, 8)),
        "<CAPTION>",
        "",
        32,
        3,
    )
    assert raw == "<s>answer</s>"
    assert parsed == {"<CAPTION>": "answer"}
    assert calls["prompt"] == "<CAPTION>"
    assert calls["generation"]["do_sample"] is False
    assert calls["generation"]["num_beams"] == 3
    assert calls["generation"]["early_stopping"] is True


def test_node_cleans_text_preserves_structured_data_and_unloads_target():
    parsed = {
        "<OD>": {
            "bboxes": [[2, 2, 12, 12]],
            "labels": ["person"],
            "quad_boxes": [[14, 2, 22, 2, 22, 12, 14, 12]],
            "polygons": [[[24, 2, 30, 2, 30, 12, 24, 12]]],
        }
    }
    calls = []

    class FakePredictor:
        def run(
            self,
            image,
            task_token,
            extra_input,
            max_new_tokens,
            beams,
        ):
            calls.append((image.size, task_token, extra_input, max_new_tokens, beams))
            return "<s>person<loc_1><loc_2></s><pad>", parsed

    node = florence2.Florence2()
    node.get_or_create_model = lambda key, factory: FakePredictor()
    unloads = []
    node.maybe_clear_model = unloads.append

    output = node.run(
        torch.zeros((1, 32, 32, 3)),
        "Object detection",
        "",
        "Florence-2 base FT (fast)",
        64,
        1,
        unload_after=True,
    )

    assert len(output) == 4
    assert output[0] == "person<loc_1><loc_2>"
    assert json.loads(output[1]) == [parsed]
    assert output[2].shape == (1, 32, 32)
    assert output[2].max().item() == 1.0
    assert output[3].shape == (1, 32, 32, 3)
    assert calls == [((32, 32), "<OD>", "", 64, 1)]
    assert unloads == [True]
