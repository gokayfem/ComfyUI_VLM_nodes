import json
import threading
import time

import pytest
import torch
from ComfyUI_VLM_nodes.nodes.acceleration import (
    VLMImagePixelBudget,
    VLMPerformanceProfile,
    optimize_image_pixels,
)
from ComfyUI_VLM_nodes.nodes.runtime import (
    CachedModelNode,
    tensor_batch_to_pil,
    tensor_to_pil,
)


def test_batch_conversion_matches_single_frame_contract():
    images = torch.tensor(
        [
            [
                [[float("nan"), 0.5, 2.0], [-1.0, 0.25, 1.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            ],
            [
                [[255.0, 128.0, 0.0], [0.0, 64.0, 255.0]],
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ],
        ]
    )
    batch = tensor_batch_to_pil(images)
    assert len(batch) == 2
    for index, converted in enumerate(batch):
        assert converted.mode == "RGB"
        assert converted.size == (2, 2)
        assert converted.tobytes() == tensor_to_pil(images, index).tobytes()

    with pytest.raises(IndexError, match="only has batch index 0"):
        tensor_to_pil(images[0], 1)


def test_pixel_budget_preserves_aspect_and_patch_multiple():
    images = torch.rand((3, 1080, 1920, 3), dtype=torch.float32)
    output, report = optimize_image_pixels(
        images,
        max_megapixels=0.5,
        max_edge=1024,
        multiple=14,
        resize_quality="Fast (area)",
    )
    assert output.ndim == 4
    assert output.shape[0] == 3
    assert output.shape[1] % 14 == 0
    assert output.shape[2] % 14 == 0
    assert output.shape[1] * output.shape[2] <= 500_000
    assert output.shape[2] <= 1024
    assert report["visual_work_reduction"] > 4
    assert output.shape[2] / output.shape[1] == pytest.approx(16 / 9, rel=0.03)


def test_pixel_budget_never_upscales():
    image = torch.rand((240, 320, 3), dtype=torch.float32)
    output, report = optimize_image_pixels(
        image,
        max_megapixels=2.0,
        max_edge=2048,
        multiple=1,
        resize_quality="Quality (bicubic)",
    )
    assert output is image
    assert report["resized"] is False


def test_performance_nodes_return_standard_comfy_values():
    profile = VLMPerformanceProfile().profile("Live / robotics")
    assert profile[:5] == (24, 0.5, 896, 8, False)
    assert json.loads(profile[5])["profile"] == "Live / robotics"

    optimized = VLMImagePixelBudget().optimize(
        torch.rand((1, 1000, 1600, 3)),
        0.5,
        1024,
        "14",
        "Fast (area)",
    )
    assert optimized[1] % 14 == 0
    assert optimized[2] % 14 == 0


def test_cached_model_node_prevents_duplicate_concurrent_loads():
    node = CachedModelNode()
    factory_calls = []
    handles = []

    def factory():
        factory_calls.append(1)
        time.sleep(0.02)
        return object()

    def load():
        handles.append(node.get_or_create_model("same-model", factory))

    threads = [threading.Thread(target=load) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(factory_calls) == 1
    assert len({id(handle) for handle in handles}) == 1
