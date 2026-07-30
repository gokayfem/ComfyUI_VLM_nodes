"""Contract tests for the llama.cpp multimodal nodes in ``nodes/llavaloader.py``.

Covers batch handling, the vision message envelope, projector wiring, and the
cached-handle lifecycle.  No llama.cpp wheel, mmproj, or GGUF weights required.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
import torch
from ComfyUI_VLM_nodes.nodes import llavaloader
from ComfyUI_VLM_nodes.nodes.runtime import LlamaHandle, LlavaClipConfig

MODEL_FILE = "llava.gguf"
CLIP_FILE = "mmproj.gguf"


class FakeLlama:
    def __init__(self, contents: list[str] | None = None):
        self.contents = contents or ["a description"]
        self.calls: list[dict] = []

    def create_chat_completion(self, **kwargs):
        index = min(len(self.calls), len(self.contents) - 1)
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.contents[index]}}]}


class FakeHandle:
    instances: list[FakeHandle] = []

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.closed = False
        self.llama = FakeLlama()
        FakeHandle.instances.append(self)

    def ensure_loaded(self):
        return self.llama

    def close(self):
        self.closed = True


@pytest.fixture
def resolved_paths(monkeypatch):
    root = Path("/models/LLavacheckpoints")
    monkeypatch.setattr(llavaloader, "resolve_model_path", lambda name: root / name)
    return root


@pytest.fixture
def fake_handles(monkeypatch):
    FakeHandle.instances = []
    monkeypatch.setattr(llavaloader, "LlamaHandle", FakeHandle)
    return FakeHandle


def image_batch(count: int = 1, size: int = 4) -> torch.Tensor:
    """A ComfyUI BHWC float image batch."""

    return torch.rand(count, size, size, 3)


# --------------------------------------------------------------------------
# Widget ordering (see issue #156).
# --------------------------------------------------------------------------


def test_llava_sampler_simple_widget_order_is_frozen():
    assert list(llavaloader.LLavaSamplerSimple.INPUT_TYPES()["required"]) == [
        "image",
        "prompt",
        "model",
        "temperature",
    ]


def test_llava_sampler_advanced_widget_order_is_frozen():
    assert list(llavaloader.LLavaSamplerAdvanced.INPUT_TYPES()["required"]) == [
        "image",
        "system_msg",
        "prompt",
        "model",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "repeat_penalty",
        "seed",
    ]


def test_llava_loader_widget_order_is_frozen():
    schema = llavaloader.LLavaLoader.INPUT_TYPES()
    assert list(schema["required"]) == [
        "ckpt_name",
        "max_ctx",
        "gpu_layers",
        "n_threads",
        "clip",
    ]


def test_optional_memory_free_simple_widget_order_is_frozen():
    schema = llavaloader.LLavaOptionalMemoryFreeSimple.INPUT_TYPES()
    assert list(schema["required"]) == [
        "ckpt_name",
        "clip_name",
        "max_ctx",
        "gpu_layers",
        "n_threads",
        "image",
        "prompt",
        "temperature",
        "unload",
    ]
    assert list(schema["optional"])[0] == "handler"


def test_every_llava_node_declares_a_callable_function_and_return_types():
    for name, node_class in llavaloader.NODE_CLASS_MAPPINGS.items():
        assert isinstance(node_class.RETURN_TYPES, tuple), name
        assert node_class.RETURN_TYPES, name
        assert callable(getattr(node_class, node_class.FUNCTION, None)), name
        assert node_class.CATEGORY.startswith("VLM Nodes"), name


def test_display_names_cover_every_registered_node():
    assert set(llavaloader.NODE_CLASS_MAPPINGS) == set(
        llavaloader.NODE_DISPLAY_NAME_MAPPINGS
    )


# --------------------------------------------------------------------------
# Vision message envelope.
# --------------------------------------------------------------------------


def test_vision_messages_place_the_image_before_the_text():
    messages = llavaloader._vision_messages("sys", "what is this?", "data:image/png;b")

    assert messages[0] == {"role": "system", "content": "sys"}
    content = messages[1]["content"]
    assert messages[1]["role"] == "user"
    # llama.cpp vision handlers require the image part first.
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == "data:image/png;b"
    assert content[1] == {"type": "text", "text": "what is this?"}


def test_run_batch_sends_a_png_data_uri_per_image():
    llama = FakeLlama()
    llavaloader._run_batch(
        image_batch(1), llama, system_msg="sys", prompt="p", temperature=0.1
    )

    (call,) = llama.calls
    url = call["messages"][1]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    # The payload must be real decodable PNG bytes.
    decoded = base64.b64decode(url.split(",", 1)[1])
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")


def test_run_batch_calls_the_model_once_per_batch_item():
    llama = FakeLlama(["first", "second", "third"])
    text = llavaloader._run_batch(
        image_batch(3), llama, system_msg="sys", prompt="p", temperature=0.1
    )

    assert len(llama.calls) == 3
    # Every batch item must survive into the response.
    assert "first" in text
    assert "second" in text
    assert "third" in text
    assert "--- Image 1 ---" in text
    assert "--- Image 3 ---" in text


def test_run_batch_returns_bare_text_for_a_single_image():
    llama = FakeLlama(["only one"])
    text = llavaloader._run_batch(
        image_batch(1), llama, system_msg="sys", prompt="p", temperature=0.1
    )
    assert text == "only one"


def test_run_batch_forwards_generation_kwargs_unchanged():
    llama = FakeLlama()
    llavaloader._run_batch(
        image_batch(1),
        llama,
        system_msg="sys",
        prompt="p",
        max_tokens=32,
        temperature=0.3,
        top_p=0.7,
        top_k=10,
        seed=99,
    )

    (call,) = llama.calls
    assert call["max_tokens"] == 32
    assert call["temperature"] == 0.3
    assert call["top_p"] == 0.7
    assert call["top_k"] == 10
    assert call["seed"] == 99


def test_sampler_simple_returns_a_single_string_output():
    llama = FakeLlama(["a cat on a mat"])
    result = llavaloader.LLavaSamplerSimple().generate_text(
        image=image_batch(1), prompt="describe", model=llama, temperature=0.1
    )
    assert result == ("a cat on a mat",)


def test_sampler_advanced_uses_the_supplied_system_message():
    llama = FakeLlama()
    llavaloader.LLavaSamplerAdvanced().generate_text_advanced(
        image=image_batch(1),
        system_msg="answer in French",
        prompt="describe",
        model=llama,
        max_tokens=16,
        temperature=0.1,
        top_p=0.9,
        top_k=5,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repeat_penalty=1.0,
        seed=7,
    )

    (call,) = llama.calls
    assert call["messages"][0] == {"role": "system", "content": "answer in French"}


# --------------------------------------------------------------------------
# Projector / clip wiring.
# --------------------------------------------------------------------------


def test_clip_factory_uses_the_config_create_hook():
    config = LlavaClipConfig(Path("/models/mmproj.gguf"), "LLaVA 1.6")
    assert llavaloader._clip_factory(config) == config.create


def test_clip_factory_accepts_a_precreated_handler():
    sentinel = object()
    factory = llavaloader._clip_factory(sentinel)
    # Workflows saved before handler selection passed the handler itself.
    assert factory() is sentinel


def test_make_handle_derives_the_projector_from_the_clip_config(resolved_paths):
    config = LlavaClipConfig(resolved_paths / CLIP_FILE, "LLaVA 1.5")
    handle = llavaloader._make_handle(MODEL_FILE, 4096, -1, 4, config)

    assert isinstance(handle, LlamaHandle)
    assert handle.projector_path == resolved_paths / CLIP_FILE
    assert handle.chat_handler_factory == config.create
    assert handle.n_ctx == 4096
    # Still lazy: no llama.cpp object was constructed.
    assert handle._llm is None


def test_make_handle_keeps_an_explicit_projector_override(resolved_paths):
    config = LlavaClipConfig(resolved_paths / CLIP_FILE, "LLaVA 1.5")
    override = Path("/models/other-mmproj.gguf")
    handle = llavaloader._make_handle(
        MODEL_FILE,
        4096,
        -1,
        4,
        config,
        runtime_options={"projector_path": override},
    )
    assert handle.projector_path == override


def test_llava_loader_does_not_load_weights(resolved_paths):
    config = LlavaClipConfig(resolved_paths / CLIP_FILE, "LLaVA 1.5")
    (handle,) = llavaloader.LLavaLoader().load_llava_checkpoint(
        ckpt_name=MODEL_FILE,
        max_ctx=2048,
        gpu_layers=10,
        n_threads=8,
        clip=config,
    )

    assert isinstance(handle, LlamaHandle)
    assert handle._llm is None
    assert handle.n_gpu_layers == 10
    assert handle.n_threads == 8


def test_clip_loader_returns_a_frozen_config_with_the_chosen_handler(resolved_paths):
    (config,) = llavaloader.LlavaClipLoader().load_clip_checkpoint(
        CLIP_FILE, handler="MiniCPM-V 2.6"
    )

    assert isinstance(config, LlavaClipConfig)
    assert config.model_path == resolved_paths / CLIP_FILE
    assert config.handler == "MiniCPM-V 2.6"


def test_clip_config_rejects_an_unknown_handler(resolved_paths):
    config = LlavaClipConfig(resolved_paths / CLIP_FILE, "Not A Handler")
    with pytest.raises((ValueError, RuntimeError)) as error:
        config.create()
    # Either an unknown-handler rejection or a missing-wheel report is correct;
    # a silent fallback to the wrong prompt format is not.
    assert "handler" in str(error.value).lower() or "llama" in str(error.value).lower()


def test_clip_loader_defaults_to_the_embedded_gguf_chat_template():
    handler = llavaloader.LlavaClipLoader.INPUT_TYPES()["optional"]["handler"]
    choices, options = handler[0], handler[1]
    assert options["default"] == "Auto (GGUF chat template)"
    assert options["default"] in choices
    assert "LLaVA 1.5" in choices


# --------------------------------------------------------------------------
# Cached-handle lifecycle (issue #137: "model never unloads").
# --------------------------------------------------------------------------


def test_cached_llava_reuses_one_handle_for_identical_settings(
    fake_handles, resolved_paths
):
    node = llavaloader.LLavaOptionalMemoryFreeSimple()
    first = node._model(MODEL_FILE, CLIP_FILE, 4096, -1, 4)
    second = node._model(MODEL_FILE, CLIP_FILE, 4096, -1, 4)

    assert first is second
    assert len(fake_handles.instances) == 1


def test_cached_llava_rebuilds_when_the_projector_changes(fake_handles, resolved_paths):
    node = llavaloader.LLavaOptionalMemoryFreeSimple()
    first = node._model(MODEL_FILE, CLIP_FILE, 4096, -1, 4)
    second = node._model(MODEL_FILE, "other-mmproj.gguf", 4096, -1, 4)

    assert first is not second
    assert first.closed is True
    assert len(fake_handles.instances) == 2


def test_cached_llava_rebuilds_when_the_handler_changes(fake_handles, resolved_paths):
    node = llavaloader.LLavaOptionalMemoryFreeSimple()
    node._model(MODEL_FILE, CLIP_FILE, 4096, -1, 4, handler="LLaVA 1.5")
    node._model(MODEL_FILE, CLIP_FILE, 4096, -1, 4, handler="LLaVA 1.6")
    assert len(fake_handles.instances) == 2


def test_cached_llava_unload_releases_the_handle(fake_handles, resolved_paths):
    node = llavaloader.LLavaOptionalMemoryFreeSimple()
    handle = node._model(MODEL_FILE, CLIP_FILE, 4096, -1, 4)

    node._maybe_unload(False)
    assert handle.closed is False

    node._maybe_unload(True)
    assert handle.closed is True
    assert node._handle is None
    assert node._key is None


def test_cached_llava_unload_is_safe_before_any_load(fake_handles, resolved_paths):
    node = llavaloader.LLavaOptionalMemoryFreeSimple()
    # Must not raise when nothing was ever loaded.
    node._maybe_unload(True)
    assert node._handle is None


def _memory_free_kwargs(**overrides):
    kwargs = {
        "ckpt_name": MODEL_FILE,
        "clip_name": CLIP_FILE,
        "max_ctx": 4096,
        "gpu_layers": -1,
        "n_threads": 4,
        "image": image_batch(1),
        "prompt": "describe this",
        "temperature": 0.1,
        "unload": False,
    }
    kwargs.update(overrides)
    return kwargs


def test_memory_free_simple_generates_through_the_cached_handle(
    fake_handles, resolved_paths
):
    node = llavaloader.LLavaOptionalMemoryFreeSimple()
    (text,) = node.generate_text(**_memory_free_kwargs())

    assert text == "a description"
    (handle,) = fake_handles.instances
    assert handle.closed is False
    (call,) = handle.llama.calls
    assert call["temperature"] == 0.1
    assert call["messages"][1]["content"][1]["text"] == "describe this"


def test_memory_free_simple_processes_every_image_in_the_batch(
    fake_handles, resolved_paths
):
    node = llavaloader.LLavaOptionalMemoryFreeSimple()
    node.generate_text(**_memory_free_kwargs(image=image_batch(2)))

    (handle,) = fake_handles.instances
    assert len(handle.llama.calls) == 2


def test_memory_free_simple_unloads_after_generating_when_asked(
    fake_handles, resolved_paths
):
    node = llavaloader.LLavaOptionalMemoryFreeSimple()
    node.generate_text(**_memory_free_kwargs(unload=True))

    (handle,) = fake_handles.instances
    assert handle.closed is True
    assert node._handle is None


def test_memory_free_simple_unloads_even_when_generation_fails(
    fake_handles, resolved_paths, monkeypatch
):
    """Issue #137: a failed generation must not strand the model in VRAM."""

    def explode(*args, **kwargs):
        raise RuntimeError("llama.cpp exploded")

    monkeypatch.setattr(llavaloader, "_run_batch", explode)
    node = llavaloader.LLavaOptionalMemoryFreeSimple()

    with pytest.raises(RuntimeError, match="exploded"):
        node.generate_text(**_memory_free_kwargs(unload=True))

    (handle,) = fake_handles.instances
    assert handle.closed is True
    assert node._handle is None


def test_memory_free_advanced_forwards_the_system_message_and_sampling(
    fake_handles, resolved_paths
):
    node = llavaloader.LLavaOptionalMemoryFreeAdvanced()
    (text,) = node.generate_text_advanced(
        ckpt_name=MODEL_FILE,
        clip_name=CLIP_FILE,
        max_ctx=4096,
        gpu_layers=-1,
        n_threads=4,
        image=image_batch(1),
        system_msg="answer in German",
        prompt="describe",
        max_tokens=64,
        temperature=0.4,
        top_p=0.85,
        top_k=25,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repeat_penalty=1.05,
        seed=5,
        unload=False,
    )

    assert text == "a description"
    (handle,) = fake_handles.instances
    (call,) = handle.llama.calls
    assert call["messages"][0] == {"role": "system", "content": "answer in German"}
    assert call["max_tokens"] == 64
    assert call["temperature"] == 0.4
    assert call["top_p"] == 0.85
    assert call["top_k"] == 25
    assert call["seed"] == 5


def test_cached_llava_key_is_insensitive_to_runtime_option_ordering(
    fake_handles, resolved_paths
):
    node = llavaloader.LLavaOptionalMemoryFreeSimple()
    node._model(MODEL_FILE, CLIP_FILE, 4096, -1, 4, n_batch=256, main_gpu=1)
    node._model(MODEL_FILE, CLIP_FILE, 4096, -1, 4, main_gpu=1, n_batch=256)
    assert len(fake_handles.instances) == 1
