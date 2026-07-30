"""Contract tests for the GGUF text/LLM nodes in ``nodes/suggest.py``.

These nodes carry the pack's longest bug history (widget-index drift, unexpected
sampling kwargs, JSON that never parsed), so the assertions below pin the
behaviours those reports depended on rather than the models themselves.  No
llama.cpp wheel and no GGUF weights are required.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ComfyUI_VLM_nodes.nodes import suggest
from ComfyUI_VLM_nodes.nodes.runtime import LlamaHandle

MODEL_FILE = "some-model.gguf"


class FakeLlama:
    """Records the kwargs llama.cpp would have received."""

    def __init__(self, content: str = "generated text"):
        self.content = content
        self.calls: list[dict] = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.content}}]}


class FakeHandle:
    """Stands in for LlamaHandle so caching can be observed without weights."""

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
def resolved_model(monkeypatch):
    """Bypass folder_paths so no real GGUF has to exist on disk."""

    path = Path("/models/LLavacheckpoints") / MODEL_FILE
    monkeypatch.setattr(suggest, "resolve_model_path", lambda name: path)
    return path


@pytest.fixture
def fake_handles(monkeypatch):
    FakeHandle.instances = []
    monkeypatch.setattr(suggest, "LlamaHandle", FakeHandle)
    return FakeHandle


# --------------------------------------------------------------------------
# Widget ordering.  Comfy serializes widget values by position, so a reordered
# INPUT_TYPES silently rebinds every saved workflow (issue #156).
# --------------------------------------------------------------------------


def test_llm_sampler_widget_order_is_frozen():
    assert list(suggest.LLMSampler.INPUT_TYPES()["required"]) == [
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


def test_llm_prompt_generator_widget_order_is_frozen():
    assert list(suggest.LLMPromptGenerator.INPUT_TYPES()["required"]) == [
        "prompt",
        "model",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "repeat_penalty",
    ]


def test_llm_loader_widget_order_is_frozen():
    schema = suggest.LLMLoader.INPUT_TYPES()
    assert list(schema["required"]) == [
        "ckpt_name",
        "max_ctx",
        "gpu_layers",
        "n_threads",
    ]
    # chat_format must stay ahead of the shared runtime widgets.
    assert list(schema["optional"])[0] == "chat_format"


def test_structured_output_widget_order_is_frozen():
    assert list(suggest.StructuredOutput.INPUT_TYPES()["required"]) == [
        "prompt",
        "model",
        "temperature",
        "attribute_name",
        "attribute_type",
        "attribute_description",
        "categories",
    ]


def test_every_suggest_node_declares_a_callable_function_and_return_types():
    for name, node_class in suggest.NODE_CLASS_MAPPINGS.items():
        assert isinstance(node_class.RETURN_TYPES, tuple), name
        assert node_class.RETURN_TYPES, name
        assert callable(getattr(node_class, node_class.FUNCTION, None)), name
        assert node_class.CATEGORY.startswith("VLM Nodes"), name


def test_display_names_cover_every_registered_node():
    assert set(suggest.NODE_CLASS_MAPPINGS) == set(suggest.NODE_DISPLAY_NAME_MAPPINGS)


# --------------------------------------------------------------------------
# Sampling kwargs.  Issue #144 was an "unexpected keyword argument" crash, so
# the plumbing from node widget to create_chat_completion is asserted directly.
# --------------------------------------------------------------------------


def test_llm_sampler_forwards_every_sampling_argument():
    llama = FakeLlama("a description")
    result = suggest.LLMSampler().generate_text_advanced(
        system_msg="be terse",
        prompt="describe a cat",
        model=llama,
        max_tokens=64,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        repeat_penalty=1.3,
        seed=1234,
    )

    assert result == ("a description",)
    (call,) = llama.calls
    assert call["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "describe a cat"},
    ]
    assert call["max_tokens"] == 64
    assert call["temperature"] == 0.7
    assert call["top_p"] == 0.8
    assert call["top_k"] == 20
    assert call["frequency_penalty"] == 0.1
    assert call["presence_penalty"] == 0.2
    assert call["repeat_penalty"] == 1.3
    assert call["seed"] == 1234
    # No response_format unless a structured node asked for one.
    assert "response_format" not in call


def test_chat_unwraps_a_lazy_handle_before_generating(fake_handles, resolved_model):
    handle = FakeHandle(resolved_model)
    text = suggest._chat(handle, prompt="hi", system="sys")
    assert text == "generated text"


def test_llama_chat_content_rejects_an_empty_completion():
    llama = FakeLlama(content="   ")
    with pytest.raises(RuntimeError, match="empty response"):
        suggest.LLMSampler().generate_text_advanced(
            system_msg="s",
            prompt="p",
            model=llama,
            max_tokens=8,
            temperature=0.1,
            top_p=0.9,
            top_k=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repeat_penalty=1.0,
            seed=1,
        )


# --------------------------------------------------------------------------
# Lazy loading.  Building a loader node must not touch llama.cpp or the GGUF.
# --------------------------------------------------------------------------


def test_llm_loader_builds_a_lazy_handle_without_loading_weights(resolved_model):
    (handle,) = suggest.LLMLoader().load_llm_checkpoint(
        ckpt_name=MODEL_FILE,
        max_ctx=8192,
        gpu_layers=20,
        n_threads=6,
    )

    assert isinstance(handle, LlamaHandle)
    assert handle.model_path == resolved_model
    assert handle.n_ctx == 8192
    assert handle.n_gpu_layers == 20
    assert handle.n_threads == 6
    # Nothing was loaded: the llama.cpp object is still absent.
    assert handle._llm is None


def test_llm_loader_treats_blank_chat_format_as_the_embedded_template(resolved_model):
    (blank,) = suggest.LLMLoader().load_llm_checkpoint(
        MODEL_FILE, 2048, -1, 4, chat_format="   "
    )
    assert blank.chat_format is None

    (explicit,) = suggest.LLMLoader().load_llm_checkpoint(
        MODEL_FILE, 2048, -1, 4, chat_format="  chatml  "
    )
    assert explicit.chat_format == "chatml"


def test_llm_loader_forwards_advanced_runtime_options(resolved_model):
    (handle,) = suggest.LLMLoader().load_llm_checkpoint(
        MODEL_FILE,
        2048,
        -1,
        4,
        n_batch=256,
        n_ubatch=128,
        flash_attention="Disabled",
        use_mmap=False,
        split_mode="Single GPU",
        main_gpu=2,
        tensor_split="0.6,0.4",
    )

    assert handle.n_batch == 256
    assert handle.n_ubatch == 128
    assert handle.flash_attention == "Disabled"
    assert handle.use_mmap is False
    assert handle.split_mode == "Single GPU"
    assert handle.main_gpu == 2
    assert handle.tensor_split == [0.6, 0.4]


# --------------------------------------------------------------------------
# Structured output.
# --------------------------------------------------------------------------


def _capture_chat(monkeypatch, payload):
    """Replace _chat so the generated JSON Schema can be inspected."""

    recorded: dict = {}

    def fake_chat(model, **kwargs):
        recorded.update(kwargs)
        return payload

    monkeypatch.setattr(suggest, "_chat", fake_chat)
    return recorded


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("str", "string"),
        ("int", "integer"),
        ("float", "number"),
        ("bool", "boolean"),
    ],
)
def test_structured_output_maps_scalar_types_to_json_schema(
    monkeypatch, declared, expected
):
    recorded = _capture_chat(monkeypatch, json.dumps({"result": "value"}))
    suggest.StructuredOutput().keyword_extract(
        prompt="p",
        model=object(),
        temperature=0.1,
        attribute_name="result",
        attribute_type=declared,
        attribute_description="a description",
        categories="",
    )

    schema = recorded["response_format"]["schema"]
    assert schema["properties"]["result"]["type"] == expected
    assert schema["properties"]["result"]["description"] == "a description"
    assert schema["required"] == ["result"]
    assert schema["additionalProperties"] is False
    assert recorded["response_format"]["type"] == "json_object"


def test_structured_output_builds_an_enum_for_categories(monkeypatch):
    recorded = _capture_chat(monkeypatch, json.dumps({"mood": "calm"}))
    (value,) = suggest.StructuredOutput().keyword_extract(
        prompt="p",
        model=object(),
        temperature=0.1,
        attribute_name="  mood  ",
        attribute_type="Category",
        attribute_description="",
        categories=" calm , tense ,, bright ",
    )

    schema = recorded["response_format"]["schema"]
    assert schema["properties"]["mood"]["enum"] == ["calm", "tense", "bright"]
    assert schema["properties"]["mood"]["type"] == "string"
    assert value == "calm"


def test_structured_output_serializes_non_string_values(monkeypatch):
    _capture_chat(monkeypatch, json.dumps({"count": 7}))
    (value,) = suggest.StructuredOutput().keyword_extract(
        prompt="p",
        model=object(),
        temperature=0.1,
        attribute_name="count",
        attribute_type="int",
        attribute_description="",
        categories="",
    )
    assert value == "7"


def test_structured_output_rejects_an_empty_attribute_name():
    with pytest.raises(ValueError, match="attribute_name cannot be empty"):
        suggest.StructuredOutput().keyword_extract(
            prompt="p",
            model=object(),
            temperature=0.1,
            attribute_name="   ",
            attribute_type="str",
            attribute_description="",
            categories="",
        )


def test_structured_output_rejects_a_category_without_values():
    with pytest.raises(ValueError, match="at least one comma-separated value"):
        suggest.StructuredOutput().keyword_extract(
            prompt="p",
            model=object(),
            temperature=0.1,
            attribute_name="mood",
            attribute_type="Category",
            attribute_description="",
            categories="  ,  ",
        )


def test_structured_chat_reports_unparseable_json_with_a_bounded_excerpt(monkeypatch):
    _capture_chat(monkeypatch, "x" * 900)
    with pytest.raises(RuntimeError, match="did not return valid JSON") as error:
        suggest.KeywordExtraction().keyword_extract(
            prompt="p", model=object(), temperature=0.1
        )
    # The raw completion is truncated so a runaway response cannot flood the log.
    assert len(str(error.value)) < 600


def test_keyword_extraction_returns_the_raw_json_document(monkeypatch):
    payload = json.dumps(
        {
            "main_character": ["cat"],
            "artform": ["photo"],
            "photo_type": ["portrait"],
            "color_with_objects": ["black cat"],
            "digital_artform": [],
            "background": ["studio"],
            "lighting": ["soft"],
        }
    )
    _capture_chat(monkeypatch, payload)
    (raw,) = suggest.KeywordExtraction().keyword_extract(
        prompt="a cat", model=object(), temperature=0.1
    )
    assert json.loads(raw)["main_character"] == ["cat"]


def test_llava_prompt_generator_returns_only_the_prompt_field(monkeypatch):
    _capture_chat(monkeypatch, json.dumps({"prompt": "a moody portrait"}))
    (text,) = suggest.LLavaPromptGenerator().generate_prompts(
        prompt="p", model=object(), temperature=0.1
    )
    assert text == "a moody portrait"


def test_creative_art_prompt_generator_prefers_the_narrative(monkeypatch):
    _capture_chat(
        monkeypatch,
        json.dumps(
            {
                "techniques": {"preferred": ["ink"], "avoided": []},
                "theme": {"core_subject": "a harbour"},
                "style": {"desired": ["muted"], "undesired": []},
                "creative_descriptions": [{"description": "a quiet harbour at dawn"}],
            }
        ),
    )
    (text,) = suggest.CreativeArtPromptGenerator().create_creative_art_prompts(
        prompt="p", model=object(), temperature=0.1
    )
    assert text == "a quiet harbour at dawn"


def test_creative_art_prompt_generator_composes_a_fallback_without_narratives(
    monkeypatch,
):
    _capture_chat(
        monkeypatch,
        json.dumps(
            {
                "techniques": {"preferred": ["ink", "wash"], "avoided": []},
                "theme": {"core_subject": "a harbour"},
                "style": {"desired": ["muted", "grainy"], "undesired": []},
                "creative_descriptions": [],
            }
        ),
    )
    (text,) = suggest.CreativeArtPromptGenerator().create_creative_art_prompts(
        prompt="p", model=object(), temperature=0.1
    )
    assert text == (
        "a harbour. Techniques: ink, wash. Visual style: muted, grainy."
    )


def test_suggester_switches_instruction_on_the_randomize_toggle(monkeypatch):
    payload = json.dumps(
        {f"suggestion{index}": f"idea {index}" for index in range(1, 6)}
    )

    similar = _capture_chat(monkeypatch, payload)
    suggest.Suggester().generate_suggestions(
        prompt="p", model=object(), temperature=0.1, randomize=True
    )
    assert "close, useful variations" in similar["system"]

    different = _capture_chat(monkeypatch, payload)
    suggest.Suggester().generate_suggestions(
        prompt="p", model=object(), temperature=0.1, randomize=False
    )
    assert "deliberately different" in different["system"]


# --------------------------------------------------------------------------
# Handle caching.  Issue #137 was "model never unloads"; these pin the reuse
# and teardown rules of the optional-memory-free nodes.
# --------------------------------------------------------------------------


def test_cached_llm_reuses_one_handle_for_identical_settings(
    fake_handles, resolved_model
):
    node = suggest.LLMOptionalMemoryFreeSimple()
    first = node._model(MODEL_FILE, 2048, -1, 4)
    second = node._model(MODEL_FILE, 2048, -1, 4)

    assert first is second
    assert len(fake_handles.instances) == 1


def test_cached_llm_closes_the_previous_handle_when_settings_change(
    fake_handles, resolved_model
):
    node = suggest.LLMOptionalMemoryFreeSimple()
    first = node._model(MODEL_FILE, 2048, -1, 4)
    second = node._model(MODEL_FILE, 4096, -1, 4)

    assert first is not second
    assert first.closed is True
    assert second.closed is False
    assert len(fake_handles.instances) == 2


def test_cached_llm_rebuilds_when_an_advanced_runtime_option_changes(
    fake_handles, resolved_model
):
    node = suggest.LLMOptionalMemoryFreeSimple()
    node._model(MODEL_FILE, 2048, -1, 4, n_batch=512)
    node._model(MODEL_FILE, 2048, -1, 4, n_batch=256)
    assert len(fake_handles.instances) == 2


def test_cached_llm_unload_releases_the_handle(fake_handles, resolved_model):
    node = suggest.LLMOptionalMemoryFreeSimple()
    handle = node._model(MODEL_FILE, 2048, -1, 4)

    node._maybe_unload(False)
    assert handle.closed is False
    assert node._handle is handle

    node._maybe_unload(True)
    assert handle.closed is True
    assert node._handle is None
    assert node._key is None


def test_any_type_never_reports_a_type_mismatch():
    assert (suggest.ANY != "IMAGE") is False
    assert (suggest.ANY != "STRING") is False


# --------------------------------------------------------------------------
# ChatMusician.  Issue #149's workaround was for users to append "respond in
# ABC notation starting with X:1" themselves; the node now owns that.
# --------------------------------------------------------------------------


def _chat_musician_kwargs():
    return {
        "max_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.1,
        "seed": 42,
        "sample_rate": 44100,
    }


ABC_TUNE = "X:1\nT:Test\nM:4/4\nK:C\nCDEF|GABc|"


def test_chat_musician_asks_for_abc_notation_without_user_help(monkeypatch):
    recorded = _capture_chat(monkeypatch, ABC_TUNE)
    monkeypatch.setattr(suggest, "require_module", lambda *a, **k: _fake_symusic())

    suggest.ChatMusician().chat_musician(
        prompt="a waltz", model=object(), **_chat_musician_kwargs()
    )

    assert "ABC notation" in recorded["prompt"]
    assert "X:" in recorded["prompt"]
    assert "a waltz" in recorded["prompt"]
    assert "ABC notation" in recorded["system"]


def test_chat_musician_rejects_a_response_without_an_abc_header(monkeypatch):
    _capture_chat(monkeypatch, "Sure! Here is a lovely tune for you.")
    with pytest.raises(RuntimeError, match="did not contain ABC notation"):
        suggest.ChatMusician().chat_musician(
            prompt="p", model=object(), **_chat_musician_kwargs()
        )


def test_chat_musician_strips_preamble_before_the_abc_header(monkeypatch):
    _capture_chat(monkeypatch, "Here you go:\n\n" + ABC_TUNE)
    monkeypatch.setattr(suggest, "require_module", lambda *a, **k: _fake_symusic())

    abc, _legacy, _rate, _audio = suggest.ChatMusician().chat_musician(
        prompt="p", model=object(), **_chat_musician_kwargs()
    )
    assert abc.startswith("X:1")
    assert "Here you go" not in abc


def test_chat_musician_returns_comfy_audio_and_legacy_layouts(monkeypatch):
    _capture_chat(monkeypatch, ABC_TUNE)
    monkeypatch.setattr(suggest, "require_module", lambda *a, **k: _fake_symusic())

    _abc, legacy, rate, audio = suggest.ChatMusician().chat_musician(
        prompt="p", model=object(), **_chat_musician_kwargs()
    )

    # Comfy AUDIO is [batch, channels, samples].
    assert audio["waveform"].shape == (1, 2, 100)
    assert audio["sample_rate"] == 44100
    assert rate == 44100
    # soundfile-compatible legacy output is [samples, channels].
    assert legacy.shape == (100, 2)


def _fake_symusic():
    """A symusic stand-in so the AUDIO contract is testable without the wheel."""

    import numpy as np

    class Synthesizer:
        def __init__(self, sample_rate):
            self.sample_rate = sample_rate

        def render(self, score, stereo=True):
            return np.zeros((2, 100), dtype=np.float32)

    class Score:
        @staticmethod
        def from_abc(abc):
            return SimpleNamespace(abc=abc)

    return SimpleNamespace(Score=Score, Synthesizer=Synthesizer)


def _memory_free_kwargs(**overrides):
    kwargs = {
        "ckpt_name": MODEL_FILE,
        "max_ctx": 2048,
        "gpu_layers": -1,
        "n_threads": 4,
        "prompt": "write a haiku",
        "temperature": 0.2,
        "unload": False,
    }
    kwargs.update(overrides)
    return kwargs


def test_memory_free_simple_generates_through_the_cached_handle(
    fake_handles, resolved_model
):
    node = suggest.LLMOptionalMemoryFreeSimple()
    (text,) = node.generate_text(**_memory_free_kwargs())

    assert text == "generated text"
    (handle,) = fake_handles.instances
    assert handle.closed is False
    assert node._handle is handle
    (call,) = handle.llama.calls
    assert call["temperature"] == 0.2
    assert call["messages"][1]["content"] == "write a haiku"


def test_memory_free_simple_unloads_after_generating_when_asked(
    fake_handles, resolved_model
):
    node = suggest.LLMOptionalMemoryFreeSimple()
    node.generate_text(**_memory_free_kwargs(unload=True))

    (handle,) = fake_handles.instances
    assert handle.closed is True
    assert node._handle is None


def test_memory_free_simple_unloads_even_when_generation_fails(
    fake_handles, resolved_model, monkeypatch
):
    """Issue #137: a failed generation must not strand the model in VRAM."""

    def explode(*args, **kwargs):
        raise RuntimeError("llama.cpp exploded")

    monkeypatch.setattr(suggest, "_chat", explode)
    node = suggest.LLMOptionalMemoryFreeSimple()

    with pytest.raises(RuntimeError, match="exploded"):
        node.generate_text(**_memory_free_kwargs(unload=True))

    (handle,) = fake_handles.instances
    assert handle.closed is True
    assert node._handle is None


def test_memory_free_advanced_forwards_every_sampling_argument(
    fake_handles, resolved_model
):
    node = suggest.LLMOptionalMemoryFreeAdvanced()
    signature = suggest.LLMOptionalMemoryFreeAdvanced.INPUT_TYPES()["required"]
    assert "system_msg" in signature

    (text,) = node.generate_text_advanced(
        ckpt_name=MODEL_FILE,
        max_ctx=2048,
        gpu_layers=-1,
        n_threads=4,
        system_msg="be brief",
        prompt="a haiku",
        max_tokens=48,
        temperature=0.5,
        top_p=0.8,
        top_k=15,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        repeat_penalty=1.2,
        seed=11,
        unload=False,
    )

    assert text == "generated text"
    (handle,) = fake_handles.instances
    (call,) = handle.llama.calls
    assert call["messages"][0] == {"role": "system", "content": "be brief"}
    assert call["max_tokens"] == 48
    assert call["temperature"] == 0.5
    assert call["top_p"] == 0.8
    assert call["top_k"] == 15
    assert call["seed"] == 11


def test_cached_llm_key_is_insensitive_to_runtime_option_ordering(
    fake_handles, resolved_model
):
    node = suggest.LLMOptionalMemoryFreeSimple()
    node._model(MODEL_FILE, 2048, -1, 4, n_batch=256, main_gpu=1)
    node._model(MODEL_FILE, 2048, -1, 4, main_gpu=1, n_batch=256)
    # Keyword order must not invalidate the cache and reload the GGUF.
    assert len(fake_handles.instances) == 1


def test_schema_helper_emits_a_json_schema_for_a_pydantic_model():
    schema = suggest._schema(suggest.PromptGen)
    assert schema["properties"]["prompt"]["type"] == "string"
    assert schema["required"] == ["prompt"]


def test_response_content_extraction_matches_the_runtime_helper():
    response = {"choices": [{"message": {"content": " text "}}]}
    assert suggest._response_content(response) == "text"


def test_stub_handle_matches_the_real_handle_api():
    """Guard the stub: LlamaHandle must keep the API these tests rely on."""

    assert callable(getattr(LlamaHandle, "ensure_loaded", None))
    assert callable(getattr(LlamaHandle, "close", None))
