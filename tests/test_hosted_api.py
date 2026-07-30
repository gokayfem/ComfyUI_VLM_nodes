from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ComfyUI_VLM_nodes.nodes import hosted_api


class FakeHttpClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeResponses:
    def __init__(self, calls, failure=None, response_text="secure response"):
        self.calls = calls
        self.failure = failure
        self.response_text = response_text

    def create(self, **kwargs):
        self.calls.append(("responses", kwargs))
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(output_text=self.response_text)


class FakeChatCompletions:
    def __init__(self, calls, failure=None, response_text="secure response"):
        self.calls = calls
        self.failure = failure
        self.response_text = response_text

    def create(self, **kwargs):
        self.calls.append(("chat", kwargs))
        if self.failure is not None:
            raise self.failure
        message = SimpleNamespace(content=self.response_text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def fake_openai_module(calls, failure=None, response_text="secure response"):
    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))
            self.responses = FakeResponses(
                calls,
                failure=failure,
                response_text=response_text,
            )
            self.chat = SimpleNamespace(
                completions=FakeChatCompletions(
                    calls,
                    failure=failure,
                    response_text=response_text,
                )
            )

        def close(self):
            calls.append(("close", {}))

    return SimpleNamespace(
        OpenAI=FakeOpenAI,
        DefaultHttpxClient=FakeHttpClient,
    )


def test_api_schemas_never_accept_plaintext_keys():
    for node_class in (hosted_api.PromptGenerateAPI, hosted_api.HostedVLMAPI):
        schema = node_class.INPUT_TYPES()
        all_inputs = {
            **schema.get("required", {}),
            **schema.get("optional", {}),
            **schema.get("hidden", {}),
        }
        assert "api_key" not in all_inputs
        assert "credential_source" in all_inputs
        assert "STRING" not in repr(all_inputs["credential_source"][0])
        assert "web_search" in all_inputs
        assert "output_format" in all_inputs
        assert "json_schema" in all_inputs
        assert "schema_api_style" in all_inputs


def test_json_schema_parser_blocks_remote_refs_and_bounds_input():
    for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
        with pytest.raises(ValueError, match="only local fragment"):
            hosted_api.parse_json_schema(
                "JSON Schema",
                json.dumps(
                    {
                        "type": "object",
                        "properties": {
                            "payload": {
                                keyword: "https://attacker.example/schema.json"
                            }
                        },
                    }
                ),
            )
    with pytest.raises(ValueError, match="64,000"):
        hosted_api.parse_json_schema("JSON Schema", "x" * 64_001)


def test_local_structured_output_validation_is_strict_and_normalized():
    schema_text = json.dumps(
        {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
            "additionalProperties": False,
        }
    )
    schema = hosted_api.parse_json_schema("JSON Schema", schema_text)
    assert hosted_api.validate_structured_output(
        '```json\n{"count": 2}\n```',
        "JSON Schema",
        schema,
    ) == '{\n  "count": 2\n}'
    with pytest.raises(RuntimeError, match=r"\$\.count \(type constraint\)"):
        hosted_api.validate_structured_output(
            '{"count": "two"}',
            "JSON Schema",
            schema,
        )
    with pytest.raises(RuntimeError, match="valid JSON"):
        hosted_api.validate_structured_output(
            '{"count":',
            "JSON Schema",
            schema,
        )


def test_provider_catalog_uses_current_bound_credentials_and_endpoints():
    assert len(hosted_api.PROVIDER_PROFILES) >= 18
    expected = {
        "OpenAI": "OPENAI_API_KEY",
        "Google Gemini": "GEMINI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "xAI": "XAI_API_KEY",
        "DeepSeek": "DEEPSEEK_API_KEY",
        "Groq": "GROQ_API_KEY",
        "Mistral": "MISTRAL_API_KEY",
        "Together AI": "TOGETHER_API_KEY",
        "OpenRouter": "OPENROUTER_API_KEY",
        "Custom / Local": "CUSTOM_API_KEY",
    }
    providers = {
        profile.provider: profile.api_key_env
        for profile in hosted_api.PROVIDER_PROFILES.values()
    }
    assert expected.items() <= providers.items()
    for profile in hosted_api.PROVIDER_PROFILES.values():
        if profile.base_url is not None:
            assert profile.base_url.startswith("https://")


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/v1",
        "ftp://127.0.0.1/v1",
        "https://user:secret@example.com/v1",
        "https://example.com/v1?api_key=secret",
        "not-a-url",
    ],
)
def test_custom_endpoint_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        hosted_api.validate_custom_base_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/v1",
        "http://[::1]:11434/v1",
        "http://localhost:1234/v1",
        "https://example.com/v1/",
    ],
)
def test_custom_endpoint_accepts_https_or_loopback(url):
    normalized, loopback = hosted_api.validate_custom_base_url(url)
    assert normalized.startswith(("http://", "https://"))
    assert loopback is (url.startswith("http://"))


def test_built_in_key_cannot_be_redirected(monkeypatch):
    profile = hosted_api.provider_profile("OpenAI — GPT-5.6 Terra")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-secret-value")
    with pytest.raises(ValueError, match="pinned to official hosts"):
        hosted_api.resolve_endpoint(profile, "https://attacker.example/v1")


def test_legacy_plaintext_value_is_rejected_without_echo(monkeypatch):
    profile = hosted_api.provider_profile("OpenAI — GPT-5.6 Terra")
    secret = "sk-legacy-plaintext-that-must-not-appear"
    with pytest.raises(ValueError) as captured:
        hosted_api.resolve_api_key(profile, secret, loopback=False)
    assert secret not in str(captured.value)
    assert "legacy plaintext API key was removed" in str(captured.value)


def test_redaction_removes_exact_encoded_and_header_credentials():
    secret = "sk-ant-example-SECRET_123456789"
    message = (
        f"Authorization: Bearer {secret}; api_key={secret}; "
        f"url=https://user:{secret}@example.com; encoded={secret}"
    )
    redacted = hosted_api.redact_sensitive(message, (secret,))
    assert secret not in redacted
    assert "Bearer" not in redacted
    assert "[REDACTED]" in redacted


def test_responses_call_is_stateless_private_and_provider_bound(monkeypatch):
    calls = []
    secret = "sk-openai-provider-bound-secret"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setattr(
        hosted_api,
        "require_module",
        lambda *_args: fake_openai_module(calls),
    )

    node = hosted_api.PromptGenerateAPI()
    assert not hasattr(node, "session_history")
    result = node.generate_prompt(
        "OpenAI — GPT-5.6 Terra",
        False,
        hosted_api.PROVIDER_CREDENTIAL,
        "A scene",
        "Improve it",
        0,
        0,
        stream_output=False,
    )
    assert result == ("secure response",)

    client_kwargs = next(payload for kind, payload in calls if kind == "client")
    assert client_kwargs["api_key"] == secret
    assert "base_url" not in client_kwargs
    assert client_kwargs["http_client"].kwargs["follow_redirects"] is False
    assert client_kwargs["http_client"].kwargs["trust_env"] is False

    request = next(payload for kind, payload in calls if kind == "responses")
    assert request["model"] == "gpt-5.6-terra"
    assert request["store"] is False
    assert "previous_response_id" not in request
    assert "metadata" not in request


def test_openai_combines_web_search_structured_output_and_stream_contract(
    monkeypatch,
):
    calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-structured-search")
    monkeypatch.setattr(
        hosted_api,
        "require_module",
        lambda import_name, *_args: (
            fake_openai_module(calls, response_text='{"answer":"grounded"}')
            if import_name == "openai"
            else __import__(import_name)
        ),
    )
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )
    result = hosted_api.PromptGenerateAPI().generate_prompt(
        "OpenAI — GPT-5.6 Sol",
        False,
        hosted_api.PROVIDER_CREDENTIAL,
        "Find a current fact",
        "",
        0,
        0,
        web_search=True,
        output_format="JSON Schema",
        json_schema=schema,
        stream_output=False,
    )
    assert result == ('{\n  "answer": "grounded"\n}',)
    request = next(payload for kind, payload in calls if kind == "responses")
    assert request["tools"] == [{"type": "web_search"}]
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    assert request["text"]["format"]["schema"]["required"] == ["answer"]
    assert "JSON Schema:" in request["instructions"]


def test_unsupported_web_search_fails_before_network(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-secret")
    with pytest.raises(ValueError, match="does not expose native web search"):
        hosted_api.PromptGenerateAPI().generate_prompt(
            "DeepSeek — V4 Flash",
            False,
            hosted_api.PROVIDER_CREDENTIAL,
            "Search now",
            "",
            0,
            0,
            web_search=True,
            stream_output=False,
        )


def test_responses_stream_collects_deltas_and_closes():
    class Stream(list):
        closed = False

        def close(self):
            self.closed = True

    stream = Stream(
        [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_text.delta", delta="hello "),
            SimpleNamespace(type="response.output_text.delta", delta="world"),
        ]
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: stream)
    )
    assert hosted_api._stream_responses(client, {"model": "test"}, None) == (
        "hello world"
    )
    assert stream.closed is True


def test_chat_stream_collects_deltas_and_closes():
    class Stream(list):
        closed = False

        def close(self):
            self.closed = True

    stream = Stream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="frame "))
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="ready"))
                ]
            ),
        ]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: stream)
        )
    )
    assert hosted_api._stream_chat(client, {"model": "test"}, None) == (
        "frame ready"
    )
    assert stream.closed is True


def test_provider_failure_never_echoes_api_key(monkeypatch):
    calls = []
    secret = "sk-secret-reflected-by-provider-123456"
    failure = RuntimeError(f"Authorization: Bearer {secret} api_key={secret}")
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setattr(
        hosted_api,
        "require_module",
        lambda *_args: fake_openai_module(calls, failure=failure),
    )

    with pytest.raises(RuntimeError) as captured:
        hosted_api.PromptGenerateAPI().generate_prompt(
            "OpenAI — GPT-5.6 Terra",
            False,
            hosted_api.PROVIDER_CREDENTIAL,
            "hello",
            "",
            0,
            0,
            stream_output=False,
        )
    assert secret not in str(captured.value)
    assert "[REDACTED]" in str(captured.value)


def test_anthropic_uses_native_messages_and_keeps_key_out_of_body(monkeypatch):
    calls = []
    secret = "sk-ant-native-secret"

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "content": [
                    {"type": "text", "text": "native Anthropic response"}
                ]
            }

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))

        def post(self, url, **kwargs):
            calls.append(("post", {"url": url, **kwargs}))
            return FakeResponse()

        def close(self):
            calls.append(("close", {}))

    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setattr(
        hosted_api,
        "require_module",
        lambda import_name, *_args: (
            SimpleNamespace(Client=FakeClient)
            if import_name == "httpx"
            else pytest.fail(f"Unexpected module request: {import_name}")
        ),
    )
    result = hosted_api.HostedVLMAPI().analyze(
        "Anthropic — Claude Sonnet 5",
        hosted_api.PROVIDER_CREDENTIAL,
        "Read this image.",
        "Be concise.",
        1,
        512,
        80,
        "auto",
        images=torch.rand((1, 48, 64, 3)),
        stream_output=False,
    )
    assert result == (
        "native Anthropic response",
        "claude-sonnet-5",
        1,
    )
    request = next(payload for kind, payload in calls if kind == "post")
    assert request["url"] == "https://api.anthropic.com/v1/messages"
    assert request["headers"]["x-api-key"] == secret
    assert secret not in repr(request["json"])
    content = request["json"]["messages"][0]["content"]
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"
    assert request["json"]["stream"] is False


def test_anthropic_native_stream_collects_text_deltas(monkeypatch):
    class FakeStreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'event: content_block_delta'
            yield (
                'data: {"type":"content_block_delta","delta":'
                '{"type":"text_delta","text":"hello "}}'
            )
            yield (
                'data: {"type":"content_block_delta","delta":'
                '{"type":"text_delta","text":"world"}}'
            )

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def stream(self, *_args, **_kwargs):
            return FakeStreamResponse()

        def close(self):
            return None

    monkeypatch.setattr(
        hosted_api,
        "require_module",
        lambda import_name, *_args: (
            SimpleNamespace(Client=FakeClient)
            if import_name == "httpx"
            else __import__(import_name)
        ),
    )
    profile = hosted_api.provider_profile("Anthropic — Claude Sonnet 5")
    result = hosted_api._call_anthropic_api(
        profile=profile,
        model=profile.model,
        endpoint=profile.base_url,
        api_key="sk-ant-stream",
        system_prompt="Be concise.",
        prompt="Hello",
        image_data=[],
        timeout_seconds=30,
        max_output_tokens=100,
        stream_output=True,
        use_system_proxy=False,
        unique_id=None,
        web_search=False,
        output_format="Text",
        output_schema=None,
    )
    assert result == "hello world"


def test_anthropic_native_search_and_structured_contracts(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"content": [{"type": "text", "text": '{"answer":"yes"}'}]}

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))

        def post(self, url, **kwargs):
            calls.append(("post", {"url": url, **kwargs}))
            return FakeResponse()

        def close(self):
            return None

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-contract")
    monkeypatch.setattr(
        hosted_api,
        "require_module",
        lambda import_name, *_args: (
            SimpleNamespace(Client=FakeClient)
            if import_name == "httpx"
            else __import__(import_name)
        ),
    )
    schema = (
        '{"type":"object","properties":{"answer":{"type":"string"}},'
        '"required":["answer"],"additionalProperties":false}'
    )

    result = hosted_api.PromptGenerateAPI().generate_prompt(
        "Anthropic — Claude Sonnet 5",
        False,
        hosted_api.PROVIDER_CREDENTIAL,
        "Return a value",
        "",
        0,
        0,
        output_format="JSON Schema",
        json_schema=schema,
        stream_output=False,
    )
    assert json.loads(result[0]) == {"answer": "yes"}
    structured = next(payload for kind, payload in calls if kind == "post")
    assert structured["json"]["output_config"]["format"]["type"] == "json_schema"

    calls.clear()
    hosted_api.PromptGenerateAPI().generate_prompt(
        "Anthropic — Claude Sonnet 5",
        False,
        hosted_api.PROVIDER_CREDENTIAL,
        "Search the web",
        "",
        0,
        0,
        web_search=True,
        output_format="JSON Schema",
        json_schema=schema,
        stream_output=False,
    )
    searched = next(payload for kind, payload in calls if kind == "post")
    assert searched["json"]["tools"][0]["type"] == "web_search_20260318"
    assert searched["json"]["tools"][0]["allowed_callers"] == ["direct"]
    assert "output_config" not in searched["json"]


def test_gemini_native_search_vision_and_schema_contract(monkeypatch):
    calls = []
    secret = "gemini-provider-secret"

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": '{"objects":["tree"]}'}]
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))

        def post(self, url, **kwargs):
            calls.append(("post", {"url": url, **kwargs}))
            return FakeResponse()

        def close(self):
            return None

    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setattr(
        hosted_api,
        "require_module",
        lambda import_name, *_args: (
            SimpleNamespace(Client=FakeClient)
            if import_name == "httpx"
            else __import__(import_name)
        ),
    )
    schema = (
        '{"type":"object","properties":{"objects":{"type":"array",'
        '"items":{"type":"string"}}},"required":["objects"]}'
    )
    result = hosted_api.HostedVLMAPI().analyze(
        "Google — Gemini 3.6 Flash",
        hosted_api.PROVIDER_CREDENTIAL,
        "Identify objects using current context.",
        "Be precise.",
        1,
        512,
        80,
        "auto",
        images=torch.rand((1, 32, 48, 3)),
        web_search=True,
        output_format="JSON Schema",
        json_schema=schema,
        stream_output=False,
    )
    assert json.loads(result[0]) == {"objects": ["tree"]}
    request = next(payload for kind, payload in calls if kind == "post")
    assert request["url"].endswith(
        "/models/gemini-3.6-flash:generateContent"
    )
    assert request["headers"]["x-goog-api-key"] == secret
    assert secret not in repr(request["json"])
    assert request["json"]["tools"] == [{"google_search": {}}]
    assert (
        request["json"]["generationConfig"]["responseFormat"]["text"]["schema"][
            "required"
        ]
        == ["objects"]
    )
    inline = request["json"]["contents"][0]["parts"][1]["inlineData"]
    assert inline["mimeType"] == "image/jpeg"
    assert inline["data"]


def test_gemini_native_stream_collects_sse_deltas(monkeypatch):
    class FakeStreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield (
                'data: {"candidates":[{"content":{"parts":'
                '[{"text":"frame "}]}}]}'
            )
            yield (
                'data: {"candidates":[{"content":{"parts":'
                '[{"text":"ready"}]}}]}'
            )

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def stream(self, *_args, **_kwargs):
            return FakeStreamResponse()

        def close(self):
            return None

    monkeypatch.setattr(
        hosted_api,
        "require_module",
        lambda import_name, *_args: (
            SimpleNamespace(Client=FakeClient)
            if import_name == "httpx"
            else __import__(import_name)
        ),
    )
    profile = hosted_api.provider_profile("Google — Gemini 3.6 Flash")
    result = hosted_api._call_gemini_api(
        profile=profile,
        model=profile.model,
        api_key="gemini-stream",
        system_prompt="Be concise.",
        prompt="Describe.",
        image_data=[],
        timeout_seconds=30,
        max_output_tokens=100,
        stream_output=True,
        use_system_proxy=False,
        unique_id=None,
        web_search=True,
        output_format="Text",
        output_schema=None,
    )
    assert result == "frame ready"


def test_vlm_uniformly_samples_and_bounds_image_batch(monkeypatch):
    calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-only")
    monkeypatch.setattr(
        hosted_api,
        "require_module",
        lambda *_args: fake_openai_module(calls),
    )
    images = torch.rand((10, 96, 128, 3), dtype=torch.float32)
    result = hosted_api.HostedVLMAPI().analyze(
        "OpenAI — GPT-5.6 Terra",
        hosted_api.PROVIDER_CREDENTIAL,
        "Compare the sampled frames.",
        "Be precise.",
        4,
        768,
        82,
        "low",
        images=images,
        stream_output=False,
    )
    assert result == ("secure response", "gpt-5.6-terra", 4)
    request = next(payload for kind, payload in calls if kind == "responses")
    content = request["input"][0]["content"]
    image_parts = [part for part in content if part["type"] == "input_image"]
    assert len(image_parts) == 4
    assert all(part["image_url"].startswith("data:image/jpeg;base64,") for part in image_parts)
    assert all(part["detail"] == "low" for part in image_parts)


def test_open_source_vlm_llama_cpp_schema_dialect_and_local_validation(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        hosted_api,
        "require_module",
        lambda import_name, *_args: (
            fake_openai_module(calls, response_text='{"objects":["cat"]}')
            if import_name == "openai"
            else __import__(import_name)
        ),
    )
    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["objects"],
            "additionalProperties": False,
        }
    )
    result = hosted_api.HostedVLMAPI().analyze(
        "Custom / Local — OpenAI compatible",
        hosted_api.LOCAL_NO_KEY,
        "List visible objects.",
        "Be precise.",
        1,
        512,
        80,
        "auto",
        images=torch.rand((1, 32, 48, 3)),
        base_url="http://127.0.0.1:8080/v1",
        model_override="local-vlm",
        output_format="JSON Schema",
        json_schema=schema,
        schema_api_style="llama.cpp JSON Schema",
        stream_output=False,
    )
    assert result == (
        '{\n  "objects": [\n    "cat"\n  ]\n}',
        "local-vlm",
        1,
    )
    request = next(payload for kind, payload in calls if kind == "chat")
    assert request["response_format"] == {
        "type": "json_schema",
        "schema": json.loads(schema),
    }
    image = request["messages"][1]["content"][1]
    assert image["type"] == "image_url"
    assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_custom_openai_schema_style_uses_standard_wrapper(monkeypatch):
    calls = []
    monkeypatch.setattr(
        hosted_api,
        "require_module",
        lambda import_name, *_args: (
            fake_openai_module(calls, response_text='{"ok":true}')
            if import_name == "openai"
            else __import__(import_name)
        ),
    )
    schema = '{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}'
    result, _, _ = hosted_api.execute_hosted(
        model_name="Custom / Local — OpenAI compatible",
        credential_source=hosted_api.LOCAL_NO_KEY,
        prompt="Return status.",
        system_prompt="Be exact.",
        base_url="http://localhost:8000/v1",
        model_override="local",
        api_mode="Chat Completions",
        timeout_seconds=30,
        max_output_tokens=100,
        reasoning_effort="none",
        seed=0,
        stream_output=False,
        use_system_proxy=False,
        unique_id=None,
        output_format="JSON Schema",
        json_schema=schema,
    )
    assert json.loads(result) == {"ok": True}
    request = next(payload for kind, payload in calls if kind == "chat")
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"]["schema"]["required"] == [
        "ok"
    ]


def test_groq_auto_uses_documented_chat_route_for_structured_output(monkeypatch):
    calls = []
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-structured")
    monkeypatch.setattr(
        hosted_api,
        "require_module",
        lambda import_name, *_args: (
            fake_openai_module(calls, response_text='{"ok":true}')
            if import_name == "openai"
            else __import__(import_name)
        ),
    )
    hosted_api.execute_hosted(
        model_name="Groq — GPT-OSS 20B",
        credential_source=hosted_api.PROVIDER_CREDENTIAL,
        prompt="Return status.",
        system_prompt="Be exact.",
        base_url="",
        model_override="",
        api_mode="Auto",
        timeout_seconds=30,
        max_output_tokens=100,
        reasoning_effort="none",
        seed=0,
        stream_output=False,
        use_system_proxy=False,
        unique_id=None,
        output_format="JSON Schema",
        json_schema=(
            '{"type":"object","properties":{"ok":{"type":"boolean"}},'
            '"required":["ok"],"additionalProperties":false}'
        ),
    )
    assert any(kind == "chat" for kind, _payload in calls)
    assert not any(kind == "responses" for kind, _payload in calls)


def test_frontend_scrubs_legacy_key_before_graph_configuration():
    web_root = Path(__file__).resolve().parents[1] / "web" / "js"
    source = (
        web_root / "apiSecurity.js"
    ).read_text("utf-8")
    assert "beforeConfigureGraph" in source
    assert "delete values.api_key" in source
    assert "CREDENTIAL_WIDGET_INDEX = 2" in source
    view_text = (web_root / "viewText.js").read_text("utf-8")
    assert '"PromptGenerateAPI"' in view_text
    assert '"HostedVLMAPI"' in view_text
