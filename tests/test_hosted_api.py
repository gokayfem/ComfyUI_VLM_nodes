from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ComfyUI_VLM_nodes.nodes import hosted_api


class FakeHttpClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeResponses:
    def __init__(self, calls, failure=None):
        self.calls = calls
        self.failure = failure

    def create(self, **kwargs):
        self.calls.append(("responses", kwargs))
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(output_text="secure response")


class FakeChatCompletions:
    def __init__(self, calls, failure=None):
        self.calls = calls
        self.failure = failure

    def create(self, **kwargs):
        self.calls.append(("chat", kwargs))
        if self.failure is not None:
            raise self.failure
        message = SimpleNamespace(content="secure response")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def fake_openai_module(calls, failure=None):
    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))
            self.responses = FakeResponses(calls, failure=failure)
            self.chat = SimpleNamespace(
                completions=FakeChatCompletions(calls, failure=failure)
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
