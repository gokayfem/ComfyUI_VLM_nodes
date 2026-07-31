from types import SimpleNamespace

import numpy as np
import pytest
from ComfyUI_VLM_nodes.nodes import minimax_music


def generation_request(**overrides):
    values = {
        "region": "global_en",
        "model": "music-3.0",
        "prompt": "Reflective acoustic pop",
        "lyrics": "[Verse]\nA quiet road under evening light",
        "stream": False,
        "output_format": "hex",
        "audio_format": "mp3",
        "sample_rate": 44100,
        "bitrate": 256000,
        "lyrics_optimizer": False,
        "is_instrumental": False,
        "aigc_watermark": False,
        "audio_url": "",
        "audio_base64": "",
        "cover_feature_id": "",
    }
    values.update(overrides)
    return minimax_music.build_music_request(**values)


def test_music_contract_matches_current_models_regions_and_formats():
    assert minimax_music.REGION_ENDPOINTS == {
        "global_en": "https://api.minimax.io/v1/music_generation",
        "cn_zh": "https://api.minimaxi.com/v1/music_generation",
    }
    assert minimax_music.DEFAULT_MODEL == "music-3.0"
    assert minimax_music.GENERATION_MODELS == (
        "music-3.0",
        "music-2.6",
        "music-3.0-free",
        "music-2.6-free",
    )
    assert minimax_music.COVER_MODELS == ("music-cover", "music-cover-free")
    assert minimax_music.OUTPUT_FORMATS == ("url", "hex")
    assert minimax_music.AUDIO_FORMATS == ("mp3", "wav", "pcm")
    assert {
        "model",
        "prompt",
        "lyrics",
        "stream",
        "output_format",
        "audio_setting",
        "lyrics_optimizer",
        "is_instrumental",
        "audio_url",
        "audio_base64",
        "cover_feature_id",
    } == minimax_music.REQUEST_FIELDS
    assert minimax_music.REGIONAL_FIELDS == {
        "global_en": (),
        "cn_zh": ("aigc_watermark",),
    }


def test_generation_request_covers_generation_and_cn_fields():
    request = generation_request(
        region="cn_zh",
        stream=True,
        lyrics="",
        lyrics_optimizer=True,
        aigc_watermark=True,
        audio_format="wav",
        sample_rate=32000,
        bitrate=128000,
    )
    assert request == {
        "model": "music-3.0",
        "prompt": "Reflective acoustic pop",
        "stream": True,
        "output_format": "hex",
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "wav",
        },
        "lyrics_optimizer": True,
        "is_instrumental": False,
        "aigc_watermark": True,
    }


@pytest.mark.parametrize(
    ("source", "value"),
    [
        ("audio_url", "https://media.example/reference.wav"),
        ("audio_base64", "dGVzdA=="),
        ("cover_feature_id", "feature-123"),
    ],
)
def test_cover_request_supports_each_documented_source(source, value):
    overrides = {
        "model": "music-cover",
        "prompt": "Warm orchestral cover",
        "lyrics": "Updated words for the cover",
        source: value,
    }
    request = generation_request(**overrides)
    assert request[source] == value
    assert "lyrics_optimizer" not in request
    assert "is_instrumental" not in request


def test_streaming_requires_hex_and_cover_sources_are_exclusive():
    with pytest.raises(ValueError, match="output_format='hex'"):
        generation_request(stream=True, output_format="url")
    with pytest.raises(ValueError, match="exactly one"):
        generation_request(
            model="music-cover",
            prompt="Warm orchestral cover",
            audio_url="https://media.example/reference.wav",
            audio_base64="dGVzdA==",
        )


def test_stream_response_joins_hex_chunks_and_requires_completion():
    class Response:
        def iter_lines(self):
            return iter(
                [
                    'data: {"data":{"status":1,"audio":"0001"},'
                    '"base_resp":{"status_code":0}}',
                    'data: {"data":{"status":2,"audio":"0203"},'
                    '"extra_info":{"music_sample_rate":32000,"music_channel":2},'
                    '"base_resp":{"status_code":0}}',
                    "data: [DONE]",
                ]
            )

    audio, metadata = minimax_music._stream_audio(Response())
    assert audio == "00010203"
    assert metadata == {"music_sample_rate": 32000, "music_channel": 2}


def test_url_and_hex_response_decoding():
    class DownloadResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            return iter((b"ab", b"cd"))

    class Client:
        def stream(self, method, url):
            assert method == "GET"
            assert url == "https://media.example/music.wav"
            return DownloadResponse()

    client = Client()
    assert minimax_music._audio_bytes(client, "61626364", "hex") == b"abcd"
    assert (
        minimax_music._audio_bytes(
            client,
            "https://media.example/music.wav",
            "url",
        )
        == b"abcd"
    )


def test_pcm_decoding_uses_response_sample_rate_and_channel_count():
    content = np.array([0, 32767, -32768, 0], dtype="<i2").tobytes()
    samples, sample_rate = minimax_music._decode_audio(
        content,
        "pcm",
        44100,
        {"music_sample_rate": 32000, "music_channel": 2},
    )
    assert samples.shape == (2, 2)
    assert sample_rate == 32000
    assert samples[0, 1] == pytest.approx(32767 / 32768)


def test_node_posts_to_fixed_region_and_returns_comfy_audio(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": {"status": 2, "audio": "0102"},
                "extra_info": {"music_sample_rate": 44100, "music_channel": 2},
                "base_resp": {"status_code": 0},
            }

    class Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        def post(self, endpoint, *, headers, json):
            captured["endpoint"] = endpoint
            captured["headers"] = headers
            captured["request"] = json
            return Response()

        def close(self):
            captured["closed"] = True

    def fake_soundfile_read(buffer, **kwargs):
        assert buffer.read() == b"\x01\x02"
        assert kwargs == {"dtype": "float32", "always_2d": True}
        return np.zeros((8, 2), dtype=np.float32), 44100

    def fake_require_module(name, *_args):
        if name == "httpx":
            return SimpleNamespace(Client=Client)
        if name == "soundfile":
            return SimpleNamespace(read=fake_soundfile_read)
        raise AssertionError(name)

    monkeypatch.setenv(minimax_music.API_KEY_ENV, "test-key-not-for-production")
    monkeypatch.setattr(minimax_music, "require_module", fake_require_module)
    result = minimax_music.MiniMaxMusicNode().generate_music(
        region="global_en",
        model="music-3.0",
        prompt="Reflective acoustic pop",
        lyrics="[Verse]\nA quiet road under evening light",
        stream=False,
        output_format="hex",
        audio_format="wav",
        sample_rate=44100,
        bitrate=256000,
        lyrics_optimizer=False,
        is_instrumental=False,
        aigc_watermark=False,
    )
    assert captured["endpoint"] == minimax_music.REGION_ENDPOINTS["global_en"]
    assert captured["headers"]["Authorization"].startswith("Bearer ")
    assert captured["client"] == {
        "timeout": 600.0,
        "follow_redirects": False,
        "trust_env": False,
    }
    assert captured["closed"] is True
    assert len(result) == 3
    assert result[1] == 44100
    assert result[2]["waveform"].shape == (1, 2, 8)


def test_request_failures_redact_the_resolved_key(monkeypatch):
    resolved_value = "unit-key"

    class Client:
        def __init__(self, **_kwargs):
            pass

        def post(self, *_args, **_kwargs):
            raise RuntimeError(f"Authorization: Bearer {resolved_value}")

        def close(self):
            pass

    monkeypatch.setenv(minimax_music.API_KEY_ENV, resolved_value)
    monkeypatch.setattr(
        minimax_music,
        "require_module",
        lambda *_args: SimpleNamespace(Client=Client),
    )
    with pytest.raises(RuntimeError) as captured:
        minimax_music.MiniMaxMusicNode().generate_music(
            region="global_en",
            model="music-3.0",
            prompt="Reflective acoustic pop",
            lyrics="[Verse]\nA quiet road under evening light",
            stream=False,
            output_format="hex",
            audio_format="mp3",
            sample_rate=44100,
            bitrate=256000,
            lyrics_optimizer=False,
            is_instrumental=False,
            aigc_watermark=False,
        )
    assert resolved_value not in str(captured.value)
    assert "[REDACTED]" in str(captured.value)
