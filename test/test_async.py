"""Tests for async/await support in Nerif."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nerif.model.audio_inference import AudioInferenceModel
from nerif.model.audio_model import AudioModel, SpeechModel
from nerif.model.image_generation import ImageGenerationModel, NanoBananaModel
from nerif.model.model import SimpleChatModel, SimpleEmbeddingModel
from nerif.utils import get_embedding_async, get_model_response_async, get_model_response_stream_async

# ---------------------------------------------------------------------------
# Helpers: build minimal mock HTTP responses
# ---------------------------------------------------------------------------


def _make_chat_response_json(content="hello"):
    return {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _make_embedding_response_json(dim=4):
    return {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [{"object": "embedding", "embedding": [0.1] * dim, "index": 0}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
    }


def _make_mock_response(json_data, status_code=200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status = MagicMock()
    mock_resp.content = b"audio_bytes"
    return mock_resp


# ---------------------------------------------------------------------------
# Tests: get_model_response_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_model_response_async_openai():
    """get_model_response_async routes to OpenAI-compat endpoint and returns response."""
    mock_resp = _make_mock_response(_make_chat_response_json("world"))

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    messages = [{"role": "user", "content": "hi"}]

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await get_model_response_async(messages, model="gpt-4o")

    assert result.choices[0].message.content == "world"
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_get_model_response_async_anthropic():
    """get_model_response_async routes to Anthropic endpoint."""
    anthropic_json = {
        "id": "msg-1",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": "anthropic reply"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    mock_resp = _make_mock_response(anthropic_json)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    messages = [{"role": "user", "content": "hello"}]

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await get_model_response_async(
            messages,
            model="anthropic/claude-3-5-sonnet-20241022",
        )

    assert result.choices[0].message.content == "anthropic reply"


@pytest.mark.asyncio
async def test_get_model_response_async_gemini():
    """get_model_response_async routes to Gemini endpoint."""
    gemini_json = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "gemini reply"}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
    }
    mock_resp = _make_mock_response(gemini_json)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    messages = [{"role": "user", "content": "hello"}]

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await get_model_response_async(
            messages,
            model="gemini/gemini-2.0-flash",
        )

    assert result.choices[0].message.content == "gemini reply"


# ---------------------------------------------------------------------------
# Tests: get_embedding_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_embedding_async():
    """get_embedding_async returns embedding data."""
    mock_resp = _make_mock_response(_make_embedding_response_json(dim=4))

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await get_embedding_async("test text", model="text-embedding-3-small")

    assert len(result.data) == 1
    assert len(result.data[0]["embedding"]) == 4


# ---------------------------------------------------------------------------
# Tests: SimpleChatModel.achat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_chat_model_achat():
    """SimpleChatModel.achat() returns text response."""
    mock_resp = _make_mock_response(_make_chat_response_json("async response"))

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    model = SimpleChatModel(model="gpt-4o")

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await model.achat("hello")

    assert result == "async response"


@pytest.mark.asyncio
async def test_simple_chat_model_achat_append():
    """SimpleChatModel.achat() appends to history when append=True."""
    mock_resp = _make_mock_response(_make_chat_response_json("reply"))

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    model = SimpleChatModel(model="gpt-4o")

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await model.achat("hello", append=True)

    assert result == "reply"
    # system + user + assistant = 3 messages
    assert len(model.messages) == 3


@pytest.mark.asyncio
async def test_simple_chat_model_achat_no_append():
    """SimpleChatModel.achat() resets history when append=False."""
    mock_resp = _make_mock_response(_make_chat_response_json("reply"))

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    model = SimpleChatModel(model="gpt-4o")

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await model.achat("hello", append=False)

    assert result == "reply"
    # After reset: just system message
    assert len(model.messages) == 1


# ---------------------------------------------------------------------------
# Tests: SimpleChatModel.astream_chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_chat_model_astream_chat():
    """SimpleChatModel.astream_chat() yields text chunks."""

    async def mock_stream(*args, **kwargs):
        from nerif.utils.utils import StreamChunk

        yield StreamChunk(content="chunk1")
        yield StreamChunk(content="chunk2")
        yield StreamChunk(content="", finish_reason="stop")

    model = SimpleChatModel(model="gpt-4o")

    with patch("nerif.model.model.get_model_response_stream_async", side_effect=mock_stream):
        chunks = []
        async for chunk in model.astream_chat("hello"):
            chunks.append(chunk)

    assert chunks == ["chunk1", "chunk2"]


# ---------------------------------------------------------------------------
# Tests: SimpleEmbeddingModel.aembed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_embedding_model_aembed():
    """SimpleEmbeddingModel.aembed() returns a list of floats."""
    mock_resp = _make_mock_response(_make_embedding_response_json(dim=8))

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    model = SimpleEmbeddingModel(model="text-embedding-3-small")

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await model.aembed("embed this")

    assert isinstance(result, list)
    assert len(result) == 8


# ---------------------------------------------------------------------------
# Tests: AudioModel.atranscribe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_model_atranscribe(tmp_path):
    """AudioModel.atranscribe() sends a request and returns JSON."""
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake_wav_data")

    mock_resp = _make_mock_response({"text": "hello world"})
    mock_resp.json.return_value = {"text": "hello world"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    audio = AudioModel(api_key="test-key")

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await audio.atranscribe(audio_file)

    assert result == {"text": "hello world"}
    mock_client.post.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: SpeechModel.atext_to_speech
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speech_model_atext_to_speech():
    """SpeechModel.atext_to_speech() sends a request and returns bytes."""
    mock_resp = _make_mock_response({})
    mock_resp.content = b"audio_bytes"

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    speech = SpeechModel(api_key="test-key")

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await speech.atext_to_speech("hello", voice="alloy")

    assert result == b"audio_bytes"
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    body = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs.kwargs.get("json", {})
    assert body.get("input") == "hello"
    assert body.get("voice") == "alloy"


# ---------------------------------------------------------------------------
# Tests: ImageGenerationModel.agenerate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_generation_model_agenerate():
    """ImageGenerationModel.agenerate() returns ImageGenerationResult."""
    payload = {
        "created": 1234567890,
        "data": [{"b64_json": "abc123", "url": None, "revised_prompt": "a cat"}],
    }
    mock_resp = _make_mock_response(payload)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    img_model = ImageGenerationModel(model="gpt-image-1", api_key="test-key")

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await img_model.agenerate("a cat")

    assert len(result.data) == 1
    assert result.data[0].b64_json == "abc123"
    assert result.data[0].revised_prompt == "a cat"


# ---------------------------------------------------------------------------
# Tests: NanoBananaModel.agenerate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nano_banana_model_agenerate():
    """NanoBananaModel.agenerate() returns ImageGenerationResult."""
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Here is an image"},
                        {"inlineData": {"data": "base64imgdata", "mimeType": "image/png"}},
                    ]
                }
            }
        ]
    }
    mock_resp = _make_mock_response(payload)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    nano = NanoBananaModel(api_key="test-key")

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await nano.agenerate("generate something")

    assert result.text == "Here is an image"
    assert len(result.data) == 1
    assert result.data[0].b64_json == "base64imgdata"
    assert result.data[0].mime_type == "image/png"


# ---------------------------------------------------------------------------
# Tests: AudioInferenceModel async methods
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_inference_model_aanalyze_url():
    """AudioInferenceModel.aanalyze_url() calls achat with audio URL."""
    model = AudioInferenceModel(model="gpt-4o-audio-preview")

    async def mock_achat(message, **kwargs):
        return "audio description"

    with patch.object(model._chat_model, "achat", side_effect=mock_achat):
        result = await model.aanalyze_url("http://example.com/audio.wav", prompt="describe this")

    assert result == "audio description"


@pytest.mark.asyncio
async def test_audio_inference_model_aanalyze_path(tmp_path):
    """AudioInferenceModel.aanalyze_path() calls achat with audio path."""
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake_audio")

    model = AudioInferenceModel(model="gpt-4o-audio-preview")

    async def mock_achat(message, **kwargs):
        return "path description"

    with patch.object(model._chat_model, "achat", side_effect=mock_achat):
        result = await model.aanalyze_path(str(audio_file))

    assert result == "path description"


@pytest.mark.asyncio
async def test_audio_inference_model_aanalyze_base64():
    """AudioInferenceModel.aanalyze_base64() calls achat with base64 audio."""
    model = AudioInferenceModel(model="gpt-4o-audio-preview")

    async def mock_achat(message, **kwargs):
        return "base64 description"

    with patch.object(model._chat_model, "achat", side_effect=mock_achat):
        result = await model.aanalyze_base64("dGVzdA==", format="wav")

    assert result == "base64 description"


# ---------------------------------------------------------------------------
# Tests: Concurrent async calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_achat_calls():
    """Multiple achat() calls can run concurrently with asyncio.gather."""
    responses = ["response_1", "response_2", "response_3"]
    call_count = 0

    async def fake_get_model_response_async(messages, **kwargs):
        nonlocal call_count
        idx = call_count
        call_count += 1
        from nerif.utils.utils import (
            ChatCompletionResponse,
            _Choice,
            _Message,
            _Usage,
        )

        return ChatCompletionResponse(
            id=f"id-{idx}",
            choices=[_Choice(index=0, message=_Message(role="assistant", content=responses[idx]))],
            usage=_Usage(),
        )

    models = [SimpleChatModel(model="gpt-4o") for _ in range(3)]

    with patch("nerif.model.model.get_model_response_async", side_effect=fake_get_model_response_async):
        results = await asyncio.gather(*[m.achat(f"message {i}") for i, m in enumerate(models)])

    assert results == ["response_1", "response_2", "response_3"]


@pytest.mark.asyncio
async def test_concurrent_aembed_calls():
    """Multiple aembed() calls can run concurrently."""
    call_count = 0

    async def fake_get_embedding_async(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        from nerif.utils.utils import EmbeddingResponse, _Usage

        return EmbeddingResponse(
            model="text-embedding-3-small",
            data=[{"embedding": [0.1, 0.2], "index": 0}],
            usage=_Usage(),
        )

    models = [SimpleEmbeddingModel() for _ in range(3)]

    with patch("nerif.model.model.get_embedding_async", side_effect=fake_get_embedding_async):
        results = await asyncio.gather(*[m.aembed(f"text {i}") for i, m in enumerate(models)])

    assert len(results) == 3
    assert call_count == 3


# ---------------------------------------------------------------------------
# Tests: Async streaming via get_model_response_stream_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_model_response_stream_async():
    """get_model_response_stream_async yields StreamChunk objects."""
    sse_lines = [
        'data: {"choices": [{"delta": {"content": "hello"}, "finish_reason": null}], "usage": null}',
        'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": null}], "usage": null}',
        'data: {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}',
        "data: [DONE]",
    ]

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line

    mock_stream_resp = MagicMock()
    mock_stream_resp.raise_for_status = MagicMock()
    mock_stream_resp.aiter_lines = mock_aiter_lines

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_resp)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)

    messages = [{"role": "user", "content": "hi"}]

    with patch("httpx.AsyncClient", return_value=mock_client):
        chunks = []
        async for chunk in get_model_response_stream_async(messages, model="gpt-4o"):
            chunks.append(chunk)

    text_chunks = [c.content for c in chunks if c.content]
    assert "hello" in text_chunks
    assert " world" in text_chunks


# ---------------------------------------------------------------------------
# Tests: Retry works with async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_model_response_async_retries_on_500():
    """get_model_response_async retries on 500 errors."""
    import httpx

    call_count = 0

    async def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            mock_err_resp = MagicMock()
            mock_err_resp.status_code = 500
            mock_err_resp.headers = {}
            raise httpx.HTTPStatusError("Server Error", request=MagicMock(), response=mock_err_resp)
        return _make_mock_response(_make_chat_response_json("finally"))

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    from nerif.utils.retry import RetryConfig

    fast_retry = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
    messages = [{"role": "user", "content": "hi"}]

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await get_model_response_async(messages, model="gpt-4o", retry_config=fast_retry)

    assert result.choices[0].message.content == "finally"
    assert call_count == 3
