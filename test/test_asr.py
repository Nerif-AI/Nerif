"""Tests for enhanced ASR module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from nerif.asr.transcriber import Transcriber
from nerif.model.audio_model import AudioModel


class TestAudioModelEnhanced:
    def test_init_custom(self):
        model = AudioModel(
            model="whisper-1", api_key="k", base_url="http://x/v1", language="en", response_format="verbose_json"
        )
        assert model.language == "en"
        assert model.response_format == "verbose_json"

    def test_transcribe_with_options(self):
        model = AudioModel(api_key="k", base_url="http://test/v1")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "hello"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = model.transcribe(
                file=("test.wav", b"audio_data"),
                prompt="context",
                language="en",
                response_format="verbose_json",
                temperature=0.5,
            )
            assert result == {"text": "hello"}
            call_kwargs = mock_post.call_args
            assert call_kwargs[1]["data"]["language"] == "en"
            assert call_kwargs[1]["data"]["prompt"] == "context"
            assert call_kwargs[1]["data"]["response_format"] == "verbose_json"

    def test_transcribe_file_path(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        model = AudioModel(api_key="k", base_url="http://test/v1")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "hello"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp):
            result = model.transcribe(file=audio_file)
            assert result == {"text": "hello"}

    def test_transcribe_http_error(self):
        model = AudioModel(api_key="k", base_url="http://test/v1")
        mock_resp = httpx.Response(401, request=httpx.Request("POST", "http://test"))

        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(httpx.HTTPStatusError):
                model.transcribe(file=("test.wav", b"data"))

    def test_translate(self):
        model = AudioModel(api_key="k", base_url="http://test/v1")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "hello in english"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = model.translate(file=("test.wav", b"audio"))
            assert result == {"text": "hello in english"}
            assert "/audio/translations" in mock_post.call_args[0][0]

    def test_atranscribe(self):
        model = AudioModel(api_key="k", base_url="http://test/v1")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "async hello"}
        mock_resp.raise_for_status = MagicMock()

        async def run():
            with patch("httpx.AsyncClient") as MockClient:
                instance = MockClient.return_value.__aenter__.return_value
                instance.post = AsyncMock(return_value=mock_resp)
                result = await model.atranscribe(file=("test.wav", b"data"))
                assert result == {"text": "async hello"}

        asyncio.run(run())

    def test_atranslate(self):
        model = AudioModel(api_key="k", base_url="http://test/v1")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "translated"}
        mock_resp.raise_for_status = MagicMock()

        async def run():
            with patch("httpx.AsyncClient") as MockClient:
                instance = MockClient.return_value.__aenter__.return_value
                instance.post = AsyncMock(return_value=mock_resp)
                result = await model.atranslate(file=("test.wav", b"data"))
                assert result == {"text": "translated"}

        asyncio.run(run())


class TestTranscriber:
    def test_transcribe_returns_text(self):
        t = Transcriber(api_key="k", base_url="http://test/v1")
        with patch.object(t._audio_model, "transcribe", return_value={"text": "hello"}):
            assert t.transcribe(file=("f", b"d")) == "hello"

    def test_translate_returns_text(self):
        t = Transcriber(api_key="k", base_url="http://test/v1")
        with patch.object(t._audio_model, "translate", return_value={"text": "english"}):
            assert t.translate(file=("f", b"d")) == "english"

    def test_atranscribe_returns_text(self):
        t = Transcriber(api_key="k", base_url="http://test/v1")

        async def run():
            with patch.object(
                t._audio_model, "atranscribe", new_callable=AsyncMock, return_value={"text": "async hello"}
            ):
                result = await t.atranscribe(file=("f", b"d"))
                assert result == "async hello"

        asyncio.run(run())

    def test_empty_response(self):
        t = Transcriber(api_key="k", base_url="http://test/v1")
        with patch.object(t._audio_model, "transcribe", return_value={}):
            assert t.transcribe(file=("f", b"d")) == ""
