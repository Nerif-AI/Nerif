"""Tests for enhanced TTS module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from nerif.model.audio_model import SpeechModel
from nerif.tts.synthesizer import Synthesizer


class TestSpeechModelEnhanced:
    def test_init_defaults(self):
        model = SpeechModel(api_key="k", base_url="http://test/v1")
        assert model.model == "tts-1"
        assert model.voice == "alloy"
        assert model.response_format == "mp3"
        assert model.speed == 1.0

    def test_init_custom(self):
        model = SpeechModel(api_key="k", base_url="http://test/v1",
                            model="tts-1-hd", voice="nova", response_format="opus", speed=1.5)
        assert model.model == "tts-1-hd"
        assert model.voice == "nova"

    def test_voices_constant(self):
        assert "alloy" in SpeechModel.VOICES
        assert "nova" in SpeechModel.VOICES
        assert len(SpeechModel.VOICES) >= 6

    def test_text_to_speech_params(self):
        model = SpeechModel(api_key="k", base_url="http://test/v1")
        mock_resp = MagicMock()
        mock_resp.content = b"audio_bytes"
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = model.text_to_speech("Hello", voice="echo", response_format="opus", speed=1.5)
            assert result == b"audio_bytes"
            body = mock_post.call_args[1]["json"]
            assert body["voice"] == "echo"
            assert body["response_format"] == "opus"
            assert body["speed"] == 1.5

    def test_text_to_speech_file(self, tmp_path):
        model = SpeechModel(api_key="k", base_url="http://test/v1")
        mock_resp = MagicMock()
        mock_resp.content = b"audio_bytes"
        mock_resp.raise_for_status = MagicMock()

        output = tmp_path / "output.mp3"
        with patch("httpx.post", return_value=mock_resp):
            result = model.text_to_speech_file("Hello", output)
            assert result == output
            assert output.read_bytes() == b"audio_bytes"

    def test_atext_to_speech(self):
        model = SpeechModel(api_key="k", base_url="http://test/v1")
        mock_resp = MagicMock()
        mock_resp.content = b"async_audio"
        mock_resp.raise_for_status = MagicMock()

        async def run():
            with patch("httpx.AsyncClient") as MockClient:
                instance = MockClient.return_value.__aenter__.return_value
                instance.post = AsyncMock(return_value=mock_resp)
                result = await model.atext_to_speech("Hello")
                assert result == b"async_audio"

        asyncio.get_event_loop().run_until_complete(run())

    def test_http_error(self):
        model = SpeechModel(api_key="k", base_url="http://test/v1")
        mock_resp = httpx.Response(401, request=httpx.Request("POST", "http://test"))

        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(httpx.HTTPStatusError):
                model.text_to_speech("Hello")


class TestSynthesizer:
    def test_speak(self):
        s = Synthesizer(api_key="k", base_url="http://test/v1")
        with patch.object(s._speech_model, "text_to_speech", return_value=b"audio"):
            assert s.speak("hello") == b"audio"

    def test_speak_to_file(self, tmp_path):
        s = Synthesizer(api_key="k", base_url="http://test/v1")
        output = tmp_path / "out.mp3"
        with patch.object(s._speech_model, "text_to_speech_file", return_value=output):
            assert s.speak_to_file("hello", output) == output

    def test_available_voices(self):
        s = Synthesizer(api_key="k", base_url="http://test/v1")
        assert "alloy" in s.available_voices

    def test_aspeak(self):
        s = Synthesizer(api_key="k", base_url="http://test/v1")

        async def run():
            with patch.object(s._speech_model, "atext_to_speech",
                              new_callable=AsyncMock, return_value=b"async_audio"):
                result = await s.aspeak("hello")
                assert result == b"async_audio"

        asyncio.get_event_loop().run_until_complete(run())
