"""Tests for AudioModel (ASR/transcription) and SpeechModel (TTS)."""

import os
from unittest.mock import MagicMock, mock_open, patch

import httpx
import pytest

from nerif.model.audio_model import AudioModel, SpeechModel


# ===========================================================================
# AudioModel initialization
# ===========================================================================
class TestAudioModelInit:
    def test_defaults_from_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_API_BASE": "https://custom.api/v1"}):
            import importlib

            import nerif.model.audio_model as am

            importlib.reload(am)

            model = am.AudioModel()
            assert model.api_key == "test-key"
            assert model.base_url == "https://custom.api/v1"

    def test_explicit_params_override_env(self):
        model = AudioModel(api_key="my-key", base_url="https://my-api.com/v1/")
        assert model.api_key == "my-key"
        assert model.base_url == "https://my-api.com/v1"

    def test_trailing_slashes_all_stripped(self):
        model = AudioModel(api_key="k", base_url="https://api.test.com/v1///")
        assert model.base_url == "https://api.test.com/v1"

    def test_base_url_single_trailing_slash_stripped(self):
        model = AudioModel(api_key="k", base_url="https://api.test.com/v1/")
        assert model.base_url == "https://api.test.com/v1"


# ===========================================================================
# AudioModel.transcribe
# ===========================================================================
class TestAudioModelTranscribe:
    @patch("nerif.model.audio_model.httpx.post")
    def test_transcribe_sends_correct_request(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "hello world"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        model = AudioModel(api_key="test-key", base_url="https://api.example.com/v1")

        fake_audio = b"fake audio bytes"
        m = mock_open(read_data=fake_audio)
        with patch("builtins.open", m):
            result = model.transcribe("/tmp/audio.wav")

        assert result == {"text": "hello world"}
        mock_post.assert_called_once()

        call_kwargs = mock_post.call_args
        url = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1].get("url")
        assert url == "https://api.example.com/v1/audio/transcriptions"

        headers = call_kwargs[1]["headers"]
        assert headers["Authorization"] == "Bearer test-key"

        data = call_kwargs[1]["data"]
        assert data["model"] == "whisper-1"

    @patch("nerif.model.audio_model.httpx.post")
    def test_transcribe_with_real_file(self, mock_post, tmp_path):
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "transcribed text"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        audio_file = tmp_path / "recording.wav"
        audio_file.write_bytes(b"RIFF fake wav data")

        model = AudioModel(api_key="k", base_url="https://api.example.com/v1")
        result = model.transcribe(audio_file)

        assert result["text"] == "transcribed text"

    @patch("nerif.model.audio_model.httpx.post")
    def test_transcribe_raises_on_http_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=MagicMock()
        )
        mock_post.return_value = mock_response

        model = AudioModel(api_key="k", base_url="https://api.example.com/v1")

        m = mock_open(read_data=b"audio")
        with patch("builtins.open", m):
            with pytest.raises(httpx.HTTPStatusError):
                model.transcribe("/tmp/audio.wav")

    @patch("nerif.model.audio_model.httpx.post")
    def test_transcribe_sends_file_tuple(self, mock_post):
        """Verify the files dict contains a tuple with the filename."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "ok"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        model = AudioModel(api_key="k", base_url="https://api.example.com/v1")

        m = mock_open(read_data=b"audio")
        with patch("builtins.open", m):
            model.transcribe("/tmp/my_recording.wav")

        call_kwargs = mock_post.call_args[1]
        files = call_kwargs["files"]
        assert "file" in files
        assert files["file"][0] == "/tmp/my_recording.wav"


# ===========================================================================
# SpeechModel initialization
# ===========================================================================
class TestSpeechModelInit:
    def test_defaults_from_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "speech-key", "OPENAI_API_BASE": "https://speech.api/v1"}):
            import importlib

            import nerif.model.audio_model as am

            importlib.reload(am)

            model = am.SpeechModel()
            assert model.api_key == "speech-key"
            assert model.base_url == "https://speech.api/v1"

    def test_explicit_params_override_env(self):
        model = SpeechModel(api_key="my-speech-key", base_url="https://my-speech.com/v1/")
        assert model.api_key == "my-speech-key"
        assert model.base_url == "https://my-speech.com/v1"


# ===========================================================================
# SpeechModel.text_to_speech
# ===========================================================================
class TestSpeechModelTextToSpeech:
    @patch("nerif.model.audio_model.httpx.post")
    def test_sends_correct_request(self, mock_post):
        mock_response = MagicMock()
        mock_response.content = b"audio bytes output"
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        model = SpeechModel(api_key="tts-key", base_url="https://api.example.com/v1")
        result = model.text_to_speech("Hello there", voice="nova")

        assert result == b"audio bytes output"
        mock_post.assert_called_once()

        call_kwargs = mock_post.call_args
        url = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1].get("url")
        assert url == "https://api.example.com/v1/audio/speech"

        headers = call_kwargs[1]["headers"]
        assert headers["Authorization"] == "Bearer tts-key"
        assert headers["Content-Type"] == "application/json"

        body = call_kwargs[1]["json"]
        assert body["model"] == "tts-1"
        assert body["input"] == "Hello there"
        assert body["voice"] == "nova"

    @patch("nerif.model.audio_model.httpx.post")
    def test_default_voice_is_alloy(self, mock_post):
        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        model = SpeechModel(api_key="k", base_url="https://api.example.com/v1")
        model.text_to_speech("test")

        body = mock_post.call_args[1]["json"]
        assert body["voice"] == "alloy"

    @patch("nerif.model.audio_model.httpx.post")
    def test_returns_bytes(self, mock_post):
        expected = b"\x00\x01\x02\x03audio data"
        mock_response = MagicMock()
        mock_response.content = expected
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        model = SpeechModel(api_key="k", base_url="https://api.example.com/v1")
        result = model.text_to_speech("text")

        assert isinstance(result, bytes)
        assert result == expected

    @patch("nerif.model.audio_model.httpx.post")
    def test_raises_on_http_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=MagicMock()
        )
        mock_post.return_value = mock_response

        model = SpeechModel(api_key="bad-key", base_url="https://api.example.com/v1")

        with pytest.raises(httpx.HTTPStatusError):
            model.text_to_speech("test")

    @patch("nerif.model.audio_model.httpx.post")
    def test_different_voices(self, mock_post):
        """Verify various voice options are passed through."""
        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        model = SpeechModel(api_key="k", base_url="https://api.example.com/v1")

        for voice in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
            model.text_to_speech("text", voice=voice)
            body = mock_post.call_args[1]["json"]
            assert body["voice"] == voice


# ===========================================================================
# Accessing AudioModel/SpeechModel via feature subpackages
# ===========================================================================
class TestAccessViaSubpackage:
    def test_audio_model_via_asr(self):
        from nerif.asr import AudioModel as AsrAudioModel

        model = AsrAudioModel(api_key="k", base_url="https://api.test.com/v1")
        assert model.api_key == "k"

    def test_speech_model_via_tts(self):
        from nerif.tts import SpeechModel as TtsSpeechModel

        model = TtsSpeechModel(api_key="k", base_url="https://api.test.com/v1")
        assert model.api_key == "k"
