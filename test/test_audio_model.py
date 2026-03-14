import os
from unittest.mock import MagicMock, mock_open, patch

from nerif.model.audio_model import AudioModel, SpeechModel


class TestAudioModelInit:
    def test_defaults_from_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_API_BASE": "https://custom.api/v1"}):
            # Re-import to pick up patched env vars
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


class TestSpeechModelTextToSpeech:
    @patch("nerif.model.audio_model.httpx.post")
    def test_text_to_speech_sends_correct_request(self, mock_post):
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
