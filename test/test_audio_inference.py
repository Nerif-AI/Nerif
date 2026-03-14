"""Tests for AudioInferenceModel (audio multimodal reasoning via SimpleChatModel)."""

import base64
from unittest.mock import patch

from nerif.asr import AudioInferenceModel


class TestAudioInferenceModelInit:
    def test_creates_with_required_model_param(self):
        m = AudioInferenceModel(model="gpt-4o-audio-preview")
        assert m._chat_model is not None

    def test_custom_default_prompt(self):
        m = AudioInferenceModel(model="gpt-4o-audio-preview", default_prompt="Custom prompt")
        assert m._chat_model.default_prompt == "Custom prompt"

    def test_custom_temperature(self):
        m = AudioInferenceModel(model="gpt-4o-audio-preview", temperature=0.7)
        assert m._chat_model.temperature == 0.7

    def test_default_temperature_is_zero(self):
        m = AudioInferenceModel(model="gpt-4o-audio-preview")
        assert m._chat_model.temperature == 0.0


class TestAudioInferenceModelAnalyzePath:
    @patch("nerif.model.audio_inference.SimpleChatModel.chat", return_value="transcribed and analyzed")
    def test_builds_audio_multimodal_message(self, mock_chat, tmp_path):
        audio_path = tmp_path / "sample.wav"
        audio_path.write_bytes(b"fake audio bytes")

        model = AudioInferenceModel(model="gpt-4o-audio-preview")
        result = model.analyze_path(str(audio_path), prompt="Summarize this audio", format="wav")

        assert result == "transcribed and analyzed"
        message = mock_chat.call_args[0][0]
        content = message.to_content()
        assert content[0]["type"] == "input_audio"
        assert content[0]["input_audio"]["format"] == "wav"
        expected_b64 = base64.b64encode(b"fake audio bytes").decode()
        assert content[0]["input_audio"]["data"] == expected_b64
        assert content[1] == {"type": "text", "text": "Summarize this audio"}

    @patch("nerif.model.audio_inference.SimpleChatModel.chat", return_value="ok")
    def test_default_prompt(self, mock_chat, tmp_path):
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"data")

        model = AudioInferenceModel(model="gpt-4o-audio-preview")
        model.analyze_path(str(audio_path))

        message = mock_chat.call_args[0][0]
        content = message.to_content()
        assert content[1]["text"] == "Transcribe and analyze this audio."

    @patch("nerif.model.audio_inference.SimpleChatModel.chat", return_value="ok")
    def test_custom_format(self, mock_chat, tmp_path):
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"mp3 data")

        model = AudioInferenceModel(model="gpt-4o-audio-preview")
        model.analyze_path(str(audio_path), format="mp3")

        message = mock_chat.call_args[0][0]
        content = message.to_content()
        assert content[0]["input_audio"]["format"] == "mp3"

    @patch("nerif.model.audio_inference.SimpleChatModel.chat", return_value="ok")
    def test_max_tokens_passed_through(self, mock_chat, tmp_path):
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"data")

        model = AudioInferenceModel(model="gpt-4o-audio-preview")
        model.analyze_path(str(audio_path), max_tokens=100)

        assert mock_chat.call_args.kwargs["max_tokens"] == 100


class TestAudioInferenceModelAnalyzeUrl:
    @patch("nerif.model.audio_inference.SimpleChatModel.chat", return_value="url result")
    def test_builds_url_message(self, mock_chat):
        model = AudioInferenceModel(model="gpt-4o-audio-preview")
        result = model.analyze_url("https://example.com/audio.wav", prompt="Describe")

        assert result == "url result"
        message = mock_chat.call_args[0][0]
        content = message.to_content()
        assert content[0]["type"] == "input_audio"
        assert content[0]["input_audio"]["url"] == "https://example.com/audio.wav"
        assert content[1] == {"type": "text", "text": "Describe"}

    @patch("nerif.model.audio_inference.SimpleChatModel.chat", return_value="ok")
    def test_default_prompt(self, mock_chat):
        model = AudioInferenceModel(model="gpt-4o-audio-preview")
        model.analyze_url("https://example.com/audio.wav")

        message = mock_chat.call_args[0][0]
        content = message.to_content()
        assert content[1]["text"] == "Describe this audio."

    @patch("nerif.model.audio_inference.SimpleChatModel.chat", return_value="ok")
    def test_max_tokens_passed_through(self, mock_chat):
        model = AudioInferenceModel(model="gpt-4o-audio-preview")
        model.analyze_url("https://example.com/audio.wav", max_tokens=50)

        assert mock_chat.call_args.kwargs["max_tokens"] == 50


class TestAudioInferenceModelAnalyzeBase64:
    @patch("nerif.model.audio_inference.SimpleChatModel.chat", return_value="b64 result")
    def test_builds_base64_message(self, mock_chat):
        b64_data = base64.b64encode(b"audio content").decode()
        model = AudioInferenceModel(model="gpt-4o-audio-preview")
        result = model.analyze_base64(b64_data, prompt="Analyze", format="wav")

        assert result == "b64 result"
        message = mock_chat.call_args[0][0]
        content = message.to_content()
        assert content[0]["type"] == "input_audio"
        assert content[0]["input_audio"]["data"] == b64_data
        assert content[0]["input_audio"]["format"] == "wav"
        assert content[1] == {"type": "text", "text": "Analyze"}

    @patch("nerif.model.audio_inference.SimpleChatModel.chat", return_value="ok")
    def test_default_prompt_and_format(self, mock_chat):
        model = AudioInferenceModel(model="gpt-4o-audio-preview")
        model.analyze_base64("YWJj")

        message = mock_chat.call_args[0][0]
        content = message.to_content()
        assert content[0]["input_audio"]["format"] == "wav"
        assert content[1]["text"] == "Transcribe and analyze this audio."


class TestAudioInferenceModelReset:
    @patch("nerif.model.audio_inference.SimpleChatModel.reset")
    def test_reset_delegates_to_chat_model(self, mock_reset):
        model = AudioInferenceModel(model="gpt-4o-audio-preview")
        model.reset()
        mock_reset.assert_called_once()
