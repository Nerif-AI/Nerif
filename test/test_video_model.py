from unittest.mock import MagicMock, patch

from nerif.model.model import SimpleChatModel, VideoModel


def _make_mock_response(text="mock response"):
    """Create a mock ChatCompletionResponse with a text response."""
    message = MagicMock()
    message.content = text
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


class TestVideoModelInit:
    @patch("nerif.model.model.get_model_response", return_value=_make_mock_response())
    def test_creates_internal_simple_chat_model(self, mock_resp):
        vm = VideoModel(model="test-model", temperature=0.5)
        assert isinstance(vm._chat_model, SimpleChatModel)
        assert vm._chat_model.model == "test-model"
        assert vm._chat_model.temperature == 0.5


class TestAnalyzeUrl:
    @patch("nerif.model.model.get_model_response", return_value=_make_mock_response("video description"))
    def test_analyze_url_creates_multimodal_message(self, mock_resp):
        vm = VideoModel()
        result = vm.analyze_url("https://example.com/video.mp4", prompt="What is this?")

        assert result == "video description"
        mock_resp.assert_called_once()

        # Verify the message passed contains video_url and text parts
        call_args = mock_resp.call_args
        messages = call_args[0][0]
        user_msg = messages[-1]
        content = user_msg["content"]
        types = [part["type"] for part in content]
        assert "video_url" in types
        assert "text" in types


class TestAnalyzePath:
    @patch("nerif.model.model.get_model_response", return_value=_make_mock_response("path description"))
    @patch("builtins.open", create=True)
    def test_analyze_path_creates_multimodal_message(self, mock_open, mock_resp):
        import base64

        fake_bytes = b"fake video data"
        mock_open.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=fake_bytes)))
        mock_open.return_value.__exit__ = MagicMock(return_value=False)

        vm = VideoModel()
        result = vm.analyze_path("/tmp/test.mp4", prompt="Describe")

        assert result == "path description"
        mock_resp.assert_called_once()

        call_args = mock_resp.call_args
        messages = call_args[0][0]
        user_msg = messages[-1]
        content = user_msg["content"]
        types = [part["type"] for part in content]
        assert "video_url" in types
        assert "text" in types

        # Verify the video data is base64 encoded
        video_part = [p for p in content if p["type"] == "video_url"][0]
        expected_b64 = base64.b64encode(fake_bytes).decode("utf-8")
        assert expected_b64 in video_part["video_url"]["url"]


class TestReset:
    @patch("nerif.model.model.get_model_response", return_value=_make_mock_response())
    def test_reset_delegates_to_internal_model(self, mock_resp):
        vm = VideoModel()
        vm._chat_model.messages.append({"role": "user", "content": "test"})
        assert len(vm._chat_model.messages) > 1

        vm.reset()
        # After reset, only the system message remains
        assert len(vm._chat_model.messages) == 1
        assert vm._chat_model.messages[0]["role"] == "system"
