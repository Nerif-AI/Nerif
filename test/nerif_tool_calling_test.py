"""Tests for tool calling and structured output features."""



from nerif.model.model import MultiModalMessage, ToolCallResult, ToolDefinition
from nerif.utils.format import FormatVerifierJson, NerifFormat


class TestToolDefinition:
    def test_to_dict(self):
        td = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        result = td.to_dict()
        assert result["type"] == "function"
        assert result["function"]["name"] == "test_tool"
        assert result["function"]["description"] == "A test tool"

    def test_tool_call_result_repr(self):
        tc = ToolCallResult(id="call_1", name="foo", arguments='{"x": 1}')
        assert "foo" in repr(tc)
        assert "call_1" in repr(tc)


class TestMultiModalMessage:
    def test_add_text(self):
        msg = MultiModalMessage()
        msg.add_text("hello")
        content = msg.to_content()
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "hello"

    def test_add_image_url(self):
        msg = MultiModalMessage()
        msg.add_image_url("https://example.com/img.png")
        content = msg.to_content()
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == "https://example.com/img.png"

    def test_add_image_base64(self):
        msg = MultiModalMessage()
        msg.add_image_base64("abc123")
        content = msg.to_content()
        assert "base64" in content[0]["image_url"]["url"]

    def test_chaining(self):
        msg = MultiModalMessage().add_text("describe").add_image_url("https://img.com/x.png")
        assert len(msg.to_content()) == 2

    def test_add_video_url(self):
        msg = MultiModalMessage()
        msg.add_video_url("https://example.com/video.mp4")
        content = msg.to_content()
        assert content[0]["type"] == "video_url"

    def test_add_audio_base64(self):
        msg = MultiModalMessage()
        msg.add_audio_base64("abc123", format="mp3")
        content = msg.to_content()
        assert content[0]["type"] == "input_audio"
        assert content[0]["input_audio"]["format"] == "mp3"


class TestFormatVerifierJson:
    def test_verify_valid_json(self):
        v = FormatVerifierJson()
        assert v.verify('{"key": "value"}')

    def test_verify_invalid_json(self):
        v = FormatVerifierJson()
        assert not v.verify("not json")

    def test_match_json_in_text(self):
        v = FormatVerifierJson()
        result = v.match('Here is the result: {"name": "Alice", "age": 30}. Done.')
        assert result == {"name": "Alice", "age": 30}

    def test_match_json_array(self):
        v = FormatVerifierJson()
        result = v.match('Result: [1, 2, 3]')
        assert result == [1, 2, 3]

    def test_convert(self):
        v = FormatVerifierJson()
        result = v.convert('{"a": 1}')
        assert result == {"a": 1}


class TestNerifFormatJsonParse:
    def test_parse_plain_json(self):
        result = NerifFormat.json_parse('{"x": 1}')
        assert result == {"x": 1}

    def test_parse_markdown_code_block(self):
        result = NerifFormat.json_parse('```json\n{"x": 1}\n```')
        assert result == {"x": 1}

    def test_parse_json_with_surrounding_text(self):
        result = NerifFormat.json_parse('Sure! Here is the JSON:\n{"x": 1}\nHope that helps!')
        assert result == {"x": 1}
