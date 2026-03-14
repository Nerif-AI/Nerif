"""Tests for Anthropic provider functions in nerif.utils.utils."""

import json
from unittest.mock import MagicMock, patch

from nerif.utils.utils import (
    ChatCompletionResponse,
    _anthropic_completion,
    _convert_content_to_anthropic,
    _convert_tool_choice_to_anthropic,
    _convert_tools_to_anthropic,
)


def _mock_httpx_response(data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response with the given JSON data."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = data
    mock_resp.raise_for_status.return_value = None
    return mock_resp


# --------------------------------------------------------------------------
# 1. Basic text completion
# --------------------------------------------------------------------------


@patch("nerif.utils.utils.httpx.post")
def test_basic_text_completion(mock_post):
    """Mock httpx.post to return a simple Anthropic text response and verify
    the ChatCompletionResponse is correctly populated."""
    anthropic_response = {
        "id": "msg_123",
        "model": "claude-3-sonnet-20240229",
        "content": [{"type": "text", "text": "Hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    mock_post.return_value = _mock_httpx_response(anthropic_response)

    result = _anthropic_completion(
        base_url="https://api.anthropic.com",
        api_key="test-key",
        model="claude-3-sonnet-20240229",
        messages=[{"role": "user", "content": "Say hello"}],
    )

    assert isinstance(result, ChatCompletionResponse)
    assert result.id == "msg_123"
    assert result.model == "claude-3-sonnet-20240229"
    assert len(result.choices) == 1
    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].message.content == "Hello"
    assert result.choices[0].finish_reason == "stop"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15


# --------------------------------------------------------------------------
# 2. System message extraction
# --------------------------------------------------------------------------


@patch("nerif.utils.utils.httpx.post")
def test_system_message_extraction(mock_post):
    """Verify that a system message is separated from conversation messages
    and placed in body['system']."""
    anthropic_response = {
        "id": "msg_sys",
        "model": "claude-3-sonnet-20240229",
        "content": [{"type": "text", "text": "I am helpful"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 20, "output_tokens": 3},
    }
    mock_post.return_value = _mock_httpx_response(anthropic_response)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi"},
    ]
    _anthropic_completion(
        base_url="https://api.anthropic.com",
        api_key="test-key",
        model="claude-3-sonnet-20240229",
        messages=messages,
    )

    # Inspect the JSON body sent to httpx.post
    call_kwargs = mock_post.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

    # System message should be in body["system"], not in body["messages"]
    assert body["system"] == "You are a helpful assistant."
    for msg in body["messages"]:
        assert msg["role"] != "system"


# --------------------------------------------------------------------------
# 3. Tool format conversion
# --------------------------------------------------------------------------


def test_convert_tools_to_anthropic():
    """Verify OpenAI tool format is converted to Anthropic format
    with input_schema instead of parameters."""
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    result = _convert_tools_to_anthropic(openai_tools)

    assert len(result) == 1
    tool = result[0]
    assert tool["name"] == "get_weather"
    assert tool["description"] == "Get the weather for a location"
    assert "input_schema" in tool
    assert "parameters" not in tool
    assert tool["input_schema"]["type"] == "object"
    assert "location" in tool["input_schema"]["properties"]


# --------------------------------------------------------------------------
# 4. Tool choice conversion
# --------------------------------------------------------------------------


def test_convert_tool_choice_auto():
    assert _convert_tool_choice_to_anthropic("auto") == {"type": "auto"}


def test_convert_tool_choice_required():
    assert _convert_tool_choice_to_anthropic("required") == {"type": "any"}


def test_convert_tool_choice_specific_function():
    openai_choice = {"type": "function", "function": {"name": "get_weather"}}
    result = _convert_tool_choice_to_anthropic(openai_choice)
    assert result == {"type": "tool", "name": "get_weather"}


def test_convert_tool_choice_none_defaults_to_auto():
    assert _convert_tool_choice_to_anthropic(None) == {"type": "auto"}


# --------------------------------------------------------------------------
# 5. Tool use response parsing
# --------------------------------------------------------------------------


@patch("nerif.utils.utils.httpx.post")
def test_tool_use_response_parsing(mock_post):
    """Mock a response with tool_use content blocks and verify they are
    parsed into _ToolCall objects with correct function names and arguments."""
    anthropic_response = {
        "id": "msg_tool",
        "model": "claude-3-sonnet-20240229",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_abc123",
                "name": "get_weather",
                "input": {"location": "San Francisco"},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 15, "output_tokens": 20},
    }
    mock_post.return_value = _mock_httpx_response(anthropic_response)

    result = _anthropic_completion(
        base_url="https://api.anthropic.com",
        api_key="test-key",
        model="claude-3-sonnet-20240229",
        messages=[{"role": "user", "content": "What is the weather in SF?"}],
    )

    assert result.choices[0].finish_reason == "tool_calls"
    tool_calls = result.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "toolu_abc123"
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "get_weather"
    parsed_args = json.loads(tool_calls[0].function.arguments)
    assert parsed_args == {"location": "San Francisco"}
    # Content should be None when only tool_use blocks are present
    assert result.choices[0].message.content is None


# --------------------------------------------------------------------------
# 6. Multi-modal content conversion
# --------------------------------------------------------------------------


def test_convert_content_plain_text():
    """Plain text strings should pass through unchanged."""
    assert _convert_content_to_anthropic("hello") == "hello"


def test_convert_content_base64_image():
    """image_url with a base64 data URI should be converted to Anthropic
    image source format."""
    content = [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
        }
    ]
    result = _convert_content_to_anthropic(content)

    assert len(result) == 1
    assert result[0]["type"] == "image"
    assert result[0]["source"]["type"] == "base64"
    assert result[0]["source"]["media_type"] == "image/png"
    assert result[0]["source"]["data"] == "iVBORw0KGgo="


def test_convert_content_text_part():
    """Text parts in a content array should be preserved."""
    content = [{"type": "text", "text": "Describe this image"}]
    result = _convert_content_to_anthropic(content)

    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "Describe this image"


def test_convert_content_url_image():
    """image_url with a regular URL should produce a URL source."""
    content = [
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
        }
    ]
    result = _convert_content_to_anthropic(content)

    assert len(result) == 1
    assert result[0]["type"] == "image"
    assert result[0]["source"]["type"] == "url"
    assert result[0]["source"]["url"] == "https://example.com/image.png"


# --------------------------------------------------------------------------
# 7. Tool result message conversion
# --------------------------------------------------------------------------


@patch("nerif.utils.utils.httpx.post")
def test_tool_result_message_conversion(mock_post):
    """Verify role='tool' messages are converted to Anthropic tool_result format
    in the request body."""
    anthropic_response = {
        "id": "msg_result",
        "model": "claude-3-sonnet-20240229",
        "content": [{"type": "text", "text": "The weather is sunny."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 30, "output_tokens": 10},
    }
    mock_post.return_value = _mock_httpx_response(anthropic_response)

    messages = [
        {"role": "user", "content": "What is the weather in SF?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_abc123",
            "content": '{"temperature": 72, "condition": "sunny"}',
        },
    ]

    _anthropic_completion(
        base_url="https://api.anthropic.com",
        api_key="test-key",
        model="claude-3-sonnet-20240229",
        messages=messages,
    )

    call_kwargs = mock_post.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

    # Find the tool result message in the converted conversation
    tool_result_msg = None
    for msg in body["messages"]:
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if block.get("type") == "tool_result":
                    tool_result_msg = block
                    break

    assert tool_result_msg is not None
    assert tool_result_msg["type"] == "tool_result"
    assert tool_result_msg["tool_use_id"] == "toolu_abc123"
    assert tool_result_msg["content"] == '{"temperature": 72, "condition": "sunny"}'
