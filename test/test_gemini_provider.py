"""Tests for Gemini provider functions in nerif.utils.utils."""

import json
from unittest.mock import MagicMock, patch

from nerif.utils.utils import (
    ChatCompletionResponse,
    _convert_content_to_gemini_parts,
    _convert_tool_choice_to_gemini,
    _convert_tools_to_gemini,
    _gemini_completion,
)

# ---------------------------------------------------------------------------
# 1. Basic text completion
# ---------------------------------------------------------------------------


@patch("nerif.utils.utils.httpx.post")
def test_basic_text_completion(mock_post):
    """Mock httpx.post to return a Gemini-style response and verify ChatCompletionResponse."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello"}],
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    result = _gemini_completion(
        base_url="https://generativelanguage.googleapis.com",
        api_key="fake-key",
        model="gemini-pro",
        messages=[{"role": "user", "content": "Say hello"}],
    )

    assert isinstance(result, ChatCompletionResponse)
    assert len(result.choices) == 1
    assert result.choices[0].message.content == "Hello"
    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].finish_reason == "stop"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15
    assert result.model == "gemini-pro"

    # Verify httpx.post was called with the correct URL
    call_args = mock_post.call_args
    assert "gemini-pro:generateContent" in call_args[0][0] or "gemini-pro:generateContent" in call_args.kwargs.get("url", call_args[0][0])
    assert "key=fake-key" in call_args[0][0]


# ---------------------------------------------------------------------------
# 2. System instruction handling
# ---------------------------------------------------------------------------


@patch("nerif.utils.utils.httpx.post")
def test_system_instruction_handling(mock_post):
    """Verify system messages end up in body['systemInstruction']."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "OK"}]}}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    _gemini_completion(
        base_url="https://generativelanguage.googleapis.com",
        api_key="fake-key",
        model="gemini-pro",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ],
    )

    call_args = mock_post.call_args
    body = call_args.kwargs.get("json") or call_args[1]["json"]

    assert "systemInstruction" in body
    assert body["systemInstruction"] == {"parts": [{"text": "You are a helpful assistant."}]}
    # System message should NOT appear in contents
    for content_entry in body["contents"]:
        assert content_entry["role"] != "system"


# ---------------------------------------------------------------------------
# 3. Tool format conversion (_convert_tools_to_gemini)
# ---------------------------------------------------------------------------


def test_convert_tools_to_gemini():
    """Verify OpenAI tool format is converted to Gemini functionDeclarations format."""
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
            },
        },
    ]

    result = _convert_tools_to_gemini(openai_tools)

    assert len(result) == 2

    assert result[0]["name"] == "get_weather"
    assert result[0]["description"] == "Get the current weather"
    assert "parameters" in result[0]
    assert result[0]["parameters"]["type"] == "object"
    assert "location" in result[0]["parameters"]["properties"]

    assert result[1]["name"] == "search"
    assert result[1]["description"] == "Search the web"
    assert "parameters" not in result[1]


# ---------------------------------------------------------------------------
# 4. Tool choice conversion (_convert_tool_choice_to_gemini)
# ---------------------------------------------------------------------------


def test_tool_choice_auto():
    result = _convert_tool_choice_to_gemini("auto")
    assert result == {"functionCallingConfig": {"mode": "AUTO"}}


def test_tool_choice_none():
    result = _convert_tool_choice_to_gemini("none")
    assert result == {"functionCallingConfig": {"mode": "NONE"}}


def test_tool_choice_required():
    result = _convert_tool_choice_to_gemini("required")
    assert result == {"functionCallingConfig": {"mode": "ANY"}}


def test_tool_choice_specific_function():
    result = _convert_tool_choice_to_gemini(
        {"type": "function", "function": {"name": "get_weather"}}
    )
    assert result == {
        "functionCallingConfig": {
            "mode": "ANY",
            "allowedFunctionNames": ["get_weather"],
        }
    }


def test_tool_choice_none_value():
    """None (Python None) should default to AUTO."""
    result = _convert_tool_choice_to_gemini(None)
    assert result == {"functionCallingConfig": {"mode": "AUTO"}}


# ---------------------------------------------------------------------------
# 5. Function call response parsing
# ---------------------------------------------------------------------------


@patch("nerif.utils.utils.httpx.post")
def test_function_call_response_parsing(mock_post):
    """Mock a response with functionCall parts and verify _ToolCall parsing."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"location": "Paris"},
                            }
                        }
                    ],
                }
            }
        ],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4, "totalTokenCount": 12},
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    result = _gemini_completion(
        base_url="https://generativelanguage.googleapis.com",
        api_key="fake-key",
        model="gemini-pro",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                },
            }
        ],
    )

    assert result.choices[0].finish_reason == "tool_calls"
    assert result.choices[0].message.content is None
    tool_calls = result.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == {"location": "Paris"}
    assert tool_calls[0].type == "function"
    assert tool_calls[0].id == "get_weather"  # Gemini uses name as ID


# ---------------------------------------------------------------------------
# 6. Multi-modal content conversion (_convert_content_to_gemini_parts)
# ---------------------------------------------------------------------------


def test_convert_plain_string():
    result = _convert_content_to_gemini_parts("Hello world")
    assert result == [{"text": "Hello world"}]


def test_convert_base64_image():
    """Test base64 image conversion to inlineData format."""
    content = [
        {"type": "text", "text": "Describe this image"},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,iVBORw0KGgoAAAANS"
            },
        },
    ]

    result = _convert_content_to_gemini_parts(content)

    assert len(result) == 2
    assert result[0] == {"text": "Describe this image"}
    assert "inlineData" in result[1]
    assert result[1]["inlineData"]["mimeType"] == "image/png"
    assert result[1]["inlineData"]["data"] == "iVBORw0KGgoAAAANS"


def test_convert_http_image_url():
    """HTTP image URLs should be converted to a text placeholder (Gemini limitation)."""
    content = [
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/photo.jpg"},
        }
    ]

    result = _convert_content_to_gemini_parts(content)
    assert len(result) == 1
    assert "text" in result[0]
    assert "https://example.com/photo.jpg" in result[0]["text"]


# ---------------------------------------------------------------------------
# 7. Structured output (response_format with json_object)
# ---------------------------------------------------------------------------


@patch("nerif.utils.utils.httpx.post")
def test_structured_output_json_object(mock_post):
    """Verify response_format with 'json_object' adds responseMimeType to generationConfig."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": '{"answer": 42}'}]}}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    _gemini_completion(
        base_url="https://generativelanguage.googleapis.com",
        api_key="fake-key",
        model="gemini-pro",
        messages=[{"role": "user", "content": "Return JSON"}],
        response_format={"type": "json_object"},
    )

    call_args = mock_post.call_args
    body = call_args.kwargs.get("json") or call_args[1]["json"]

    assert "generationConfig" in body
    assert body["generationConfig"]["responseMimeType"] == "application/json"
