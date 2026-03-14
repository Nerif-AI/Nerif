"""Tests for streaming response support."""

import json
from unittest.mock import patch

import httpx

from nerif.utils.utils import (
    _anthropic_completion_stream,
    _gemini_completion_stream,
    _openai_compatible_completion_stream,
    _parse_sse_line,
    get_model_response_stream,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockStreamResponse:
    def __init__(self, lines, status_code=200):
        self.lines = lines
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def iter_lines(self):
        for line in self.lines:
            yield line

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Tests for _parse_sse_line
# ---------------------------------------------------------------------------


class TestParseSSELine:
    def test_valid_data_line(self):
        line = 'data: {"choices": [{"delta": {"content": "hello"}}]}'
        result = _parse_sse_line(line)
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "hello"

    def test_empty_line_returns_none(self):
        assert _parse_sse_line("") is None
        assert _parse_sse_line("   ") is None

    def test_done_sentinel_returns_none(self):
        assert _parse_sse_line("data: [DONE]") is None

    def test_invalid_json_returns_none(self):
        assert _parse_sse_line("data: not-json") is None
        assert _parse_sse_line("data: {bad json}") is None

    def test_non_data_line_returns_none(self):
        assert _parse_sse_line("event: message_start") is None
        assert _parse_sse_line(": comment line") is None

    def test_strips_whitespace(self):
        line = '  data: {"key": "value"}  '
        result = _parse_sse_line(line)
        assert result is not None
        assert result["key"] == "value"


# ---------------------------------------------------------------------------
# Tests for OpenAI-compatible streaming
# ---------------------------------------------------------------------------


class TestOpenAICompatibleStream:
    def _make_openai_lines(self, chunks):
        """Helper to build SSE lines for OpenAI-style chunks."""
        lines = []
        for chunk in chunks:
            lines.append(f"data: {json.dumps(chunk)}")
        lines.append("data: [DONE]")
        return lines

    def test_basic_content_chunks(self):
        lines = self._make_openai_lines(
            [
                {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": None},
            ]
        )
        mock_resp = MockStreamResponse(lines)

        with patch("httpx.stream", return_value=mock_resp):
            chunks = list(
                _openai_compatible_completion_stream(
                    base_url="https://api.openai.com/v1",
                    api_key="test-key",
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Hi"}],
                )
            )

        contents = [c.content for c in chunks if c.content]
        assert "Hello" in contents
        assert " world" in contents

    def test_usage_in_final_chunk(self):
        lines = self._make_openai_lines(
            [
                {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            ]
        )
        mock_resp = MockStreamResponse(lines)

        with patch("httpx.stream", return_value=mock_resp):
            chunks = list(
                _openai_compatible_completion_stream(
                    base_url="https://api.openai.com/v1",
                    api_key="test-key",
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Hi"}],
                )
            )

        usage_chunks = [c for c in chunks if c.usage is not None]
        assert len(usage_chunks) >= 1
        assert usage_chunks[-1].usage.prompt_tokens == 10
        assert usage_chunks[-1].usage.completion_tokens == 5
        assert usage_chunks[-1].usage.total_tokens == 15

    def test_empty_chunks_skipped(self):
        lines = self._make_openai_lines(
            [
                {"choices": [{"delta": {}, "finish_reason": None}]},  # no content
                {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
            ]
        )
        mock_resp = MockStreamResponse(lines)

        with patch("httpx.stream", return_value=mock_resp):
            chunks = list(
                _openai_compatible_completion_stream(
                    base_url="https://api.openai.com/v1",
                    api_key="test-key",
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Hi"}],
                )
            )

        # Only the chunk with content should be yielded
        assert any(c.content == "Hi" for c in chunks)

    def test_usage_only_chunk_without_choices(self):
        """Usage can appear in a chunk with no choices (stream_options usage chunk)."""
        lines = self._make_openai_lines(
            [
                {"choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
            ]
        )
        mock_resp = MockStreamResponse(lines)

        with patch("httpx.stream", return_value=mock_resp):
            chunks = list(
                _openai_compatible_completion_stream(
                    base_url="https://api.openai.com/v1",
                    api_key="test-key",
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Hi"}],
                )
            )

        usage_chunks = [c for c in chunks if c.usage is not None]
        assert len(usage_chunks) >= 1


# ---------------------------------------------------------------------------
# Tests for Anthropic streaming
# ---------------------------------------------------------------------------


class TestAnthropicStream:
    def _make_anthropic_lines(self, events):
        """Build SSE lines for Anthropic-style events."""
        lines = []
        for event_data in events:
            lines.append(f"data: {json.dumps(event_data)}")
        return lines

    def test_basic_content_stream(self):
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
            {"type": "content_block_start", "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " there"}},
            {"type": "content_block_stop"},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5},
            },
        ]
        lines = self._make_anthropic_lines(events)
        mock_resp = MockStreamResponse(lines)

        with patch("httpx.stream", return_value=mock_resp):
            chunks = list(
                _anthropic_completion_stream(
                    base_url="https://api.anthropic.com",
                    api_key="test-key",
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": "Hi"}],
                )
            )

        contents = [c.content for c in chunks if c.content]
        assert "Hello" in contents
        assert " there" in contents

    def test_usage_reported_in_message_delta(self):
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 20}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 8},
            },
        ]
        lines = self._make_anthropic_lines(events)
        mock_resp = MockStreamResponse(lines)

        with patch("httpx.stream", return_value=mock_resp):
            chunks = list(
                _anthropic_completion_stream(
                    base_url="https://api.anthropic.com",
                    api_key="test-key",
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": "Hi"}],
                )
            )

        usage_chunks = [c for c in chunks if c.usage is not None]
        assert len(usage_chunks) == 1
        assert usage_chunks[0].usage.prompt_tokens == 20
        assert usage_chunks[0].usage.completion_tokens == 8
        assert usage_chunks[0].usage.total_tokens == 28
        assert usage_chunks[0].finish_reason == "stop"

    def test_system_message_extracted(self):
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "OK"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        ]
        lines = self._make_anthropic_lines(events)
        mock_resp = MockStreamResponse(lines)

        captured_body = {}

        def fake_stream(method, url, json=None, headers=None, timeout=None):
            captured_body.update(json or {})
            return mock_resp

        with patch("httpx.stream", side_effect=fake_stream):
            chunks = list(
                _anthropic_completion_stream(
                    base_url="https://api.anthropic.com",
                    api_key="test-key",
                    model="claude-3-5-sonnet-20241022",
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hi"},
                    ],
                )
            )

        assert captured_body.get("system") == "You are helpful."
        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Tests for Gemini streaming
# ---------------------------------------------------------------------------


class TestGeminiStream:
    def _make_gemini_lines(self, responses):
        """Build SSE lines for Gemini-style responses."""
        lines = []
        for resp_data in responses:
            lines.append(f"data: {json.dumps(resp_data)}")
        return lines

    def test_basic_content_stream(self):
        responses = [
            {"candidates": [{"content": {"parts": [{"text": "Hello"}]}, "finishReason": None}]},
            {
                "candidates": [{"content": {"parts": [{"text": " world"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
            },
        ]
        lines = self._make_gemini_lines(responses)
        mock_resp = MockStreamResponse(lines)

        with patch("httpx.stream", return_value=mock_resp):
            chunks = list(
                _gemini_completion_stream(
                    base_url="https://generativelanguage.googleapis.com",
                    api_key="test-key",
                    model="gemini-2.0-flash",
                    messages=[{"role": "user", "content": "Hi"}],
                )
            )

        contents = [c.content for c in chunks if c.content]
        assert "Hello" in contents
        assert " world" in contents

    def test_stop_reason_normalized(self):
        responses = [
            {
                "candidates": [{"content": {"parts": [{"text": "Done"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
            }
        ]
        lines = self._make_gemini_lines(responses)
        mock_resp = MockStreamResponse(lines)

        with patch("httpx.stream", return_value=mock_resp):
            chunks = list(
                _gemini_completion_stream(
                    base_url="https://generativelanguage.googleapis.com",
                    api_key="test-key",
                    model="gemini-2.0-flash",
                    messages=[{"role": "user", "content": "Hi"}],
                )
            )

        stop_chunks = [c for c in chunks if c.finish_reason == "stop"]
        assert len(stop_chunks) >= 1

    def test_usage_reported(self):
        responses = [
            {
                "candidates": [{"content": {"parts": [{"text": "Hi"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4, "totalTokenCount": 14},
            }
        ]
        lines = self._make_gemini_lines(responses)
        mock_resp = MockStreamResponse(lines)

        with patch("httpx.stream", return_value=mock_resp):
            chunks = list(
                _gemini_completion_stream(
                    base_url="https://generativelanguage.googleapis.com",
                    api_key="test-key",
                    model="gemini-2.0-flash",
                    messages=[{"role": "user", "content": "Hi"}],
                )
            )

        usage_chunks = [c for c in chunks if c.usage is not None]
        assert len(usage_chunks) >= 1
        assert usage_chunks[0].usage.prompt_tokens == 10
        assert usage_chunks[0].usage.completion_tokens == 4
        assert usage_chunks[0].usage.total_tokens == 14


# ---------------------------------------------------------------------------
# Tests for get_model_response_stream routing
# ---------------------------------------------------------------------------


class TestGetModelResponseStreamRouting:
    def test_routes_to_openai_for_default(self):
        mock_resp = MockStreamResponse(
            ['data: {"choices": [{"delta": {"content": "Hi"}, "finish_reason": null}]}', "data: [DONE]"]
        )
        with patch("httpx.stream", return_value=mock_resp) as mock_stream:
            list(
                get_model_response_stream(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gpt-4o",
                    api_key="test-key",
                )
            )
        # Should have called httpx.stream with an OpenAI URL
        call_args = mock_stream.call_args
        assert "/chat/completions" in call_args[0][1]

    def test_routes_to_anthropic(self):
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        ]
        lines = [f"data: {json.dumps(e)}" for e in events]
        mock_resp = MockStreamResponse(lines)

        with patch("httpx.stream", return_value=mock_resp) as mock_stream:
            list(
                get_model_response_stream(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="anthropic/claude-3-5-sonnet-20241022",
                    api_key="test-key",
                )
            )
        call_args = mock_stream.call_args
        assert "/v1/messages" in call_args[0][1]

    def test_routes_to_gemini(self):
        responses = [
            {
                "candidates": [{"content": {"parts": [{"text": "Hi"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
            }
        ]
        lines = [f"data: {json.dumps(r)}" for r in responses]
        mock_resp = MockStreamResponse(lines)

        with patch("httpx.stream", return_value=mock_resp) as mock_stream:
            list(
                get_model_response_stream(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gemini/gemini-2.0-flash",
                    api_key="test-key",
                )
            )
        call_args = mock_stream.call_args
        assert "streamGenerateContent" in call_args[0][1]

    def test_yields_stream_chunk_objects(self):
        mock_resp = MockStreamResponse(
            ['data: {"choices": [{"delta": {"content": "Hi"}, "finish_reason": null}]}', "data: [DONE]"]
        )
        with patch("nerif.utils.utils.httpx.stream", return_value=mock_resp):
            chunks = list(
                get_model_response_stream(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gpt-4o",
                    api_key="test-key",
                )
            )
        assert len(chunks) > 0
        assert all(type(c).__name__ == "StreamChunk" for c in chunks)
        assert chunks[0].content == "Hi"


# ---------------------------------------------------------------------------
# Tests for SimpleChatModel.stream_chat
# ---------------------------------------------------------------------------


class TestSimpleChatModelStreamChat:
    def test_yields_text_strings(self):
        from nerif.model.model import SimpleChatModel

        mock_resp = MockStreamResponse(
            [
                'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}',
                'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": null}]}',
                'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
                "data: [DONE]",
            ]
        )

        model = SimpleChatModel(model="gpt-4o")
        with patch("httpx.stream", return_value=mock_resp):
            result = list(model.stream_chat("Say hello"))

        assert "Hello" in result
        assert " world" in result

    def test_full_response_assembled_correctly(self):
        from nerif.model.model import SimpleChatModel

        mock_resp = MockStreamResponse(
            [
                'data: {"choices": [{"delta": {"content": "Part1"}, "finish_reason": null}]}',
                'data: {"choices": [{"delta": {"content": "Part2"}, "finish_reason": null}]}',
                'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
                "data: [DONE]",
            ]
        )

        model = SimpleChatModel(model="gpt-4o")
        with patch("httpx.stream", return_value=mock_resp):
            chunks = list(model.stream_chat("Hi"))

        assert "".join(chunks) == "Part1Part2"

    def test_history_reset_when_append_false(self):
        from nerif.model.model import SimpleChatModel

        mock_resp = MockStreamResponse(
            [
                'data: {"choices": [{"delta": {"content": "Hi"}, "finish_reason": null}]}',
                'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
                "data: [DONE]",
            ]
        )

        model = SimpleChatModel(model="gpt-4o")
        initial_len = len(model.messages)

        with patch("httpx.stream", return_value=mock_resp):
            list(model.stream_chat("Hello", append=False))

        # After reset, should be back to initial (system message only)
        assert len(model.messages) == initial_len

    def test_history_kept_when_append_true(self):
        from nerif.model.model import SimpleChatModel

        mock_resp = MockStreamResponse(
            [
                'data: {"choices": [{"delta": {"content": "I am fine"}, "finish_reason": null}]}',
                'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
                "data: [DONE]",
            ]
        )

        model = SimpleChatModel(model="gpt-4o")
        with patch("httpx.stream", return_value=mock_resp):
            list(model.stream_chat("How are you?", append=True))

        # Should have: system + user + assistant
        assert len(model.messages) == 3
        assert model.messages[-1]["role"] == "assistant"
        assert model.messages[-1]["content"] == "I am fine"

    def test_empty_chunks_not_yielded(self):
        from nerif.model.model import SimpleChatModel

        mock_resp = MockStreamResponse(
            [
                'data: {"choices": [{"delta": {}, "finish_reason": null}]}',  # empty
                'data: {"choices": [{"delta": {"content": "Hi"}, "finish_reason": null}]}',
                'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
                "data: [DONE]",
            ]
        )

        model = SimpleChatModel(model="gpt-4o")
        with patch("httpx.stream", return_value=mock_resp):
            chunks = list(model.stream_chat("Hello"))

        # Only non-empty content should be yielded
        assert all(c != "" for c in chunks)
        assert "Hi" in chunks

    def test_multimodal_message_supported(self):
        from nerif.model.model import MultiModalMessage, SimpleChatModel

        mock_resp = MockStreamResponse(
            [
                'data: {"choices": [{"delta": {"content": "I see an image"}, "finish_reason": null}]}',
                'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
                "data: [DONE]",
            ]
        )

        model = SimpleChatModel(model="gpt-4o")
        msg = MultiModalMessage().add_text("Describe this.")
        with patch("httpx.stream", return_value=mock_resp):
            chunks = list(model.stream_chat(msg))

        assert "I see an image" in chunks
