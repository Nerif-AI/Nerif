"""Tests for v1.3.1 fixes."""

import asyncio
import json as _json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from nerif.exceptions import FormatError
from nerif.memory.conversation import ConversationMemory
from nerif.model.model import SimpleChatModel
from nerif.utils.callbacks import CallbackHandler, CallbackManager
from nerif.utils.format import FormatVerifierStringList
from nerif.utils.rate_limit import RateLimitConfig, RateLimiter


class TestFormatVerifierStringListSecurity:
    """Test that FormatVerifierStringList uses safe parsing."""

    def test_convert_valid_list(self):
        v = FormatVerifierStringList()
        result = v.convert('["hello", "world"]')
        assert result == ["hello", "world"]

    def test_convert_invalid_input_raises(self):
        v = FormatVerifierStringList()
        with pytest.raises(FormatError):
            v.convert("not a list at all")

    def test_convert_rejects_code_injection(self):
        """eval() would execute this; ast.literal_eval must reject it."""
        v = FormatVerifierStringList()
        with pytest.raises(FormatError):
            v.convert("__import__('os').system('echo pwned')")

    def test_convert_single_element(self):
        v = FormatVerifierStringList()
        result = v.convert('["single"]')
        assert result == ["single"]


class TestRateLimiterAsyncSafety:
    """Test that async lock initialization is safe under concurrency."""

    def test_concurrent_aacquire_no_race(self):
        # No max_concurrent so aacquire() returns immediately after the
        # interval check; no semaphore deadlock between gather and arelease.
        config = RateLimitConfig(requests_per_second=100, max_concurrent=0)
        limiter = RateLimiter(config)

        async def run():
            tasks = [limiter.aacquire() for _ in range(10)]
            await asyncio.gather(*tasks)

        asyncio.run(run())
        # If no exception, the async primitives initialised without a race


class TestCallbacksWired:
    """Test that callbacks actually fire during chat()."""

    @patch("nerif.model.model.get_model_response")
    def test_callbacks_fire_on_chat(self, mock_response):
        mock_choice = MagicMock()
        mock_choice.message.content = "hello"
        mock_choice.message.tool_calls = None
        mock_response.return_value = MagicMock(choices=[mock_choice])

        handler = MagicMock(spec=CallbackHandler)
        cb = CallbackManager()
        cb.add_handler(handler)

        model = SimpleChatModel(callbacks=cb)
        model.chat("test")

        handler.on_llm_start.assert_called_once()
        handler.on_llm_end.assert_called_once()

    @patch("nerif.model.model.get_model_response")
    def test_callbacks_fire_on_error(self, mock_response):
        mock_response.side_effect = RuntimeError("api error")

        handler = MagicMock(spec=CallbackHandler)
        cb = CallbackManager()
        cb.add_handler(handler)

        model = SimpleChatModel(callbacks=cb)
        with pytest.raises(RuntimeError):
            model.chat("test")

        handler.on_llm_error.assert_called_once()


class TestRateLimiterWired:
    """Test that rate limiter is consulted during chat()."""

    @patch("nerif.model.model.get_model_response")
    def test_rate_limiter_acquire_release(self, mock_response):
        mock_choice = MagicMock()
        mock_choice.message.content = "hello"
        mock_choice.message.tool_calls = None
        mock_response.return_value = MagicMock(choices=[mock_choice])

        config = RateLimitConfig(requests_per_second=10)
        limiter = RateLimiter(config)
        limiter.acquire = MagicMock()
        limiter.release = MagicMock()

        model = SimpleChatModel(rate_limiter=limiter)
        model.chat("test")

        limiter.acquire.assert_called_once()
        limiter.release.assert_called_once()


class TestMemoryPersistConfig:
    """Test that Memory save/load preserves configuration."""

    def test_save_load_preserves_config(self):
        mem = ConversationMemory(
            max_messages=10,
            max_tokens=500,
            summarize=True,
            summarize_model="gpt-4o-mini",
        )
        mem.add_message("user", "hello")
        mem.add_message("assistant", "hi there")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            mem.save(path)
            loaded = ConversationMemory.load(path)
            assert loaded.max_messages == 10
            assert loaded.max_tokens == 500
            assert loaded.summarize is True
            assert loaded.summarize_model == "gpt-4o-mini"
            assert len(loaded._messages) == 2
        finally:
            os.unlink(path)

    def test_load_v1_format_backward_compat(self):
        """Old format without config should load with defaults."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            _json.dump(
                {
                    "version": "1.0",
                    "messages": [{"role": "user", "content": "hi"}],
                    "metadata": {},
                    "summary": None,
                },
                f,
            )
            path = f.name
        try:
            loaded = ConversationMemory.load(path)
            assert loaded.max_messages is None
            assert loaded.max_tokens is None
            assert loaded.summarize is False
            assert len(loaded._messages) == 1
        finally:
            os.unlink(path)
