"""Integration tests for cross-feature combinations in Nerif v1.2.0.

Tests feature interactions that are not covered by individual module tests.
All tests use mocks — no real API calls needed.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from nerif.exceptions import NerifError, ProviderError
from nerif.memory import ConversationMemory
from nerif.model.model import SimpleChatModel
from nerif.rag import NumpyVectorStore, SimpleRAG
from nerif.utils.token_counter import NerifTokenCounter

# ---------------------------------------------------------------------------
# Helpers to build mock responses
# ---------------------------------------------------------------------------


def _make_response(content="Hello!", model="gpt-4o", prompt_tokens=10, completion_tokens=5):
    """Create a mock ChatCompletionResponse-like object."""

    @dataclass
    class _Usage:
        prompt_tokens: int = 0
        completion_tokens: int = 0

    @dataclass
    class _Message:
        role: str = "assistant"
        content: Optional[str] = None
        tool_calls: Optional[list] = None

    @dataclass
    class _Choice:
        index: int = 0
        message: _Message = field(default_factory=_Message)
        finish_reason: str = "stop"

    @dataclass
    class _Response:
        model: str = ""
        choices: List[_Choice] = field(default_factory=list)
        usage: _Usage = field(default_factory=_Usage)

    return _Response(
        model=model,
        choices=[_Choice(message=_Message(content=content))],
        usage=_Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _make_tool_call_response(tool_name, arguments, tool_call_id="call_1", model="gpt-4o"):
    """Create a mock response with tool calls."""

    @dataclass
    class _FunctionCall:
        name: str = ""
        arguments: str = ""

    @dataclass
    class _ToolCall:
        id: str = ""
        type: str = "function"
        function: _FunctionCall = field(default_factory=_FunctionCall)

    @dataclass
    class _Usage:
        prompt_tokens: int = 10
        completion_tokens: int = 5

    @dataclass
    class _Message:
        role: str = "assistant"
        content: Optional[str] = None
        tool_calls: Optional[list] = None

    @dataclass
    class _Choice:
        index: int = 0
        message: _Message = field(default_factory=_Message)
        finish_reason: str = "tool_calls"

    @dataclass
    class _Response:
        model: str = ""
        choices: List[_Choice] = field(default_factory=list)
        usage: _Usage = field(default_factory=_Usage)

    tc = _ToolCall(id=tool_call_id, function=_FunctionCall(name=tool_name, arguments=arguments))
    return _Response(
        model=model,
        choices=[_Choice(message=_Message(content=None, tool_calls=[tc]))],
        usage=_Usage(prompt_tokens=15, completion_tokens=8),
    )


# ===========================================================================
# 1. Memory + TokenCounter
# ===========================================================================


class TestMemoryWithTokenCounter:
    """Verify that token counting works with memory-managed conversations."""

    @patch("nerif.utils.utils.retry_sync")
    def test_chat_with_memory_tracks_tokens(self, mock_retry):
        """Tokens should be tracked when using memory with chat."""
        mock_retry.return_value = _make_response("Hi!", prompt_tokens=20, completion_tokens=10)

        counter = NerifTokenCounter()
        memory = ConversationMemory(max_messages=10)
        model = SimpleChatModel(counter=counter, memory=memory)

        model.chat("Hello", append=True)

        # Counter should have recorded the request
        assert counter.total_requests == 1
        assert counter.successful_requests == 1
        assert "gpt-4o" in counter.model_token.model_cost
        assert counter.model_token["gpt-4o"].request == 20
        assert counter.model_token["gpt-4o"].response == 10

    @patch("nerif.utils.utils.retry_sync")
    def test_multi_turn_with_memory_accumulates_cost(self, mock_retry):
        """Multiple turns with memory should accumulate costs correctly."""
        mock_retry.side_effect = [
            _make_response("Reply 1", prompt_tokens=15, completion_tokens=8),
            _make_response("Reply 2", prompt_tokens=30, completion_tokens=12),
            _make_response("Reply 3", prompt_tokens=45, completion_tokens=15),
        ]

        counter = NerifTokenCounter()
        memory = ConversationMemory(max_messages=20)
        model = SimpleChatModel(counter=counter, memory=memory)

        model.chat("Turn 1", append=True)
        model.chat("Turn 2", append=True)
        model.chat("Turn 3", append=True)

        assert counter.total_requests == 3
        assert counter.model_token["gpt-4o"].request == 15 + 30 + 45
        assert counter.model_token["gpt-4o"].response == 8 + 12 + 15
        assert counter.total_cost() > 0

    @patch("nerif.memory.conversation.get_model_response")
    @patch("nerif.utils.utils.retry_sync")
    def test_summarization_tracks_tokens_when_counter_set(self, mock_retry, mock_summary):
        """When memory has a counter, summarization LLM calls should be tracked."""
        mock_retry.return_value = _make_response("Ok", prompt_tokens=10, completion_tokens=5)
        mock_summary.return_value = _make_response("Summary of conversation", model="gpt-4o-mini")

        counter = NerifTokenCounter()
        memory = ConversationMemory(max_messages=4, summarize=True, counter=counter)
        model = SimpleChatModel(counter=counter, memory=memory)

        # Add enough messages to trigger summarization
        for i in range(6):
            model.chat(f"Message {i}", append=True)

        # Summarization should have been called with counter
        assert mock_summary.called
        call_kwargs = mock_summary.call_args
        assert "counter" in call_kwargs.kwargs or (len(call_kwargs.args) > 1)


# ===========================================================================
# 2. Memory + RAG
# ===========================================================================


class TestMemoryWithRAG:
    """Verify memory and RAG work together for context-augmented conversations."""

    @patch("nerif.utils.utils.retry_sync")
    def test_rag_query_with_context_and_memory(self, mock_retry):
        """RAG query_with_context should work with a memory-backed model.

        Note: SimpleRAG.query_with_context calls model.chat() with default append=False,
        so after each call the model resets. This is correct behavior — RAG queries
        are standalone context lookups. For multi-turn RAG, use append=True explicitly.
        """
        mock_retry.return_value = _make_response(
            "Paris is the capital of France.", prompt_tokens=50, completion_tokens=10
        )

        memory = ConversationMemory(max_messages=20)
        model = SimpleChatModel(memory=memory)

        store = NumpyVectorStore()
        store.add(
            texts=["France is in Europe", "Paris is the capital of France"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )

        mock_embed = MagicMock()
        mock_embed.embed.return_value = [0.1, 0.9]

        rag = SimpleRAG(embed_model=mock_embed, store=store)

        result = rag.query_with_context("What is the capital of France?", model=model)

        assert result == "Paris is the capital of France."
        # After chat with append=False, model resets, so only system message remains
        assert len(memory._messages) == 1  # system only (reset after response)

    @patch("nerif.utils.utils.retry_sync")
    def test_rag_with_append_preserves_memory(self, mock_retry):
        """When using model.chat with append=True, RAG results accumulate in memory."""
        mock_retry.side_effect = [
            _make_response("Paris", prompt_tokens=30, completion_tokens=5),
            _make_response("Berlin", prompt_tokens=40, completion_tokens=5),
        ]

        memory = ConversationMemory(max_messages=20)
        model = SimpleChatModel(memory=memory)

        store = NumpyVectorStore()
        store.add(
            texts=["Paris is France's capital", "Berlin is Germany's capital"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )

        mock_embed = MagicMock()
        mock_embed.embed.side_effect = [[0.9, 0.1], [0.1, 0.9]]

        rag = SimpleRAG(embed_model=mock_embed, store=store)

        # Use model.chat directly with append=True to accumulate in memory
        results1 = rag.query("Capital of France?", top_k=1)
        context1 = "\n".join(r.text for r in results1)
        model.chat(f"Context: {context1}\nQuestion: Capital of France?", append=True)

        results2 = rag.query("Capital of Germany?", top_k=1)
        context2 = "\n".join(r.text for r in results2)
        model.chat(f"Context: {context2}\nQuestion: Capital of Germany?", append=True)

        non_system = [m for m in memory._messages if m["role"] != "system"]
        assert len(non_system) == 4  # 2 user + 2 assistant


# ===========================================================================
# 3. Memory + Exceptions
# ===========================================================================


class TestMemoryWithExceptions:
    def test_save_to_invalid_path_raises_error(self):
        """Saving to a non-existent directory should raise an appropriate error."""
        memory = ConversationMemory()
        memory.add_message("user", "hello")

        with pytest.raises(OSError):
            memory.save("/nonexistent/path/file.json")

    def test_load_from_nonexistent_file_raises_error(self):
        """Loading from a missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ConversationMemory.load("/nonexistent/file.json")

    def test_load_from_corrupted_json_raises_error(self):
        """Loading corrupted JSON should raise json.JSONDecodeError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json{{{")
            path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                ConversationMemory.load(path)
        finally:
            os.unlink(path)

    @patch("nerif.utils.utils.retry_sync")
    def test_provider_error_preserves_memory_state(self, mock_retry):
        """If an API call fails, memory should still contain the user message."""
        import httpx

        mock_retry.side_effect = httpx.HTTPStatusError(
            "rate limit",
            request=MagicMock(),
            response=MagicMock(status_code=429),
        )

        memory = ConversationMemory(max_messages=10)
        model = SimpleChatModel(memory=memory)

        with pytest.raises(ProviderError):
            model.chat("Hello", append=True)

        # The user message should be in memory (it was added before the API call)
        user_msgs = [m for m in memory._messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "Hello"


# ===========================================================================
# 4. Memory clear() reference integrity
# ===========================================================================


class TestMemoryClearReference:
    """Verify that memory.clear() doesn't break the model's messages reference."""

    def test_clear_preserves_reference(self):
        """After clear(), model.messages should still be the same list as memory._messages."""
        memory = ConversationMemory(max_messages=10)
        model = SimpleChatModel(memory=memory)

        # Add some messages directly
        memory.add_message("user", "test")
        assert len(model.messages) >= 2  # system + user

        # Clear via memory
        memory.clear()
        memory.add_message("system", "new prompt")

        # model.messages should reflect the cleared state
        assert model.messages is memory._messages
        assert len(model.messages) == 1

    @patch("nerif.utils.utils.retry_sync")
    def test_reset_preserves_reference(self, mock_retry):
        """After model.reset(), messages reference should be re-established."""
        mock_retry.return_value = _make_response("Hi")

        memory = ConversationMemory(max_messages=10)
        model = SimpleChatModel(memory=memory)

        model.chat("Hello", append=True)
        model.reset()

        # After reset, model.messages should still be memory._messages
        assert model.messages is memory._messages
        assert len(model.messages) == 1  # only system message


# ===========================================================================
# 5. Memory save/load + continued conversation
# ===========================================================================


class TestMemorySaveLoadContinuation:
    """Verify conversations can be saved, loaded, and continued."""

    @patch("nerif.utils.utils.retry_sync")
    def test_save_load_continue(self, mock_retry):
        """Save a conversation, load it into a new model, and continue."""
        mock_retry.side_effect = [
            _make_response("I'm fine!", prompt_tokens=15, completion_tokens=5),
            _make_response("I remember!", prompt_tokens=30, completion_tokens=5),
        ]

        memory1 = ConversationMemory(max_messages=20)
        model1 = SimpleChatModel(memory=memory1)
        model1.chat("How are you?", append=True)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            memory1.save(path)

            # Load into a new memory and model
            memory2 = ConversationMemory.load(path)
            model2 = SimpleChatModel(memory=memory2)

            # The loaded model should have the previous conversation
            user_msgs = [m for m in model2.messages if m.get("role") == "user"]
            assert any("How are you?" in str(m.get("content", "")) for m in user_msgs)

            # Continue the conversation
            model2.chat("Do you remember?", append=True)
            assert len(model2.messages) > len(memory1._messages)
        finally:
            os.unlink(path)

    def test_save_load_with_summary(self):
        """Summary should be preserved across save/load."""
        memory = ConversationMemory(max_messages=10)
        memory._summary = "We discussed Python programming."
        memory.add_message("user", "What about testing?")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            memory.save(path)
            loaded = ConversationMemory.load(path)

            assert loaded._summary == "We discussed Python programming."
            msgs = loaded.get_messages()
            assert any("Previous conversation summary" in str(m.get("content", "")) for m in msgs)
        finally:
            os.unlink(path)


# ===========================================================================
# 6. TokenCounter + RAG
# ===========================================================================


class TestTokenCounterWithRAG:
    """Verify cost tracking through RAG operations."""

    @patch("nerif.utils.utils.retry_sync")
    def test_rag_query_with_context_tracks_cost(self, mock_retry):
        """RAG's query_with_context should track tokens/cost when model has a counter."""
        mock_retry.return_value = _make_response("The answer", prompt_tokens=100, completion_tokens=20)

        counter = NerifTokenCounter()
        model = SimpleChatModel(counter=counter)

        store = NumpyVectorStore()
        store.add(texts=["doc1", "doc2"], embeddings=[[1.0, 0.0], [0.0, 1.0]])

        mock_embed = MagicMock()
        mock_embed.embed.return_value = [0.5, 0.5]

        rag = SimpleRAG(embed_model=mock_embed, store=store)
        rag.query_with_context("question", model=model)

        assert counter.total_requests == 1
        assert counter.total_cost() > 0
        assert counter.avg_latency() > 0


# ===========================================================================
# 7. Memory + Streaming
# ===========================================================================


class TestMemoryWithStreaming:
    """Verify streaming chat works correctly with memory."""

    @patch("nerif.utils.utils._openai_compatible_completion_stream")
    def test_stream_chat_with_memory(self, mock_stream):
        """Streaming should work with memory and accumulate messages."""
        from nerif.utils.utils import StreamChunk

        mock_stream.return_value = iter(
            [
                StreamChunk(content="Hello"),
                StreamChunk(content=" world"),
                StreamChunk(content="!", finish_reason="stop"),
            ]
        )

        memory = ConversationMemory(max_messages=10)
        model = SimpleChatModel(memory=memory)

        chunks = list(model.stream_chat("Hi there", append=True))

        assert "".join(chunks) == "Hello world!"
        # Memory should have user + assistant messages
        non_system = [m for m in memory._messages if m["role"] != "system"]
        assert len(non_system) == 2
        assert non_system[0]["role"] == "user"
        assert non_system[0]["content"] == "Hi there"
        assert non_system[1]["role"] == "assistant"
        assert non_system[1]["content"] == "Hello world!"


# ===========================================================================
# 8. Memory + Async
# ===========================================================================


class TestMemoryWithAsync:
    """Verify async chat works correctly with memory."""

    @patch("nerif.utils.utils.retry_async")
    @pytest.mark.asyncio(loop_scope="function")
    async def test_achat_with_memory(self, mock_retry):
        """Async chat should work with memory."""
        mock_retry.return_value = _make_response("Async reply!", prompt_tokens=10, completion_tokens=5)

        memory = ConversationMemory(max_messages=10)
        model = SimpleChatModel(memory=memory)

        result = await model.achat("Async question", append=True)

        assert result == "Async reply!"
        non_system = [m for m in memory._messages if m["role"] != "system"]
        assert len(non_system) == 2
        assert non_system[0]["content"] == "Async question"
        assert non_system[1]["content"] == "Async reply!"


# ===========================================================================
# 9. RAG save/load roundtrip with continued queries
# ===========================================================================


class TestRAGSaveLoadContinuation:
    """Verify RAG store can be saved, loaded, and continue serving queries."""

    def test_save_load_and_query(self, tmp_path):
        """Save a store, load it, add more docs, query all."""
        store = NumpyVectorStore()
        store.add(
            texts=["Python is great", "JavaScript is popular"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )

        path = str(tmp_path / "store.npz")
        store.save(path)

        loaded = NumpyVectorStore.load(path)
        loaded.add(
            texts=["Rust is fast"],
            embeddings=[[0.0, 0.0, 1.0]],
        )

        assert loaded.count() == 3
        results = loaded.search([0.9, 0.1, 0.0], top_k=1)
        assert results[0].text == "Python is great"


# ===========================================================================
# 10. Exception wrapping in provider calls
# ===========================================================================


class TestExceptionWrapping:
    """Verify provider errors are properly wrapped in custom exceptions."""

    @patch("nerif.utils.utils.retry_sync")
    def test_http_error_becomes_provider_error(self, mock_retry):
        """httpx.HTTPStatusError should be wrapped as ProviderError."""
        import httpx

        mock_retry.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        model = SimpleChatModel()
        with pytest.raises(ProviderError) as exc_info:
            model.chat("test")

        assert exc_info.value.status_code == 500
        # Should also be catchable as NerifError
        with pytest.raises(NerifError):
            mock_retry.side_effect = httpx.HTTPStatusError(
                "Bad Request",
                request=MagicMock(),
                response=MagicMock(status_code=400),
            )
            model.chat("test")

    def test_format_error_backward_compat(self):
        """FormatError should be catchable as ValueError."""
        from nerif.utils.format import FormatVerifierInt

        verifier = FormatVerifierInt()
        with pytest.raises(ValueError):
            verifier("no numbers here at all in this text")


# ===========================================================================
# 11. TokenCounter callbacks with memory
# ===========================================================================


class TestCallbacksWithMemory:
    """Verify callbacks fire correctly in memory-backed conversations."""

    @patch("nerif.utils.utils.retry_sync")
    def test_callbacks_fire_per_turn(self, mock_retry):
        """on_request_end should fire for each chat turn."""
        mock_retry.side_effect = [
            _make_response("R1", prompt_tokens=10, completion_tokens=5),
            _make_response("R2", prompt_tokens=20, completion_tokens=8),
        ]

        events = []
        counter = NerifTokenCounter()
        counter.on_request_end = lambda e: events.append(e)

        memory = ConversationMemory(max_messages=10)
        model = SimpleChatModel(counter=counter, memory=memory)

        model.chat("Turn 1", append=True)
        model.chat("Turn 2", append=True)

        assert len(events) == 2
        assert events[0].prompt_tokens == 10
        assert events[1].prompt_tokens == 20

    @patch("nerif.utils.utils.retry_sync")
    def test_error_callback_fires_on_failure(self, mock_retry):
        """on_error should fire when an API call fails."""
        import httpx

        mock_retry.side_effect = httpx.HTTPStatusError(
            "error",
            request=MagicMock(),
            response=MagicMock(status_code=429),
        )

        errors = []
        counter = NerifTokenCounter()
        counter.on_error = lambda e: errors.append(e)

        memory = ConversationMemory(max_messages=10)
        model = SimpleChatModel(counter=counter, memory=memory)

        with pytest.raises(ProviderError):
            model.chat("test", append=True)

        assert len(errors) == 1
        assert errors[0].latency_ms > 0


# ===========================================================================
# 12. Memory window management during multi-turn conversation
# ===========================================================================


class TestMemoryWindowIntegration:
    """Verify window management doesn't corrupt conversation flow."""

    @patch("nerif.utils.utils.retry_sync")
    def test_window_drops_oldest_preserves_recent(self, mock_retry):
        """Window management should keep most recent messages."""
        responses = [_make_response(f"Reply {i}") for i in range(8)]
        mock_retry.side_effect = responses

        memory = ConversationMemory(max_messages=4)
        model = SimpleChatModel(memory=memory)

        for i in range(8):
            model.chat(f"Message {i}", append=True)

        non_system = [m for m in memory._messages if m["role"] != "system"]
        assert len(non_system) <= 4
        # Most recent messages should be present
        contents = [m["content"] for m in non_system if m["role"] == "user"]
        assert "Message 7" in contents

    @patch("nerif.utils.utils.retry_sync")
    def test_model_messages_consistent_with_memory(self, mock_retry):
        """model.messages should always equal memory._messages."""
        responses = [_make_response(f"Reply {i}") for i in range(5)]
        mock_retry.side_effect = responses

        memory = ConversationMemory(max_messages=4)
        model = SimpleChatModel(memory=memory)

        for i in range(5):
            model.chat(f"Msg {i}", append=True)
            assert model.messages is memory._messages
