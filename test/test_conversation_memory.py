"""Tests for ConversationMemory and its integration with SimpleChatModel."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from nerif.memory import ConversationMemory
from nerif.model import SimpleChatModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(text: str):
    """Build a minimal fake get_model_response return value."""
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Basic message operations
# ---------------------------------------------------------------------------


def test_add_message_basic():
    mem = ConversationMemory()
    mem.add_message("user", "Hello")
    mem.add_message("assistant", "Hi there!")
    msgs = mem.get_messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello"
    assert msgs[1]["role"] == "assistant"


def test_get_messages_no_summary():
    mem = ConversationMemory()
    mem.add_message("user", "A")
    mem.add_message("assistant", "B")
    result = mem.get_messages()
    # No summary -> returned list equals internal list
    assert result == mem._messages
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Sliding window — max_messages
# ---------------------------------------------------------------------------


def test_max_messages_window():
    mem = ConversationMemory(max_messages=5)
    # Add a system message first
    mem.add_message("system", "You are helpful.")
    # Add 10 user/assistant pairs = 20 non-system messages
    for i in range(10):
        mem.add_message("user", f"msg {i}")
        mem.add_message("assistant", f"reply {i}")
    non_system = [m for m in mem._messages if m["role"] != "system"]
    assert len(non_system) <= 5


def test_max_messages_does_not_drop_system():
    mem = ConversationMemory(max_messages=2)
    mem.add_message("system", "System prompt")
    for i in range(5):
        mem.add_message("user", f"msg {i}")
    # System message must still be present
    system_msgs = [m for m in mem._messages if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "System prompt"


# ---------------------------------------------------------------------------
# Sliding window — max_tokens
# ---------------------------------------------------------------------------


def test_max_tokens_window():
    # Each message with content "x" * 40 => ~10 tokens
    mem = ConversationMemory(max_tokens=25)
    for i in range(10):
        mem.add_message("user", "x" * 40)  # ~10 tokens each
    assert mem.token_count() <= 25


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def test_summarize_basic():
    mock_resp = _make_mock_response("Summary of first messages.")

    with patch("nerif.memory.conversation.get_model_response", return_value=mock_resp) as mock_fn:
        mem = ConversationMemory(max_messages=4, summarize=True, summarize_model="gpt-4o-mini")
        for i in range(6):
            mem.add_message("user", f"message {i}")
            mem.add_message("assistant", f"response {i}")

        # Summarization should have been triggered at least once
        assert mock_fn.called
        assert mem._summary is not None
        assert "Summary" in mem._summary


def test_summarize_prepends_summary_in_get_messages():
    mock_resp = _make_mock_response("Key facts from prior conversation.")

    with patch("nerif.memory.conversation.get_model_response", return_value=mock_resp):
        mem = ConversationMemory(max_messages=4, summarize=True)
        for i in range(6):
            mem.add_message("user", f"msg {i}")
            mem.add_message("assistant", f"rep {i}")

    if mem._summary:
        msgs = mem.get_messages()
        assert msgs[0]["role"] == "system"
        assert "Previous conversation summary:" in msgs[0]["content"]


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip():
    mem = ConversationMemory()
    mem.add_message("system", "You are a test assistant.")
    mem.add_message("user", "Hello world")
    mem.add_message("assistant", "Hello back")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name

    try:
        mem.save(path)

        loaded = ConversationMemory.load(path)
        assert len(loaded._messages) == len(mem._messages)
        for orig, loaded_msg in zip(mem._messages, loaded._messages):
            assert orig["role"] == loaded_msg["role"]
            assert orig["content"] == loaded_msg["content"]
    finally:
        os.unlink(path)


def test_save_load_with_summary():
    mem = ConversationMemory()
    mem._summary = "Prior summary text."
    mem.add_message("user", "Recent question")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name

    try:
        mem.save(path)
        loaded = ConversationMemory.load(path)
        assert loaded._summary == "Prior summary text."
        assert len(loaded._messages) == 1
    finally:
        os.unlink(path)


def test_save_produces_valid_json():
    mem = ConversationMemory()
    mem.add_message("user", "test")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name

    try:
        mem.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data["version"] == "1.1"
        assert "messages" in data
        assert "summary" in data
        assert "metadata" in data
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


def test_clear():
    mem = ConversationMemory()
    mem.add_message("user", "Hello")
    mem.add_message("assistant", "Hi")
    mem._summary = "some summary"
    mem.clear()
    assert mem._messages == []
    assert mem._summary is None


def test_clear_then_add():
    mem = ConversationMemory()
    mem.add_message("user", "First")
    mem.clear()
    mem.add_message("user", "Second")
    assert len(mem._messages) == 1
    assert mem._messages[0]["content"] == "Second"


# ---------------------------------------------------------------------------
# System message preservation
# ---------------------------------------------------------------------------


def test_system_message_preserved_drop_mode():
    """System message is never dropped when max_messages is enforced."""
    mem = ConversationMemory(max_messages=2)
    mem.add_message("system", "SYSTEM")
    for i in range(10):
        mem.add_message("user", f"u{i}")
    system_msgs = [m for m in mem._messages if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "SYSTEM"


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def test_estimate_tokens_string():
    mem = ConversationMemory()
    msg = {"role": "user", "content": "a" * 40}  # 40 chars -> 10 tokens
    assert mem._estimate_tokens(msg) == 10


def test_estimate_tokens_short_string():
    mem = ConversationMemory()
    msg = {"role": "user", "content": "hi"}  # 2 chars -> max(1, 0) = 1
    assert mem._estimate_tokens(msg) == 1


def test_estimate_tokens_multimodal():
    mem = ConversationMemory()
    content = [
        {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
        {"type": "text", "text": "a" * 40},
    ]
    msg = {"role": "user", "content": content}
    estimated = mem._estimate_tokens(msg)
    # 85 for image + 10 for text
    assert estimated == 95


def test_estimate_tokens_none_content():
    mem = ConversationMemory()
    msg = {"role": "assistant", "content": None}
    assert mem._estimate_tokens(msg) == 0


# ---------------------------------------------------------------------------
# Token count
# ---------------------------------------------------------------------------


def test_token_count_accumulates():
    mem = ConversationMemory()
    mem.add_message("user", "a" * 40)  # 10 tokens
    mem.add_message("assistant", "b" * 80)  # 20 tokens
    assert mem.token_count() == 30


# ---------------------------------------------------------------------------
# Integration: SimpleChatModel with memory
# ---------------------------------------------------------------------------


def test_integration_with_chat_model():
    """Verify SimpleChatModel routes messages through memory when provided."""
    mem = ConversationMemory()
    model = SimpleChatModel(
        model="gpt-4o",
        default_prompt="You are helpful.",
        memory=mem,
    )

    # After __init__, the system message should be in memory
    assert len(mem._messages) == 1
    assert mem._messages[0]["role"] == "system"
    assert mem._messages[0]["content"] == "You are helpful."

    # model.messages should be the same object as mem._messages
    assert model.messages is mem._messages

    # Simulate a chat call without hitting the API
    mock_resp = _make_mock_response("4")
    with patch("nerif.model.model.get_model_response", return_value=mock_resp):
        result = model.chat("What is 2+2?", append=True)

    assert result == "4"
    # Memory should have: system + user + assistant = 3
    assert len(mem._messages) == 3
    roles = [m["role"] for m in mem._messages]
    assert roles == ["system", "user", "assistant"]


def test_integration_reset_clears_memory():
    mem = ConversationMemory()
    model = SimpleChatModel(default_prompt="System.", memory=mem)

    mock_resp = _make_mock_response("ok")
    with patch("nerif.model.model.get_model_response", return_value=mock_resp):
        model.chat("Hello", append=True)

    assert len(mem._messages) == 3  # system + user + assistant

    model.reset()

    assert len(mem._messages) == 1
    assert mem._messages[0]["role"] == "system"
    assert mem._summary is None


# ---------------------------------------------------------------------------
# Backward compatibility: no memory
# ---------------------------------------------------------------------------


def test_memory_default_none():
    """SimpleChatModel without memory works exactly as before."""
    model = SimpleChatModel(default_prompt="System.")
    assert model.memory is None
    # messages is a plain list with just the system message
    assert len(model.messages) == 1
    assert model.messages[0]["role"] == "system"

    mock_resp = _make_mock_response("pong")
    with patch("nerif.model.model.get_model_response", return_value=mock_resp):
        result = model.chat("ping", append=True)

    assert result == "pong"
    assert len(model.messages) == 3


def test_memory_none_reset():
    model = SimpleChatModel(default_prompt="Hello.")
    assert model.memory is None

    mock_resp = _make_mock_response("reply")
    with patch("nerif.model.model.get_model_response", return_value=mock_resp):
        model.chat("msg", append=True)

    model.reset()
    assert len(model.messages) == 1
    assert model.messages[0]["content"] == "Hello."
