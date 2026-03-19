"""Conversation memory management for Nerif chat models."""

import json
from typing import Any, Dict, List, Optional

from ..utils import get_model_response

DEFAULT_SUMMARY_PROMPT = "Summarize the following conversation concisely, preserving key facts and context:"


class ConversationMemory:
    """
    Manages conversation history with optional sliding window and summarization.

    Supports two window strategies:
    - max_messages: Keep only the most recent N non-system messages.
    - max_tokens: Keep messages within an approximate token budget.

    When summarize=True, exceeded messages are summarized via an LLM call and
    stored as a preceding system context. When summarize=False, oldest messages
    are dropped silently (FIFO). System messages are never dropped.

    Args:
        max_messages: Maximum number of non-system messages to retain.
        max_tokens: Maximum approximate token count across all messages.
        summarize: If True, summarize dropped messages instead of discarding.
        summarize_model: Model used for summarization (default: gpt-4o-mini).
        summary_prompt: Custom prompt for the summarization request.
    """

    def __init__(
        self,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        summarize: bool = False,
        summarize_model: str = "gpt-4o-mini",
        summary_prompt: Optional[str] = None,
        counter: Optional[Any] = None,
    ):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summarize = summarize
        self.summarize_model = summarize_model
        self.summary_prompt = summary_prompt if summary_prompt is not None else DEFAULT_SUMMARY_PROMPT
        self.counter = counter

        self._messages: List[Dict[str, Any]] = []
        self._summary: Optional[str] = None
        self._metadata: Dict[str, Any] = {}

    def add_message(self, role: str, content: Any) -> None:
        """Append a message and trigger window management."""
        self._messages.append({"role": role, "content": content})
        self._manage_window()

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return the effective message list for API calls.

        If a summary exists, it is prepended as a system message before the
        current messages list.
        """
        if self._summary is None:
            return list(self._messages)

        summary_msg = {
            "role": "system",
            "content": f"Previous conversation summary: {self._summary}",
        }
        return [summary_msg] + list(self._messages)

    def _manage_window(self) -> None:
        """Enforce max_messages and max_tokens constraints."""
        if self.max_messages is not None:
            non_system = [m for m in self._messages if m["role"] != "system"]
            if len(non_system) > self.max_messages:
                if self.summarize:
                    self._summarize_oldest()
                else:
                    self._drop_oldest()

        if self.max_tokens is not None:
            while self.token_count() > self.max_tokens:
                non_system = [m for m in self._messages if m["role"] != "system"]
                if len(non_system) < 2:
                    break
                if self.summarize:
                    self._summarize_oldest()
                    # After summarization, check again but avoid infinite loop
                    # if a single message still exceeds the limit
                    if self.token_count() > self.max_tokens:
                        non_system = [m for m in self._messages if m["role"] != "system"]
                        if len(non_system) < 2:
                            break
                        # Drop anyway to prevent infinite loop
                        self._drop_oldest_single()
                else:
                    self._drop_oldest_single()

    def _drop_oldest(self) -> None:
        """Drop oldest non-system messages until within max_messages."""
        while True:
            non_system = [m for m in self._messages if m["role"] != "system"]
            if len(non_system) <= self.max_messages:
                break
            # Find and remove the first non-system message
            for i, msg in enumerate(self._messages):
                if msg["role"] != "system":
                    self._messages.pop(i)
                    break

    def _drop_oldest_single(self) -> None:
        """Drop a single oldest non-system message."""
        for i, msg in enumerate(self._messages):
            if msg["role"] != "system":
                self._messages.pop(i)
                break

    def _summarize_oldest(self) -> None:
        """Summarize the oldest 50% of non-system messages (minimum 2)."""
        non_system = [m for m in self._messages if m["role"] != "system"]
        count = len(non_system)
        # Summarize at least 2 and at most 50% of non-system messages
        num_to_summarize = max(2, count // 2)

        # Collect the oldest num_to_summarize non-system messages
        to_summarize = []
        indices_to_remove = []
        collected = 0
        for i, msg in enumerate(self._messages):
            if msg["role"] != "system":
                to_summarize.append(msg)
                indices_to_remove.append(i)
                collected += 1
                if collected >= num_to_summarize:
                    break

        # Build conversation text for summarization
        convo_lines = []
        for msg in to_summarize:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, str):
                convo_lines.append(f"{role}: {content}")
            else:
                # Multimodal: use text parts only
                text_parts = []
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                convo_lines.append(f"{role}: {' '.join(text_parts)}")

        convo_text = "\n".join(convo_lines)

        # Build the summarization prompt
        prompt_messages = [
            {"role": "system", "content": self.summary_prompt},
            {"role": "user", "content": convo_text},
        ]

        # If there is a prior summary, prepend it for context
        if self._summary:
            combined = f"Prior summary: {self._summary}\n\nNew messages to incorporate:\n{convo_text}"
            prompt_messages = [
                {"role": "system", "content": self.summary_prompt},
                {"role": "user", "content": combined},
            ]

        kwargs = {"model": self.summarize_model}
        if self.counter is not None:
            kwargs["counter"] = self.counter
        result = get_model_response(prompt_messages, **kwargs)
        new_summary = result.choices[0].message.content

        self._summary = new_summary

        # Remove summarized messages from _messages (in reverse index order)
        for i in reversed(indices_to_remove):
            self._messages.pop(i)

    def _estimate_tokens(self, message: Dict[str, Any]) -> int:
        """Estimate token count for a single message."""
        content = message.get("content")
        if content is None:
            return 0
        if isinstance(content, str):
            return max(1, len(content) // 4)
        if isinstance(content, list):
            total = 0
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type", "")
                if part_type == "text":
                    text = part.get("text", "")
                    total += max(1, len(text) // 4)
                elif part_type in ("image_url", "image"):
                    total += 85
                else:
                    # audio, video, or other parts — use a flat estimate
                    total += 85
            return total
        # Fallback for any other content type
        return max(1, len(str(content)) // 4)

    def token_count(self) -> int:
        """Return the approximate total token count for all current messages."""
        return sum(self._estimate_tokens(m) for m in self._messages)

    def clear(self) -> None:
        """Reset messages and summary.

        Uses list.clear() instead of reassignment to preserve the reference
        held by SimpleChatModel.messages.
        """
        self._messages.clear()
        self._summary = None

    def save(self, path: str) -> None:
        """Serialize conversation state to a JSON file."""
        data = {
            "version": "1.1",
            "config": {
                "max_messages": self.max_messages,
                "max_tokens": self.max_tokens,
                "summarize": self.summarize,
                "summarize_model": self.summarize_model,
                "summary_prompt": self.summary_prompt if self.summary_prompt != DEFAULT_SUMMARY_PROMPT else None,
            },
            "metadata": self._metadata,
            "summary": self._summary,
            "messages": self._messages,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConversationMemory":
        """Deserialize a ConversationMemory from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = data.get("config", {})
        instance = cls(
            max_messages=config.get("max_messages"),
            max_tokens=config.get("max_tokens"),
            summarize=config.get("summarize", False),
            summarize_model=config.get("summarize_model", "gpt-4o-mini"),
            summary_prompt=config.get("summary_prompt"),
        )
        instance._summary = data.get("summary")
        instance._messages = data.get("messages", [])
        instance._metadata = data.get("metadata", {})
        return instance
