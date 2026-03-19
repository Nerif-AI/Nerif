"""Callback / hook system for Nerif LLM operations."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .constants import LOGGER_NAME

_CALLBACK_LOGGER = logging.getLogger(f"{LOGGER_NAME}.callbacks")


@dataclass
class LLMStartEvent:
    """Fired before an LLM API call."""

    model: str
    messages: List[Dict]
    timestamp: float
    kwargs: Dict[str, Any]


@dataclass
class LLMEndEvent:
    """Fired after a successful LLM API call."""

    model: str
    response: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass
class LLMErrorEvent:
    """Fired when an LLM API call fails."""

    model: str
    error: Exception
    latency_ms: float
    will_retry: bool


@dataclass
class ToolCallEvent:
    """Fired when an agent executes a tool."""

    tool_name: str
    arguments: Dict[str, Any]
    result: str
    latency_ms: float
    success: bool


@dataclass
class FallbackEvent:
    """Fired when a model fallback occurs."""

    failed_model: str
    next_model: str
    error: Exception


@dataclass
class RetryEvent:
    """Fired when a request is being retried."""

    model: str
    attempt: int
    max_retries: int
    delay: float
    error: Exception


@dataclass
class MemoryEvent:
    """Fired on memory operations (summarize, window trim)."""

    action: str
    messages_before: int
    messages_after: int
    summary: Optional[str] = None


class CallbackHandler:
    """Base class for callback handlers. Override methods you care about."""

    def on_llm_start(self, event: LLMStartEvent) -> None:
        pass

    def on_llm_end(self, event: LLMEndEvent) -> None:
        pass

    def on_llm_error(self, event: LLMErrorEvent) -> None:
        pass

    def on_tool_call(self, event: ToolCallEvent) -> None:
        pass

    def on_fallback(self, event: FallbackEvent) -> None:
        pass

    def on_retry(self, event: RetryEvent) -> None:
        pass

    def on_memory(self, event: MemoryEvent) -> None:
        pass


class CallbackManager:
    """Manages a list of callback handlers."""

    def __init__(self):
        self._handlers: List[CallbackHandler] = []

    def add_handler(self, handler: CallbackHandler) -> None:
        self._handlers.append(handler)

    def remove_handler(self, handler: CallbackHandler) -> None:
        self._handlers.remove(handler)

    def fire(self, method_name: str, event: Any) -> None:
        for handler in self._handlers:
            fn = getattr(handler, method_name, None)
            if fn is not None:
                try:
                    fn(event)
                except Exception:
                    _CALLBACK_LOGGER.debug(
                        "Callback %s.%s raised an exception",
                        type(handler).__name__,
                        method_name,
                        exc_info=True,
                    )


class LoggingCallbackHandler(CallbackHandler):
    """Built-in handler that logs all events via Python logging."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(f"{LOGGER_NAME}.callbacks")

    def on_llm_start(self, event: LLMStartEvent) -> None:
        self.logger.info("LLM request to %s (%d messages)", event.model, len(event.messages))

    def on_llm_end(self, event: LLMEndEvent) -> None:
        self.logger.info("LLM response from %s (%.0fms, $%.6f)", event.model, event.latency_ms, event.cost_usd)

    def on_llm_error(self, event: LLMErrorEvent) -> None:
        self.logger.warning(
            "LLM error from %s: %s (%.0fms, retry=%s)", event.model, event.error, event.latency_ms, event.will_retry
        )

    def on_tool_call(self, event: ToolCallEvent) -> None:
        self.logger.info("Tool %s called (%.0fms, success=%s)", event.tool_name, event.latency_ms, event.success)

    def on_fallback(self, event: FallbackEvent) -> None:
        self.logger.warning("Fallback: %s -> %s due to %s", event.failed_model, event.next_model, event.error)

    def on_retry(self, event: RetryEvent) -> None:
        self.logger.info(
            "Retry %d/%d for %s (delay=%.1fs)", event.attempt, event.max_retries, event.model, event.delay
        )

    def on_memory(self, event: MemoryEvent) -> None:
        self.logger.info("Memory %s: %d -> %d messages", event.action, event.messages_before, event.messages_after)
