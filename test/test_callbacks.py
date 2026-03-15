"""Tests for the callback / hook system."""

import logging

from nerif.utils.callbacks import (
    CallbackHandler,
    CallbackManager,
    FallbackEvent,
    LLMEndEvent,
    LLMErrorEvent,
    LLMStartEvent,
    LoggingCallbackHandler,
    MemoryEvent,
    RetryEvent,
    ToolCallEvent,
)


class RecordingHandler(CallbackHandler):
    """Test handler that records all events."""

    def __init__(self):
        self.events = []

    def on_llm_start(self, event):
        self.events.append(("llm_start", event))

    def on_llm_end(self, event):
        self.events.append(("llm_end", event))

    def on_llm_error(self, event):
        self.events.append(("llm_error", event))

    def on_tool_call(self, event):
        self.events.append(("tool_call", event))

    def on_fallback(self, event):
        self.events.append(("fallback", event))

    def on_retry(self, event):
        self.events.append(("retry", event))

    def on_memory(self, event):
        self.events.append(("memory", event))


class TestCallbackManager:
    def test_add_and_fire(self):
        mgr = CallbackManager()
        handler = RecordingHandler()
        mgr.add_handler(handler)
        event = LLMStartEvent(model="gpt-4o", messages=[], timestamp=0.0, kwargs={})
        mgr.fire("on_llm_start", event)
        assert len(handler.events) == 1
        assert handler.events[0] == ("llm_start", event)

    def test_multiple_handlers(self):
        mgr = CallbackManager()
        h1 = RecordingHandler()
        h2 = RecordingHandler()
        mgr.add_handler(h1)
        mgr.add_handler(h2)
        event = LLMEndEvent(
            model="gpt-4o", response="hi", latency_ms=100.0, prompt_tokens=10, completion_tokens=5, cost_usd=0.001
        )
        mgr.fire("on_llm_end", event)
        assert len(h1.events) == 1
        assert len(h2.events) == 1

    def test_remove_handler(self):
        mgr = CallbackManager()
        handler = RecordingHandler()
        mgr.add_handler(handler)
        mgr.remove_handler(handler)
        event = LLMStartEvent(model="gpt-4o", messages=[], timestamp=0.0, kwargs={})
        mgr.fire("on_llm_start", event)
        assert len(handler.events) == 0

    def test_handler_exception_does_not_crash(self):
        class BadHandler(CallbackHandler):
            def on_llm_start(self, event):
                raise RuntimeError("boom")

        mgr = CallbackManager()
        bad = BadHandler()
        good = RecordingHandler()
        mgr.add_handler(bad)
        mgr.add_handler(good)
        event = LLMStartEvent(model="gpt-4o", messages=[], timestamp=0.0, kwargs={})
        mgr.fire("on_llm_start", event)
        assert len(good.events) == 1

    def test_handler_exception_is_logged(self, caplog):
        class BadHandler(CallbackHandler):
            def on_llm_start(self, event):
                raise RuntimeError("boom")

        mgr = CallbackManager()
        mgr.add_handler(BadHandler())
        event = LLMStartEvent(model="gpt-4o", messages=[], timestamp=0.0, kwargs={})
        with caplog.at_level(logging.DEBUG, logger="nerif.callbacks"):
            mgr.fire("on_llm_start", event)
        assert "raised an exception" in caplog.text

    def test_fire_nonexistent_method(self):
        mgr = CallbackManager()
        handler = RecordingHandler()
        mgr.add_handler(handler)
        mgr.fire("on_nonexistent", "some_event")
        assert len(handler.events) == 0


class TestAllEventTypes:
    def test_all_events_fire(self):
        mgr = CallbackManager()
        handler = RecordingHandler()
        mgr.add_handler(handler)

        mgr.fire("on_llm_start", LLMStartEvent(model="m", messages=[], timestamp=0.0, kwargs={}))
        mgr.fire(
            "on_llm_end",
            LLMEndEvent(model="m", response="r", latency_ms=0, prompt_tokens=0, completion_tokens=0, cost_usd=0),
        )
        mgr.fire("on_llm_error", LLMErrorEvent(model="m", error=Exception(), latency_ms=0, will_retry=False))
        mgr.fire("on_tool_call", ToolCallEvent(tool_name="t", arguments={}, result="r", latency_ms=0, success=True))
        mgr.fire("on_fallback", FallbackEvent(failed_model="a", next_model="b", error=Exception()))
        mgr.fire("on_retry", RetryEvent(model="m", attempt=1, max_retries=3, delay=1.0, error=Exception()))
        mgr.fire("on_memory", MemoryEvent(action="trim", messages_before=10, messages_after=5))

        assert len(handler.events) == 7
        types = [e[0] for e in handler.events]
        assert types == ["llm_start", "llm_end", "llm_error", "tool_call", "fallback", "retry", "memory"]


class TestLoggingCallbackHandler:
    def test_logs_llm_start(self, caplog):
        handler = LoggingCallbackHandler()
        event = LLMStartEvent(model="gpt-4o", messages=[{}, {}], timestamp=0.0, kwargs={})
        with caplog.at_level(logging.INFO, logger="nerif.callbacks"):
            handler.on_llm_start(event)
        assert "gpt-4o" in caplog.text
        assert "2 messages" in caplog.text

    def test_logs_llm_end(self, caplog):
        handler = LoggingCallbackHandler()
        event = LLMEndEvent(
            model="gpt-4o", response="r", latency_ms=150.0, prompt_tokens=10, completion_tokens=5, cost_usd=0.001
        )
        with caplog.at_level(logging.INFO, logger="nerif.callbacks"):
            handler.on_llm_end(event)
        assert "150" in caplog.text

    def test_logs_fallback(self, caplog):
        handler = LoggingCallbackHandler()
        event = FallbackEvent(failed_model="a", next_model="b", error=Exception("timeout"))
        with caplog.at_level(logging.WARNING, logger="nerif.callbacks"):
            handler.on_fallback(event)
        assert "a" in caplog.text and "b" in caplog.text
