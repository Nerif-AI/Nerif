"""Tests for TracingCallbackHandler — verifies that callback events are mapped to spans."""

import time

from nerif.observability.tracing import (
    SpanKind,
    TraceCollector,
    TracingCallbackHandler,
    end_span,
    start_span,
)
from nerif.utils.callbacks import (
    LLMEndEvent,
    LLMErrorEvent,
    LLMStartEvent,
    ToolCallEvent,
)


def test_tracing_handler_creates_llm_spans():
    collector = TraceCollector()
    handler = TracingCallbackHandler(collector)
    token = collector.activate()
    try:
        agent_ctx = start_span("agent:test", SpanKind.AGENT)
        handler.on_llm_start(LLMStartEvent(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
            timestamp=time.time(), kwargs={},
        ))
        handler.on_llm_end(LLMEndEvent(
            model="gpt-4o", response="hello", latency_ms=100.0,
            prompt_tokens=10, completion_tokens=5, cost_usd=0.001,
        ))
        end_span(agent_ctx)
        assert collector.last_trace_id is not None
    finally:
        collector.deactivate(token)


def test_tracing_handler_error_span():
    collector = TraceCollector()
    handler = TracingCallbackHandler(collector)
    token = collector.activate()
    try:
        agent_ctx = start_span("agent:test", SpanKind.AGENT)
        handler.on_llm_start(LLMStartEvent(
            model="gpt-4o", messages=[], timestamp=time.time(), kwargs={},
        ))
        handler.on_llm_error(LLMErrorEvent(
            model="gpt-4o", error=Exception("timeout"),
            latency_ms=5000.0, will_retry=True,
        ))
        end_span(agent_ctx)
    finally:
        collector.deactivate(token)


def test_tracing_handler_tool_span():
    collector = TraceCollector()
    handler = TracingCallbackHandler(collector)
    token = collector.activate()
    try:
        agent_ctx = start_span("agent:test", SpanKind.AGENT)
        handler.on_tool_call(ToolCallEvent(
            tool_name="search", arguments={"query": "test"},
            result="found it", latency_ms=200.0, success=True,
        ))
        end_span(agent_ctx)
    finally:
        collector.deactivate(token)
