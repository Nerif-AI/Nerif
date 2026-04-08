"""Tests for the core tracing data model and context propagation."""

import time

from nerif.observability.tracing import (
    Span,
    SpanKind,
    SpanStatus,
    TraceCollector,
    end_span,
    get_current_span,
    get_current_trace_id,
    start_span,
)


def test_span_creation():
    span = Span(
        trace_id="abc123",
        span_id="def456",
        parent_span_id=None,
        name="agent:test",
        kind=SpanKind.AGENT,
        start_time=time.perf_counter(),
        end_time=None,
        status=SpanStatus.RUNNING,
        attributes={"key": "value"},
        events=[],
        token_usage=None,
        error=None,
    )
    assert span.trace_id == "abc123"
    assert span.kind == SpanKind.AGENT
    assert span.status == SpanStatus.RUNNING
    assert span.duration_ms() is None


def test_span_duration():
    start = time.perf_counter()
    span = Span(
        trace_id="a",
        span_id="b",
        parent_span_id=None,
        name="test",
        kind=SpanKind.LLM,
        start_time=start,
        end_time=start + 0.5,
        status=SpanStatus.OK,
        attributes={},
        events=[],
        token_usage=None,
        error=None,
    )
    assert abs(span.duration_ms() - 500.0) < 1.0


def test_span_add_event():
    span = Span(
        trace_id="a",
        span_id="b",
        parent_span_id=None,
        name="test",
        kind=SpanKind.AGENT,
        start_time=0.0,
        end_time=None,
        status=SpanStatus.RUNNING,
        attributes={},
        events=[],
        token_usage=None,
        error=None,
    )
    span.add_event("retry", attempt=2)
    assert len(span.events) == 1
    assert span.events[0].name == "retry"
    assert span.events[0].attributes["attempt"] == 2


def test_span_serialization():
    span = Span(
        trace_id="a",
        span_id="b",
        parent_span_id="c",
        name="llm:gpt-4o",
        kind=SpanKind.LLM,
        start_time=1.0,
        end_time=2.0,
        status=SpanStatus.OK,
        attributes={"model": "gpt-4o"},
        events=[],
        token_usage=None,
        error=None,
    )
    d = span.to_dict()
    restored = Span.from_dict(d)
    assert restored.trace_id == "a"
    assert restored.kind == SpanKind.LLM
    assert restored.status == SpanStatus.OK
    assert restored.attributes["model"] == "gpt-4o"


def test_start_end_span():
    collector = TraceCollector()
    token = collector.activate()
    try:
        ctx = start_span("agent:test", SpanKind.AGENT, task="hello")
        assert get_current_span() is ctx.span
        assert get_current_trace_id() is not None

        span = end_span(ctx)
        assert span.status == SpanStatus.OK
        assert span.end_time is not None
        assert span.duration_ms() > 0
        assert span.attributes["task"] == "hello"
        assert get_current_span() is None
    finally:
        collector.deactivate(token)


def test_nested_spans():
    collector = TraceCollector()
    token = collector.activate()
    try:
        parent_ctx = start_span("agent:parent", SpanKind.AGENT)
        child_ctx = start_span("llm:gpt-4o", SpanKind.LLM)

        assert child_ctx.span.parent_span_id == parent_ctx.span.span_id
        assert child_ctx.span.trace_id == parent_ctx.span.trace_id

        end_span(child_ctx)
        assert get_current_span() is parent_ctx.span

        end_span(parent_ctx)
        assert get_current_span() is None
    finally:
        collector.deactivate(token)


def test_collector_assembles_trace():
    collector = TraceCollector()
    token = collector.activate()
    try:
        ctx = start_span("agent:root", SpanKind.AGENT)
        child_ctx = start_span("llm:gpt-4o", SpanKind.LLM)
        end_span(child_ctx)
        end_span(ctx)  # root span end triggers finish_trace

        assert collector.last_trace_id is not None
    finally:
        collector.deactivate(token)


def test_span_error():
    collector = TraceCollector()
    token = collector.activate()
    try:
        ctx = start_span("tool:failing", SpanKind.TOOL)
        span = end_span(ctx, error=ValueError("something broke"))
        assert span.status == SpanStatus.ERROR
        assert span.error == "something broke"
    finally:
        collector.deactivate(token)


def test_no_collector_no_crash():
    """start_span/end_span should work even without a collector (spans just aren't collected)."""
    ctx = start_span("agent:orphan", SpanKind.AGENT)
    span = end_span(ctx)
    assert span.status == SpanStatus.OK
