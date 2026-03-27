import json
import os
import tempfile

from nerif.agent.state import TokenUsage
from nerif.observability.dashboard import TraceDashboard, TraceExporter
from nerif.observability.store import FileTraceStore
from nerif.observability.tracing import Span, SpanKind, SpanStatus, Trace


def _make_trace_with_children() -> Trace:
    root = Span(
        trace_id="t1", span_id="root", parent_span_id=None,
        name="agent:writer", kind=SpanKind.AGENT,
        start_time=1.0, end_time=4.0,
        status=SpanStatus.OK, attributes={}, events=[],
        token_usage=TokenUsage(500, 300, 800), error=None,
    )
    child1 = Span(
        trace_id="t1", span_id="child1", parent_span_id="root",
        name="llm:gpt-4o", kind=SpanKind.LLM,
        start_time=1.0, end_time=2.0,
        status=SpanStatus.OK, attributes={}, events=[],
        token_usage=TokenUsage(200, 100, 300), error=None,
    )
    child2 = Span(
        trace_id="t1", span_id="child2", parent_span_id="root",
        name="tool:search", kind=SpanKind.TOOL,
        start_time=2.0, end_time=3.0,
        status=SpanStatus.OK, attributes={}, events=[],
        token_usage=None, error=None,
    )
    return Trace(
        trace_id="t1", root_span=root, spans=[root, child1, child2],
        start_time=1.0, end_time=4.0,
        total_token_usage=TokenUsage(700, 400, 1100),
        total_cost_usd=0.05, metadata={},
    )


def test_trace_tree():
    with tempfile.TemporaryDirectory() as d:
        store = FileTraceStore(d)
        store.save_trace(_make_trace_with_children())
        dashboard = TraceDashboard(store)
        tree = dashboard.trace_tree("t1")
        assert "agent:writer" in tree
        assert "llm:gpt-4o" in tree
        assert "tool:search" in tree


def test_cost_summary():
    with tempfile.TemporaryDirectory() as d:
        store = FileTraceStore(d)
        store.save_trace(_make_trace_with_children())
        dashboard = TraceDashboard(store)
        summary = dashboard.cost_summary("t1")
        assert "agent:writer" in summary or "gpt-4o" in summary


def test_trace_list():
    with tempfile.TemporaryDirectory() as d:
        store = FileTraceStore(d)
        store.save_trace(_make_trace_with_children())
        dashboard = TraceDashboard(store)
        listing = dashboard.trace_list()
        assert "t1" in listing


def test_exporter_to_dict():
    trace = _make_trace_with_children()
    d = TraceExporter.to_dict(trace)
    assert d["trace_id"] == "t1"
    assert len(d["spans"]) == 3


def test_exporter_to_json():
    trace = _make_trace_with_children()
    j = TraceExporter.to_json(trace)
    parsed = json.loads(j)
    assert parsed["trace_id"] == "t1"


def test_exporter_to_csv():
    trace = _make_trace_with_children()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "export.csv")
        TraceExporter.to_csv([trace], path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 4  # header + 3 spans


def test_exporter_to_dataframe():
    trace = _make_trace_with_children()
    df = TraceExporter.to_dataframe([trace])
    assert len(df["trace_id"]) == 3
    assert df["name"][0] == "agent:writer"
