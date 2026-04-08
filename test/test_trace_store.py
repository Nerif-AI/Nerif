import os
import tempfile

from nerif.agent.state import TokenUsage
from nerif.observability.store import FileTraceStore, SQLiteTraceStore
from nerif.observability.tracing import Span, SpanKind, SpanStatus, Trace


def _make_trace(trace_id: str = "test123", name: str = "agent:test") -> Trace:
    root = Span(
        trace_id=trace_id,
        span_id="span1",
        parent_span_id=None,
        name=name,
        kind=SpanKind.AGENT,
        start_time=1000.0,
        end_time=1001.0,
        status=SpanStatus.OK,
        attributes={},
        events=[],
        token_usage=TokenUsage(100, 50, 150),
        error=None,
    )
    return Trace(
        trace_id=trace_id,
        root_span=root,
        spans=[root],
        start_time=1000.0,
        end_time=1001.0,
        total_token_usage=TokenUsage(100, 50, 150),
        total_cost_usd=0.01,
        metadata={"tag": "test"},
    )


# --- FileTraceStore tests ---


def test_file_store_save_and_get():
    with tempfile.TemporaryDirectory() as d:
        store = FileTraceStore(d)
        trace = _make_trace()
        store.save_trace(trace)
        loaded = store.get_trace("test123")
        assert loaded is not None
        assert loaded.trace_id == "test123"
        assert loaded.root_span.name == "agent:test"
        assert loaded.total_token_usage.total_tokens == 150
        assert loaded.metadata["tag"] == "test"


def test_file_store_list_traces():
    with tempfile.TemporaryDirectory() as d:
        store = FileTraceStore(d)
        store.save_trace(_make_trace("t1", "agent:first"))
        store.save_trace(_make_trace("t2", "agent:second"))
        summaries = store.list_traces(limit=10)
        assert len(summaries) == 2


def test_file_store_delete():
    with tempfile.TemporaryDirectory() as d:
        store = FileTraceStore(d)
        store.save_trace(_make_trace("t1"))
        assert store.delete_trace("t1") is True
        assert store.get_trace("t1") is None
        assert store.delete_trace("nonexistent") is False


def test_file_store_get_nonexistent():
    with tempfile.TemporaryDirectory() as d:
        store = FileTraceStore(d)
        assert store.get_trace("nope") is None


# --- SQLiteTraceStore tests ---


def test_sqlite_store_save_and_get():
    with tempfile.TemporaryDirectory() as d:
        store = SQLiteTraceStore(os.path.join(d, "traces.db"))
        trace = _make_trace()
        store.save_trace(trace)
        loaded = store.get_trace("test123")
        assert loaded is not None
        assert loaded.trace_id == "test123"
        assert loaded.root_span.name == "agent:test"
        assert loaded.total_token_usage.total_tokens == 150


def test_sqlite_store_list_traces():
    with tempfile.TemporaryDirectory() as d:
        store = SQLiteTraceStore(os.path.join(d, "traces.db"))
        store.save_trace(_make_trace("t1", "agent:first"))
        store.save_trace(_make_trace("t2", "agent:second"))
        summaries = store.list_traces(limit=10)
        assert len(summaries) == 2


def test_sqlite_store_delete():
    with tempfile.TemporaryDirectory() as d:
        store = SQLiteTraceStore(os.path.join(d, "traces.db"))
        store.save_trace(_make_trace("t1"))
        assert store.delete_trace("t1") is True
        assert store.get_trace("t1") is None
        assert store.delete_trace("nonexistent") is False


def test_sqlite_store_time_filter():
    with tempfile.TemporaryDirectory() as d:
        store = SQLiteTraceStore(os.path.join(d, "traces.db"))
        store.save_trace(_make_trace("t1"))  # start_time=1000.0
        assert len(store.list_traces(start_time=999.0)) == 1
        assert len(store.list_traces(start_time=1001.0)) == 0
        assert len(store.list_traces(end_time=1000.0)) == 1
        assert len(store.list_traces(end_time=999.0)) == 0
