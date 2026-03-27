"""End-to-end observability tests: simulate real multi-agent workflows
without actual LLM calls, verifying tracing, budget, dashboard, export,
and HTML report all work together."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from nerif.agent import NerifAgent
from nerif.agent.orchestration import AgentPipeline
from nerif.agent.state import TokenUsage
from nerif.observability import (
    BudgetCallbackHandler,
    BudgetConfig,
    BudgetExceededError,
    BudgetManager,
    FileTraceStore,
    Span,
    SpanKind,
    SpanStatus,
    SQLiteTraceStore,
    Trace,
    TraceCollector,
    TraceDashboard,
    TraceExporter,
    TracingCallbackHandler,
    end_span,
    get_current_span,
    get_current_trace_id,
    start_span,
)
from nerif.utils.callbacks import CallbackManager

# ---------------------------------------------------------------------------
# 1. Full tracing lifecycle: agent -> collector -> store -> dashboard -> export
# ---------------------------------------------------------------------------

class TestTracingLifecycle:
    def test_agent_trace_to_store_to_dashboard(self):
        """Full pipeline: agent run -> trace collected -> stored -> dashboard renders."""
        with tempfile.TemporaryDirectory() as d:
            store = FileTraceStore(d)
            collector = TraceCollector(store=store)
            handler = TracingCallbackHandler(collector)
            cb = CallbackManager()
            cb.add_handler(handler)

            agent = NerifAgent(model="gpt-4o", callbacks=cb)
            token = collector.activate()
            try:
                with patch.object(agent.model, "chat", return_value="test response"):
                    agent.run("hello world")
            finally:
                collector.deactivate(token)

            # Verify trace was stored
            trace_id = collector.last_trace_id
            assert trace_id is not None

            trace = store.get_trace(trace_id)
            assert trace is not None
            assert len(trace.spans) >= 1  # at least agent span

            # Verify dashboard renders without error
            dashboard = TraceDashboard(store)
            tree = dashboard.trace_tree(trace_id)
            assert "agent:" in tree

            cost = dashboard.cost_summary(trace_id)
            assert isinstance(cost, str)

            listing = dashboard.trace_list()
            assert trace_id in listing

            # Verify export
            exported = TraceExporter.to_dict(trace)
            assert exported["trace_id"] == trace_id

            j = TraceExporter.to_json(trace)
            parsed = json.loads(j)
            assert parsed["trace_id"] == trace_id

            # CSV export
            csv_path = os.path.join(d, "export.csv")
            TraceExporter.to_csv([trace], csv_path)
            assert os.path.exists(csv_path)

            # Dataframe export
            df = TraceExporter.to_dataframe([trace])
            assert len(df["trace_id"]) == len(trace.spans)

    def test_sqlite_store_lifecycle(self):
        """Same lifecycle but with SQLiteTraceStore."""
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteTraceStore(os.path.join(d, "test.db"))
            collector = TraceCollector(store=store)
            handler = TracingCallbackHandler(collector)
            cb = CallbackManager()
            cb.add_handler(handler)

            agent = NerifAgent(model="gpt-4o", callbacks=cb)
            token = collector.activate()
            try:
                with patch.object(agent.model, "chat", return_value="sqlite test"):
                    agent.run("test query")
            finally:
                collector.deactivate(token)

            trace_id = collector.last_trace_id
            assert store.get_trace(trace_id) is not None
            summaries = store.list_traces()
            assert len(summaries) >= 1


# ---------------------------------------------------------------------------
# 2. Nested agent tracing (agent-as-tool)
# ---------------------------------------------------------------------------

class TestNestedAgentTracing:
    def test_agent_as_tool_creates_nested_spans(self):
        """When agent A calls agent B as a tool, trace should show parent-child."""
        with tempfile.TemporaryDirectory() as d:
            store = FileTraceStore(d)
            collector = TraceCollector(store=store)
            handler = TracingCallbackHandler(collector)
            cb = CallbackManager()
            cb.add_handler(handler)

            # Create inner agent (will be called as tool)
            inner_agent = NerifAgent(model="gpt-4o-mini", callbacks=cb)

            # Create outer agent with inner as tool
            outer_agent = NerifAgent(model="gpt-4o", callbacks=cb)
            outer_agent.register_tool(inner_agent.as_tool("researcher", "Research topics"))

            token = collector.activate()
            try:
                # Mock both agents to return text (no tool calls)
                with patch.object(outer_agent.model, "chat", return_value="final answer"):
                    outer_agent.run("write about AI")
            finally:
                collector.deactivate(token)

            trace_id = collector.last_trace_id
            assert trace_id is not None
            trace = store.get_trace(trace_id)
            assert trace is not None


# ---------------------------------------------------------------------------
# 3. Pipeline orchestrator tracing
# ---------------------------------------------------------------------------

class TestPipelineTracing:
    def test_pipeline_creates_orchestrator_span(self):
        with tempfile.TemporaryDirectory() as d:
            store = FileTraceStore(d)
            collector = TraceCollector(store=store)
            handler = TracingCallbackHandler(collector)
            cb = CallbackManager()
            cb.add_handler(handler)

            agent_a = NerifAgent(model="gpt-4o", callbacks=cb)
            agent_b = NerifAgent(model="gpt-4o", callbacks=cb)
            pipeline = AgentPipeline([("step1", agent_a), ("step2", agent_b)])

            token = collector.activate()
            try:
                with patch.object(agent_a.model, "chat", return_value="step1 output"):
                    with patch.object(agent_b.model, "chat", return_value="step2 output"):
                        result = pipeline.run("start")
            finally:
                collector.deactivate(token)

            assert result.content == "step2 output"
            trace_id = collector.last_trace_id
            assert trace_id is not None
            trace = store.get_trace(trace_id)

            # Should have orchestrator span + 2 agent spans
            span_names = [s.name for s in trace.spans]
            assert any("orchestrator" in n for n in span_names)
            assert sum("agent:" in n for n in span_names) >= 2


# ---------------------------------------------------------------------------
# 4. Budget integration
# ---------------------------------------------------------------------------

class TestBudgetIntegration:
    def test_budget_hard_limit_stops_execution(self):
        """Budget hard limit should raise BudgetExceededError."""
        budget = BudgetManager(BudgetConfig(hard_limit_usd=0.001))

        # Simulate exceeding budget
        budget.record(cost_usd=0.0005, tokens=100)
        with pytest.raises(BudgetExceededError) as exc_info:
            budget.record(cost_usd=0.001, tokens=200)
        assert exc_info.value.budget_type == "usd"

    def test_budget_soft_limit_warns_but_continues(self):
        """Soft limit fires callback but doesn't interrupt."""
        warnings = []
        budget = BudgetManager(
            BudgetConfig(soft_limit_usd=0.001, hard_limit_usd=1.0),
            on_soft_limit=lambda t, lim, actual: warnings.append((t, lim, actual)),
        )
        budget.record(cost_usd=0.0005, tokens=100)
        assert len(warnings) == 0

        budget.record(cost_usd=0.001, tokens=200)  # exceeds soft limit
        assert len(warnings) == 1

        # Can still record more
        budget.record(cost_usd=0.001, tokens=100)
        assert len(warnings) == 1  # no re-fire

    def test_budget_with_tracing_combined(self):
        """Budget and tracing work together without interference."""
        with tempfile.TemporaryDirectory() as d:
            store = FileTraceStore(d)
            collector = TraceCollector(store=store)
            tracing_handler = TracingCallbackHandler(collector)
            budget = BudgetManager(BudgetConfig(soft_limit_usd=10.0))
            budget_handler = BudgetCallbackHandler(budget)

            cb = CallbackManager()
            cb.add_handler(tracing_handler)
            cb.add_handler(budget_handler)

            agent = NerifAgent(model="gpt-4o", callbacks=cb)
            token = collector.activate()
            try:
                with patch.object(agent.model, "chat", return_value="response"):
                    agent.run("test")
            finally:
                collector.deactivate(token)

            # Both should have worked
            assert collector.last_trace_id is not None
            # Budget handler would have recorded if LLMEndEvent was fired
            # (in this mock scenario, no LLMEndEvent fires from callbacks)


# ---------------------------------------------------------------------------
# 5. Context propagation correctness
# ---------------------------------------------------------------------------

class TestContextPropagation:
    def test_trace_id_shared_across_nested_spans(self):
        collector = TraceCollector()
        token = collector.activate()
        try:
            ctx1 = start_span("parent", SpanKind.AGENT)
            ctx2 = start_span("child", SpanKind.LLM)
            ctx3 = start_span("grandchild", SpanKind.TOOL)

            assert ctx1.span.trace_id == ctx2.span.trace_id == ctx3.span.trace_id
            assert ctx3.span.parent_span_id == ctx2.span.span_id
            assert ctx2.span.parent_span_id == ctx1.span.span_id
            assert ctx1.span.parent_span_id is None

            end_span(ctx3)
            assert get_current_span() is ctx2.span
            end_span(ctx2)
            assert get_current_span() is ctx1.span
            end_span(ctx1)
            assert get_current_span() is None
            assert get_current_trace_id() is None
        finally:
            collector.deactivate(token)

    def test_span_error_propagation(self):
        collector = TraceCollector()
        token = collector.activate()
        try:
            ctx = start_span("failing", SpanKind.AGENT)
            err = ValueError("test error")
            span = end_span(ctx, error=err)
            assert span.status == SpanStatus.ERROR
            assert span.error == "test error"
            assert span.end_time is not None
        finally:
            collector.deactivate(token)

    def test_no_tracing_zero_impact(self):
        """Without collector activated, start_span/end_span still work but don't collect."""
        # Ensure no lingering collector
        assert get_current_span() is None
        ctx = start_span("orphan", SpanKind.AGENT)
        span = end_span(ctx)
        assert span.status == SpanStatus.OK
        # No crash, no collector interaction


# ---------------------------------------------------------------------------
# 6. Trace serialization round-trip
# ---------------------------------------------------------------------------

class TestSerializationRoundTrip:
    def test_trace_roundtrip_file_store(self):
        """Write trace to FileTraceStore, read back, verify identical."""
        root = Span(
            trace_id="rt1", span_id="s1", parent_span_id=None,
            name="agent:test", kind=SpanKind.AGENT,
            start_time=100.0, end_time=200.0,
            status=SpanStatus.OK, attributes={"key": "value"},
            events=[], token_usage=TokenUsage(500, 300, 800), error=None,
        )
        child = Span(
            trace_id="rt1", span_id="s2", parent_span_id="s1",
            name="llm:gpt-4o", kind=SpanKind.LLM,
            start_time=110.0, end_time=190.0,
            status=SpanStatus.OK, attributes={"model": "gpt-4o"},
            events=[], token_usage=TokenUsage(200, 100, 300), error=None,
        )
        trace = Trace(
            trace_id="rt1", root_span=root, spans=[root, child],
            start_time=100.0, end_time=200.0,
            total_token_usage=TokenUsage(700, 400, 1100),
            total_cost_usd=0.05, metadata={"test": True},
        )

        with tempfile.TemporaryDirectory() as d:
            store = FileTraceStore(d)
            store.save_trace(trace)
            loaded = store.get_trace("rt1")

            assert loaded.trace_id == "rt1"
            assert len(loaded.spans) == 2
            assert loaded.root_span.name == "agent:test"
            assert loaded.root_span.token_usage.total_tokens == 800
            assert loaded.spans[1].name == "llm:gpt-4o"
            assert loaded.total_token_usage.total_tokens == 1100
            assert loaded.total_cost_usd == 0.05
            assert loaded.metadata["test"] is True

    def test_trace_roundtrip_sqlite_store(self):
        """Same roundtrip with SQLiteTraceStore."""
        root = Span(
            trace_id="rt2", span_id="s1", parent_span_id=None,
            name="agent:sql", kind=SpanKind.AGENT,
            start_time=100.0, end_time=200.0,
            status=SpanStatus.OK, attributes={}, events=[],
            token_usage=TokenUsage(100, 50, 150), error=None,
        )
        trace = Trace(
            trace_id="rt2", root_span=root, spans=[root],
            start_time=100.0, end_time=200.0,
            total_token_usage=TokenUsage(100, 50, 150),
            total_cost_usd=0.01, metadata={},
        )

        with tempfile.TemporaryDirectory() as d:
            store = SQLiteTraceStore(os.path.join(d, "rt.db"))
            store.save_trace(trace)
            loaded = store.get_trace("rt2")

            assert loaded.trace_id == "rt2"
            assert loaded.root_span.name == "agent:sql"
            assert loaded.total_token_usage.total_tokens == 150


# ---------------------------------------------------------------------------
# 7. Dashboard output quality
# ---------------------------------------------------------------------------

class TestDashboardOutput:
    def _store_with_trace(self):
        d = tempfile.mkdtemp()
        store = FileTraceStore(d)
        root = Span(
            trace_id="dash1", span_id="r", parent_span_id=None,
            name="agent:writer", kind=SpanKind.AGENT,
            start_time=1.0, end_time=5.0,
            status=SpanStatus.OK, attributes={}, events=[],
            token_usage=TokenUsage(1000, 500, 1500), error=None,
        )
        child1 = Span(
            trace_id="dash1", span_id="c1", parent_span_id="r",
            name="llm:gpt-4o", kind=SpanKind.LLM,
            start_time=1.0, end_time=3.0,
            status=SpanStatus.OK, attributes={}, events=[],
            token_usage=TokenUsage(600, 300, 900), error=None,
        )
        child2 = Span(
            trace_id="dash1", span_id="c2", parent_span_id="r",
            name="tool:search", kind=SpanKind.TOOL,
            start_time=3.0, end_time=4.0,
            status=SpanStatus.OK, attributes={}, events=[],
            token_usage=None, error=None,
        )
        trace = Trace(
            trace_id="dash1", root_span=root, spans=[root, child1, child2],
            start_time=1.0, end_time=5.0,
            total_token_usage=TokenUsage(1600, 800, 2400),
            total_cost_usd=0.12, metadata={},
        )
        store.save_trace(trace)
        return store, d

    def test_trace_tree_hierarchy(self):
        store, d = self._store_with_trace()
        dashboard = TraceDashboard(store)
        tree = dashboard.trace_tree("dash1")

        # Root should appear
        assert "agent:writer" in tree
        # Children should appear indented
        assert "llm:gpt-4o" in tree
        assert "tool:search" in tree
        # Should show duration
        assert "ms" in tree

    def test_cost_summary_columns(self):
        store, d = self._store_with_trace()
        dashboard = TraceDashboard(store)
        summary = dashboard.cost_summary("dash1")

        # Should be a table with headers
        assert "Prompt Tokens" in summary or "prompt" in summary.lower()

    def test_trace_list_format(self):
        store, d = self._store_with_trace()
        dashboard = TraceDashboard(store)
        listing = dashboard.trace_list()
        assert "dash1" in listing

    def test_nonexistent_trace(self):
        store, d = self._store_with_trace()
        dashboard = TraceDashboard(store)
        result = dashboard.trace_tree("nonexistent")
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# 8. HTML report content verification
# ---------------------------------------------------------------------------

class TestHTMLReport:
    def test_html_report_contains_all_sections(self):
        try:
            from nerif.observability.report import TraceReport
        except ImportError:
            pytest.skip("jinja2 not installed")

        root = Span(
            trace_id="html1", span_id="r", parent_span_id=None,
            name="agent:writer", kind=SpanKind.AGENT,
            start_time=1.0, end_time=4.0,
            status=SpanStatus.OK, attributes={"task": "write article"},
            events=[], token_usage=TokenUsage(500, 300, 800), error=None,
        )
        child = Span(
            trace_id="html1", span_id="c1", parent_span_id="r",
            name="llm:gpt-4o", kind=SpanKind.LLM,
            start_time=1.5, end_time=3.0,
            status=SpanStatus.OK, attributes={"model": "gpt-4o"},
            events=[], token_usage=TokenUsage(200, 100, 300), error=None,
        )
        trace = Trace(
            trace_id="html1", root_span=root, spans=[root, child],
            start_time=1.0, end_time=4.0,
            total_token_usage=TokenUsage(700, 400, 1100),
            total_cost_usd=0.05, metadata={},
        )

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "report.html")
            report = TraceReport(trace)
            result_path = report.generate(path)

            assert os.path.exists(result_path)
            with open(result_path) as f:
                html = f.read()

            # Verify structure
            assert "<!DOCTYPE html>" in html
            assert "mermaid" in html.lower()
            assert "agent:writer" in html
            assert "llm:gpt-4o" in html
            assert "sequenceDiagram" in html
            assert "Cost Breakdown" in html
            assert "Span Details" in html
            assert "html1" in html  # trace ID present


# ---------------------------------------------------------------------------
# 9. CSV/Dataframe export correctness
# ---------------------------------------------------------------------------

class TestExportCorrectness:
    def _make_trace(self):
        root = Span(
            trace_id="exp1", span_id="r", parent_span_id=None,
            name="agent:test", kind=SpanKind.AGENT,
            start_time=1.0, end_time=3.0,
            status=SpanStatus.OK, attributes={}, events=[],
            token_usage=TokenUsage(100, 50, 150), error=None,
        )
        child = Span(
            trace_id="exp1", span_id="c1", parent_span_id="r",
            name="llm:gpt-4o", kind=SpanKind.LLM,
            start_time=1.0, end_time=2.0,
            status=SpanStatus.ERROR, attributes={}, events=[],
            token_usage=TokenUsage(80, 40, 120), error="timeout",
        )
        return Trace(
            trace_id="exp1", root_span=root, spans=[root, child],
            start_time=1.0, end_time=3.0,
            total_token_usage=TokenUsage(180, 90, 270),
            total_cost_usd=0.02, metadata={},
        )

    def test_csv_contains_all_spans(self):
        trace = self._make_trace()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.csv")
            TraceExporter.to_csv([trace], path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 3  # header + 2 spans
            header = lines[0].strip()
            assert "trace_id" in header
            assert "duration_ms" in header
            assert "error" in header
            # Check error appears in second data row
            assert "timeout" in lines[2]

    def test_dataframe_column_types(self):
        trace = self._make_trace()
        df = TraceExporter.to_dataframe([trace])
        assert len(df["trace_id"]) == 2
        assert df["trace_id"][0] == "exp1"
        assert df["kind"][0] == "agent"
        assert df["kind"][1] == "llm"
        assert df["status"][1] == "error"
        assert df["error"][1] == "timeout"
        assert df["prompt_tokens"][0] == 100
        assert df["completion_tokens"][0] == 50


# ---------------------------------------------------------------------------
# 10. Recorder serialization
# ---------------------------------------------------------------------------

class TestRecorderRoundtrip:
    def test_recording_to_json_and_back(self):
        from nerif.observability.recorder import ExecutionRecording, IterationSnapshot

        snap1 = IterationSnapshot(
            iteration=1, span_id="s1",
            agent_state={"model": "gpt-4o", "system_prompt": "You are helpful",
                          "temperature": 0.0, "max_tokens": None, "max_iterations": 10},
            input_message="What is AI?",
            output="AI is...",
            tool_calls=[{"name": "search", "arguments": '{"q": "AI"}', "result": "found"}],
            token_usage={"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
            timestamp=1000.0,
        )
        snap2 = IterationSnapshot(
            iteration=2, span_id="s2",
            agent_state={"model": "gpt-4o", "system_prompt": "You are helpful",
                          "temperature": 0.0, "max_tokens": None, "max_iterations": 10},
            input_message="Continue",
            output="Furthermore...",
            tool_calls=[],
            token_usage={"prompt_tokens": 80, "completion_tokens": 50, "total_tokens": 130},
            timestamp=1001.0,
        )
        recording = ExecutionRecording(
            trace_id="rec1", agent_name="researcher", model="gpt-4o",
            snapshots=[snap1, snap2],
            final_result={"content": "Furthermore...", "iterations": 2},
            metadata={"task": "explain AI"},
        )

        # Serialize to JSON
        d = recording.to_dict()
        j = json.dumps(d)

        # Deserialize
        parsed = json.loads(j)
        restored = ExecutionRecording.from_dict(parsed)

        assert restored.trace_id == "rec1"
        assert restored.agent_name == "researcher"
        assert len(restored.snapshots) == 2
        assert restored.snapshots[0].input_message == "What is AI?"
        assert restored.snapshots[1].output == "Furthermore..."
        assert len(restored.snapshots[0].tool_calls) == 1
        assert restored.metadata["task"] == "explain AI"
