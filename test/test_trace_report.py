import os
import tempfile

import pytest

from nerif.agent.state import TokenUsage
from nerif.observability.tracing import Span, SpanKind, SpanStatus, Trace

try:
    from nerif.observability.report import TraceReport
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


def _make_trace() -> Trace:
    root = Span(
        trace_id="t1", span_id="root", parent_span_id=None,
        name="agent:writer", kind=SpanKind.AGENT,
        start_time=1.0, end_time=4.0,
        status=SpanStatus.OK, attributes={}, events=[],
        token_usage=TokenUsage(500, 300, 800), error=None,
    )
    child = Span(
        trace_id="t1", span_id="c1", parent_span_id="root",
        name="llm:gpt-4o", kind=SpanKind.LLM,
        start_time=1.5, end_time=3.0,
        status=SpanStatus.OK, attributes={}, events=[],
        token_usage=TokenUsage(200, 100, 300), error=None,
    )
    return Trace(
        trace_id="t1", root_span=root, spans=[root, child],
        start_time=1.0, end_time=4.0,
        total_token_usage=TokenUsage(700, 400, 1100),
        total_cost_usd=0.05, metadata={},
    )


@pytest.mark.skipif(not HAS_JINJA2, reason="jinja2 not installed")
def test_generate_html_report():
    trace = _make_trace()
    report = TraceReport(trace)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "report.html")
        result = report.generate(path)
        assert os.path.exists(result)
        with open(result) as f:
            html = f.read()
        assert "agent:writer" in html
        assert "llm:gpt-4o" in html
        assert "mermaid" in html.lower()
