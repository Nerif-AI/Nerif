import pytest

from nerif.agent.state import TokenUsage
from nerif.observability.tracing import Span, SpanKind, SpanStatus, Trace

try:
    from nerif.observability.otel import OTelAdapter
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False


def _make_trace() -> Trace:
    root = Span(
        trace_id="abcdef0123456789", span_id="1234567890abcdef",
        parent_span_id=None, name="agent:test", kind=SpanKind.AGENT,
        start_time=1.0, end_time=2.0, status=SpanStatus.OK,
        attributes={"key": "val"}, events=[],
        token_usage=TokenUsage(100, 50, 150), error=None,
    )
    return Trace(
        trace_id="abcdef0123456789", root_span=root, spans=[root],
        start_time=1.0, end_time=2.0,
        total_token_usage=TokenUsage(100, 50, 150),
        total_cost_usd=0.01, metadata={},
    )


@pytest.mark.skipif(not HAS_OTEL, reason="opentelemetry not installed")
def test_otel_adapter_export():
    adapter = OTelAdapter(service_name="nerif-test")
    trace = _make_trace()
    adapter.export_trace(trace)
