"""OpenTelemetry adapter for Nerif traces. Requires nerif[otel]."""
from __future__ import annotations

from typing import Optional

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
        SpanExporter,
    )
    from opentelemetry.trace import StatusCode
except ImportError:
    raise ImportError(
        "OTEL adapter requires opentelemetry. Install with: pip install nerif[otel]"
    )

from .tracing import Span, SpanKind, SpanStatus, Trace

_KIND_MAP = {
    SpanKind.AGENT: otel_trace.SpanKind.INTERNAL,
    SpanKind.LLM: otel_trace.SpanKind.CLIENT,
    SpanKind.TOOL: otel_trace.SpanKind.INTERNAL,
    SpanKind.ORCHESTRATOR: otel_trace.SpanKind.INTERNAL,
    SpanKind.CUSTOM: otel_trace.SpanKind.INTERNAL,
}


class OTelAdapter:
    def __init__(self, service_name: str = "nerif", exporter: Optional[SpanExporter] = None):
        self.provider = TracerProvider()
        processor = SimpleSpanProcessor(exporter or ConsoleSpanExporter())
        self.provider.add_span_processor(processor)
        self.tracer = self.provider.get_tracer(service_name)

    def export_trace(self, trace: Trace) -> None:
        for span in trace.spans:
            self._export_span(span)
        self.provider.force_flush()

    def _export_span(self, span: Span) -> None:
        with self.tracer.start_as_current_span(
            span.name, kind=_KIND_MAP.get(span.kind, otel_trace.SpanKind.INTERNAL),
        ) as otel_span:
            for key, value in span.attributes.items():
                otel_span.set_attribute(f"nerif.{key}", str(value))
            otel_span.set_attribute("nerif.span_kind", span.kind.value)
            if span.token_usage:
                otel_span.set_attribute("nerif.prompt_tokens", span.token_usage.prompt_tokens)
                otel_span.set_attribute("nerif.completion_tokens", span.token_usage.completion_tokens)
                otel_span.set_attribute("nerif.total_tokens", span.token_usage.total_tokens)
            for event in span.events:
                otel_span.add_event(event.name, attributes=event.attributes)
            if span.status == SpanStatus.ERROR:
                otel_span.set_status(StatusCode.ERROR, span.error or "")
            else:
                otel_span.set_status(StatusCode.OK)
