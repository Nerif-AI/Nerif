"""Core tracing data model and context propagation for Nerif observability."""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional
from uuid import uuid4

from nerif.agent.state import TokenUsage

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SpanKind(str, Enum):
    """The semantic category of a span."""

    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    ORCHESTRATOR = "orchestrator"
    CUSTOM = "custom"


class SpanStatus(str, Enum):
    """Lifecycle status of a span."""

    RUNNING = "running"
    OK = "ok"
    ERROR = "error"


# ---------------------------------------------------------------------------
# SpanEvent
# ---------------------------------------------------------------------------


@dataclass
class SpanEvent:
    """A timestamped annotation attached to a span."""

    name: str
    timestamp: float = field(default_factory=time.perf_counter)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanEvent":
        return cls(
            name=data["name"],
            timestamp=data.get("timestamp", 0.0),
            attributes=dict(data.get("attributes", {})),
        )


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


@dataclass
class Span:
    """A single unit of work within a trace."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: SpanKind
    start_time: float
    end_time: Optional[float]
    status: SpanStatus
    attributes: Dict[str, Any]
    events: List[SpanEvent]
    token_usage: Optional[TokenUsage]
    error: Optional[str]

    def duration_ms(self) -> Optional[float]:
        """Return duration in milliseconds, or None if the span is still running."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0

    def add_event(self, name: str, **attributes: Any) -> None:
        """Append a timestamped event to this span."""
        self.events.append(SpanEvent(name=name, attributes=dict(attributes)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status.value,
            "attributes": dict(self.attributes),
            "events": [e.to_dict() for e in self.events],
            "token_usage": self.token_usage.to_dict() if self.token_usage is not None else None,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Span":
        token_usage_data = data.get("token_usage")
        token_usage: Optional[TokenUsage] = None
        if token_usage_data is not None:
            token_usage = TokenUsage(**token_usage_data)

        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            name=data["name"],
            kind=SpanKind(data["kind"]),
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            status=SpanStatus(data["status"]),
            attributes=dict(data.get("attributes", {})),
            events=[SpanEvent.from_dict(e) for e in data.get("events", [])],
            token_usage=token_usage,
            error=data.get("error"),
        )


# ---------------------------------------------------------------------------
# TraceSummary
# ---------------------------------------------------------------------------


@dataclass
class TraceSummary:
    """High-level summary of a completed trace."""

    trace_id: str
    root_span_name: str
    start_time: float
    duration_ms: float
    span_count: int
    total_tokens: int
    total_cost_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "root_span_name": self.root_span_name,
            "start_time": self.start_time,
            "duration_ms": self.duration_ms,
            "span_count": self.span_count,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
        }


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


@dataclass
class Trace:
    """A complete trace comprising a root span and all descendant spans."""

    trace_id: str
    root_span: Span
    spans: List[Span]  # flat list, includes root
    start_time: float
    end_time: Optional[float]
    total_token_usage: TokenUsage
    total_cost_usd: float
    metadata: Dict[str, Any]

    def span_tree(self) -> Dict[str, List[Span]]:
        """Return a mapping from parent_span_id -> list of child spans."""
        tree: Dict[str, List[Span]] = {}
        for span in self.spans:
            key = span.parent_span_id or "__root__"
            tree.setdefault(key, []).append(span)
        return tree

    def summary(self) -> TraceSummary:
        end = self.end_time if self.end_time is not None else time.perf_counter()
        duration = (end - self.start_time) * 1000.0
        return TraceSummary(
            trace_id=self.trace_id,
            root_span_name=self.root_span.name,
            start_time=self.start_time,
            duration_ms=duration,
            span_count=len(self.spans),
            total_tokens=self.total_token_usage.total_tokens,
            total_cost_usd=self.total_cost_usd,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "root_span": self.root_span.to_dict(),
            "spans": [s.to_dict() for s in self.spans],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_token_usage": self.total_token_usage.to_dict(),
            "total_cost_usd": self.total_cost_usd,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trace":
        root_span = Span.from_dict(data["root_span"])
        spans = [Span.from_dict(s) for s in data.get("spans", [])]
        token_usage_data = data.get("total_token_usage", {})
        total_token_usage = TokenUsage(**token_usage_data)
        return cls(
            trace_id=data["trace_id"],
            root_span=root_span,
            spans=spans,
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            total_token_usage=total_token_usage,
            total_cost_usd=data.get("total_cost_usd", 0.0),
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Context vars
# ---------------------------------------------------------------------------

_current_span: ContextVar[Optional[Span]] = ContextVar("_current_span", default=None)
_current_trace_id: ContextVar[Optional[str]] = ContextVar("_current_trace_id", default=None)
_current_collector: ContextVar[Optional["TraceCollector"]] = ContextVar(
    "_current_collector", default=None
)


def _generate_id() -> str:
    """Generate a short random hex ID (16 chars)."""
    return uuid4().hex[:16]


def get_current_span() -> Optional[Span]:
    """Return the span currently active in this context, or None."""
    return _current_span.get()


def get_current_trace_id() -> Optional[str]:
    """Return the trace ID currently active in this context, or None."""
    return _current_trace_id.get()


# ---------------------------------------------------------------------------
# _SpanContext — holds span + reset tokens for contextvar cleanup
# ---------------------------------------------------------------------------


@dataclass
class _SpanContext:
    """Internal handle returned by start_span; passed to end_span."""

    span: Span
    _span_token: Token
    _trace_id_token: Token
    _is_root: bool


# ---------------------------------------------------------------------------
# start_span / end_span
# ---------------------------------------------------------------------------


def start_span(
    name: str,
    kind: SpanKind = SpanKind.CUSTOM,
    **attributes: Any,
) -> _SpanContext:
    """Open a new span and push it as the current span in this context.

    If there is already a current span, the new span inherits its trace_id and
    records it as the parent.  Otherwise a new trace_id is generated.
    """
    parent = _current_span.get()

    if parent is not None:
        trace_id = parent.trace_id
        parent_span_id: Optional[str] = parent.span_id
        is_root = False
    else:
        existing_trace_id = _current_trace_id.get()
        trace_id = existing_trace_id if existing_trace_id is not None else _generate_id()
        parent_span_id = None
        is_root = True

    span = Span(
        trace_id=trace_id,
        span_id=_generate_id(),
        parent_span_id=parent_span_id,
        name=name,
        kind=kind,
        start_time=time.perf_counter(),
        end_time=None,
        status=SpanStatus.RUNNING,
        attributes=dict(attributes),
        events=[],
        token_usage=None,
        error=None,
    )

    span_token = _current_span.set(span)
    trace_id_token = _current_trace_id.set(trace_id)

    return _SpanContext(
        span=span,
        _span_token=span_token,
        _trace_id_token=trace_id_token,
        _is_root=is_root,
    )


def end_span(ctx: _SpanContext, error: Optional[Exception] = None) -> Span:
    """Close a span, reset the context, and submit it to the active collector.

    If *ctx* is the root span and there is an active collector, the full trace
    is assembled and finished on the collector.
    """
    span = ctx.span
    span.end_time = time.perf_counter()

    if error is not None:
        span.status = SpanStatus.ERROR
        span.error = str(error)
    else:
        span.status = SpanStatus.OK

    # Restore previous span / trace_id
    _current_span.reset(ctx._span_token)
    _current_trace_id.reset(ctx._trace_id_token)

    # Submit to collector if one is active
    collector = _current_collector.get()
    if collector is not None:
        collector.submit_span(span)
        if ctx._is_root:
            collector.finish_trace(span.trace_id, root_span=span)

    return span


# ---------------------------------------------------------------------------
# TraceCollector
# ---------------------------------------------------------------------------


class TraceCollector:
    """Gathers spans and assembles completed traces.

    Activate/deactivate via the context-var mechanism so that instrumented code
    has zero coupling to the collector instance.
    """

    def __init__(self) -> None:
        # trace_id -> list of spans (in submission order)
        self._active_traces: Dict[str, List[Span]] = {}
        # Completed traces keyed by trace_id
        self.store: Dict[str, Trace] = {}
        self.last_trace_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Activation helpers
    # ------------------------------------------------------------------

    def activate(self) -> Token:
        """Install this collector as the active one and return the reset token."""
        return _current_collector.set(self)

    def deactivate(self, token: Token) -> None:
        """Restore the previous collector (or None) using the reset token."""
        _current_collector.reset(token)

    @contextmanager
    def active(self) -> Iterator["TraceCollector"]:
        """Context manager that activates and deactivates this collector."""
        token = self.activate()
        try:
            yield self
        finally:
            self.deactivate(token)

    # ------------------------------------------------------------------
    # Span / trace handling
    # ------------------------------------------------------------------

    def submit_span(self, span: Span) -> None:
        """Record a completed span."""
        self._active_traces.setdefault(span.trace_id, []).append(span)

    def finish_trace(self, trace_id: str, root_span: Span) -> Trace:
        """Assemble a Trace from all submitted spans for *trace_id*."""
        spans = self._active_traces.pop(trace_id, [])

        # Accumulate token usage across all spans
        total_usage = TokenUsage()
        for s in spans:
            if s.token_usage is not None:
                total_usage.add(s.token_usage)

        trace = Trace(
            trace_id=trace_id,
            root_span=root_span,
            spans=spans,
            start_time=root_span.start_time,
            end_time=root_span.end_time,
            total_token_usage=total_usage,
            total_cost_usd=0.0,
            metadata={},
        )

        self.store[trace_id] = trace
        self.last_trace_id = trace_id
        return trace
