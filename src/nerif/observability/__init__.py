"""Nerif Observability — opt-in tracing, budget, recording, and visualization."""

from .budget import BudgetCallbackHandler, BudgetConfig, BudgetExceededError, BudgetManager
from .dashboard import TraceDashboard, TraceExporter
from .recorder import ExecutionRecording, IterationSnapshot
from .store import FileTraceStore, SQLiteTraceStore, TraceStore
from .tracing import (
    Span,
    SpanEvent,
    SpanKind,
    SpanStatus,
    Trace,
    TraceCollector,
    TraceSummary,
    TracingCallbackHandler,
    end_span,
    get_current_span,
    get_current_trace_id,
    start_span,
)

__all__ = [
    "BudgetCallbackHandler",
    "BudgetConfig",
    "BudgetExceededError",
    "BudgetManager",
    "ExecutionRecording",
    "FileTraceStore",
    "IterationSnapshot",
    "SQLiteTraceStore",
    "Span",
    "SpanEvent",
    "SpanKind",
    "SpanStatus",
    "Trace",
    "TraceCollector",
    "TraceDashboard",
    "TraceExporter",
    "TraceStore",
    "TraceSummary",
    "TracingCallbackHandler",
    "end_span",
    "get_current_span",
    "get_current_trace_id",
    "start_span",
]
