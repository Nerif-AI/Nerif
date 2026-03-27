"""Dashboard and export utilities for Nerif observability traces."""

from __future__ import annotations

import csv
import json
from typing import Any, Dict, List, Optional

from nerif.observability.store import TraceStore
from nerif.observability.tracing import Span, Trace, TraceSummary

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _simple_table(headers: List[str], rows: List[List[str]]) -> str:
    """Format a simple ASCII table without external dependencies."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    lines = [sep, header_line, sep]
    for row in rows:
        line = "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TraceDashboard
# ---------------------------------------------------------------------------


class TraceDashboard:
    """Human-readable views over a TraceStore.

    Parameters
    ----------
    store:
        A :class:`~nerif.observability.store.TraceStore` instance to query.
    """

    def __init__(self, store: TraceStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # trace_tree
    # ------------------------------------------------------------------

    def trace_tree(self, trace_id: str) -> str:
        """Return an ASCII tree showing the span hierarchy for *trace_id*.

        Each node shows: ``<name>  [<duration_ms:.1f> ms]  [<total_tokens> tok]``

        Raises
        ------
        KeyError
            If *trace_id* is not found in the store.
        """
        trace = self._store.get_trace(trace_id)
        if trace is None:
            raise KeyError(f"Trace {trace_id!r} not found")

        # Build parent -> children mapping
        children: Dict[Optional[str], List[Span]] = {}
        for span in trace.spans:
            key = span.parent_span_id
            children.setdefault(key, []).append(span)

        lines: List[str] = [f"Trace: {trace_id}"]

        def _render(span_id: Optional[str], prefix: str, is_last: bool) -> None:
            # Retrieve span by id
            span = next((s for s in trace.spans if s.span_id == span_id), None)
            if span is None:
                return

            connector = "└── " if is_last else "├── "
            dur = span.duration_ms()
            dur_str = f"{dur:.1f} ms" if dur is not None else "running"
            tok = span.token_usage.total_tokens if span.token_usage else "-"
            lines.append(f"{prefix}{connector}{span.name}  [{dur_str}]  [{tok} tok]")

            child_list = children.get(span.span_id, [])
            child_prefix = prefix + ("    " if is_last else "│   ")
            for idx, child in enumerate(child_list):
                _render(child.span_id, child_prefix, idx == len(child_list) - 1)

        # Start from roots (spans whose parent_span_id is None)
        roots = children.get(None, [])
        for idx, root in enumerate(roots):
            _render(root.span_id, "", idx == len(roots) - 1)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # cost_summary
    # ------------------------------------------------------------------

    def cost_summary(self, trace_id: Optional[str] = None) -> str:
        """Return an ASCII table aggregating token usage by span name.

        If *trace_id* is given, only spans from that trace are included.
        Otherwise all traces in the store (up to 100) are aggregated.
        """
        if trace_id is not None:
            trace = self._store.get_trace(trace_id)
            if trace is None:
                raise KeyError(f"Trace {trace_id!r} not found")
            traces = [trace]
        else:
            summaries = self._store.list_traces(limit=100)
            traces = []
            for s in summaries:
                t = self._store.get_trace(s.trace_id)
                if t is not None:
                    traces.append(t)

        # Aggregate by span name
        agg: Dict[str, Dict[str, int]] = {}
        for t in traces:
            for span in t.spans:
                name = span.name
                if name not in agg:
                    agg[name] = {"prompt": 0, "completion": 0, "total": 0}
                if span.token_usage is not None:
                    agg[name]["prompt"] += span.token_usage.prompt_tokens
                    agg[name]["completion"] += span.token_usage.completion_tokens
                    agg[name]["total"] += span.token_usage.total_tokens

        headers = ["Span Name", "Prompt Tokens", "Completion Tokens", "Total Tokens"]
        rows = [
            [name, str(vals["prompt"]), str(vals["completion"]), str(vals["total"])]
            for name, vals in sorted(agg.items())
        ]
        return _simple_table(headers, rows)

    # ------------------------------------------------------------------
    # trace_list
    # ------------------------------------------------------------------

    def trace_list(self, limit: int = 20) -> str:
        """Return an ASCII table of recent traces (newest first).

        Columns: trace_id, root_span_name, duration_ms, span_count, total_tokens, cost_usd
        """
        summaries: List[TraceSummary] = self._store.list_traces(limit=limit)
        headers = [
            "trace_id",
            "root_span",
            "duration_ms",
            "spans",
            "total_tokens",
            "cost_usd",
        ]
        rows = [
            [
                s.trace_id,
                s.root_span_name,
                f"{s.duration_ms:.1f}",
                str(s.span_count),
                str(s.total_tokens),
                f"${s.total_cost_usd:.6f}",
            ]
            for s in summaries
        ]
        return _simple_table(headers, rows)

    # ------------------------------------------------------------------
    # agent_stats
    # ------------------------------------------------------------------

    def agent_stats(self, trace_id: str) -> str:
        """Return an ASCII table of per-agent span statistics for *trace_id*."""
        from nerif.observability.tracing import SpanKind

        trace = self._store.get_trace(trace_id)
        if trace is None:
            raise KeyError(f"Trace {trace_id!r} not found")

        agent_spans = [s for s in trace.spans if s.kind == SpanKind.AGENT]
        headers = ["Agent", "Duration (ms)", "Prompt Tokens", "Completion Tokens", "Status"]
        rows = []
        for span in agent_spans:
            dur = span.duration_ms()
            dur_str = f"{dur:.1f}" if dur is not None else "-"
            pt = span.token_usage.prompt_tokens if span.token_usage else 0
            ct = span.token_usage.completion_tokens if span.token_usage else 0
            rows.append([span.name, dur_str, str(pt), str(ct), span.status.value])
        return _simple_table(headers, rows)


# ---------------------------------------------------------------------------
# TraceExporter
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "trace_id",
    "span_id",
    "parent_span_id",
    "name",
    "kind",
    "start_time",
    "end_time",
    "duration_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "status",
    "error",
]


def _span_to_row(span: Span) -> Dict[str, Any]:
    """Convert a span to a flat dict with the CSV column schema."""
    dur = span.duration_ms()
    return {
        "trace_id": span.trace_id,
        "span_id": span.span_id,
        "parent_span_id": span.parent_span_id or "",
        "name": span.name,
        "kind": span.kind.value,
        "start_time": span.start_time,
        "end_time": span.end_time if span.end_time is not None else "",
        "duration_ms": f"{dur:.3f}" if dur is not None else "",
        "prompt_tokens": span.token_usage.prompt_tokens if span.token_usage else 0,
        "completion_tokens": span.token_usage.completion_tokens if span.token_usage else 0,
        "total_tokens": span.token_usage.total_tokens if span.token_usage else 0,
        "status": span.status.value,
        "error": span.error or "",
    }


class TraceExporter:
    """Static export methods for :class:`~nerif.observability.tracing.Trace` objects."""

    @staticmethod
    def to_dict(trace: Trace) -> Dict[str, Any]:
        """Serialize *trace* to a plain Python dict (using the Trace data model)."""
        return trace.to_dict()

    @staticmethod
    def to_json(trace: Trace, indent: int = 2) -> str:
        """Serialize *trace* to a JSON string."""
        return json.dumps(TraceExporter.to_dict(trace), indent=indent)

    @staticmethod
    def to_csv(traces: List[Trace], path: str) -> None:
        """Write one row per span across all *traces* to a CSV file at *path*.

        Columns: trace_id, span_id, parent_span_id, name, kind, start_time,
        end_time, duration_ms, prompt_tokens, completion_tokens, total_tokens,
        status, error
        """
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
            writer.writeheader()
            for trace in traces:
                for span in trace.spans:
                    writer.writerow(_span_to_row(span))

    @staticmethod
    def to_dataframe(traces: List[Trace]) -> Dict[str, List[Any]]:
        """Return a column-oriented dict with the same columns as :meth:`to_csv`.

        The result is compatible with ``pandas.DataFrame(result)`` but has no
        pandas dependency.
        """
        df: Dict[str, List[Any]] = {col: [] for col in _CSV_COLUMNS}
        for trace in traces:
            for span in trace.spans:
                row = _span_to_row(span)
                for col in _CSV_COLUMNS:
                    df[col].append(row[col])
        return df
