"""Trace persistence backends for Nerif observability."""

from __future__ import annotations

import json
import os
import sqlite3
from abc import ABC, abstractmethod
from typing import List, Optional

from nerif.observability.tracing import Trace, TraceSummary


class TraceStore(ABC):
    """Abstract base class for trace persistence backends."""

    @abstractmethod
    def save_trace(self, trace: Trace) -> None:
        """Persist a completed trace."""

    @abstractmethod
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Return a trace by ID, or None if not found."""

    @abstractmethod
    def list_traces(
        self,
        limit: int = 100,
        offset: int = 0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[TraceSummary]:
        """Return summaries, newest first, optionally filtered by time."""

    @abstractmethod
    def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace by ID.  Returns True if found and deleted."""


# ---------------------------------------------------------------------------
# FileTraceStore
# ---------------------------------------------------------------------------


class FileTraceStore(TraceStore):
    """Stores each trace as a JSON file: ``<trace_dir>/trace-<trace_id>.json``."""

    def __init__(self, trace_dir: str = "./nerif_traces") -> None:
        self._dir = trace_dir
        os.makedirs(trace_dir, exist_ok=True)

    def _path(self, trace_id: str) -> str:
        return os.path.join(self._dir, f"trace-{trace_id}.json")

    def save_trace(self, trace: Trace) -> None:
        path = self._path(trace.trace_id)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(trace.to_dict(), fh, indent=2)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        path = self._path(trace_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return Trace.from_dict(data)

    def list_traces(
        self,
        limit: int = 100,
        offset: int = 0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[TraceSummary]:
        summaries: List[TraceSummary] = []

        try:
            entries = os.listdir(self._dir)
        except FileNotFoundError:
            return summaries

        for filename in entries:
            if not (filename.startswith("trace-") and filename.endswith(".json")):
                continue
            path = os.path.join(self._dir, filename)
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue

            trace = Trace.from_dict(data)
            summary = trace.summary()

            if start_time is not None and trace.start_time < start_time:
                continue
            if end_time is not None and trace.start_time > end_time:
                continue

            summaries.append(summary)

        # Sort newest first by start_time
        summaries.sort(key=lambda s: s.start_time, reverse=True)

        return summaries[offset : offset + limit]

    def delete_trace(self, trace_id: str) -> bool:
        path = self._path(trace_id)
        if not os.path.exists(path):
            return False
        os.remove(path)
        return True


# ---------------------------------------------------------------------------
# SQLiteTraceStore
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id       TEXT PRIMARY KEY,
    root_span_name TEXT NOT NULL,
    start_time     REAL NOT NULL,
    end_time       REAL,
    duration_ms    REAL,
    span_count     INTEGER NOT NULL DEFAULT 0,
    total_tokens   INTEGER NOT NULL DEFAULT 0,
    total_cost_usd REAL    NOT NULL DEFAULT 0.0,
    data           TEXT    NOT NULL
);
"""


class SQLiteTraceStore(TraceStore):
    """Stores traces in a SQLite database using the stdlib ``sqlite3`` module."""

    def __init__(self, db_path: str = "./nerif_traces/traces.db") -> None:
        self._db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            conn.commit()

    def save_trace(self, trace: Trace) -> None:
        summary = trace.summary()
        data_json = json.dumps(trace.to_dict())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO traces
                    (trace_id, root_span_name, start_time, end_time,
                     duration_ms, span_count, total_tokens, total_cost_usd, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.trace_id,
                    summary.root_span_name,
                    trace.start_time,
                    trace.end_time,
                    summary.duration_ms,
                    summary.span_count,
                    summary.total_tokens,
                    trace.total_cost_usd,
                    data_json,
                ),
            )
            conn.commit()

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row["data"])
        return Trace.from_dict(data)

    def list_traces(
        self,
        limit: int = 100,
        offset: int = 0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[TraceSummary]:
        clauses: List[str] = []
        params: List[object] = []

        if start_time is not None:
            clauses.append("start_time >= ?")
            params.append(start_time)
        if end_time is not None:
            clauses.append("start_time <= ?")
            params.append(end_time)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT trace_id, root_span_name, start_time, duration_ms,
                   span_count, total_tokens, total_cost_usd
            FROM traces
            {where}
            ORDER BY start_time DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            TraceSummary(
                trace_id=row["trace_id"],
                root_span_name=row["root_span_name"],
                start_time=row["start_time"],
                duration_ms=row["duration_ms"] or 0.0,
                span_count=row["span_count"],
                total_tokens=row["total_tokens"],
                total_cost_usd=row["total_cost_usd"],
            )
            for row in rows
        ]

    def delete_trace(self, trace_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM traces WHERE trace_id = ?", (trace_id,)
            )
            conn.commit()
        return cursor.rowcount > 0
