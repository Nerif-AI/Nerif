"""Execution recording and replay for agent runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IterationSnapshot:
    iteration: int
    span_id: str
    agent_state: Dict[str, Any]
    input_message: str
    output: Optional[str]
    tool_calls: List[Dict[str, Any]]
    token_usage: Dict[str, Any]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "IterationSnapshot":
        return IterationSnapshot(**d)


@dataclass
class ExecutionRecording:
    trace_id: str
    agent_name: str
    model: str
    snapshots: List[IterationSnapshot]
    final_result: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "model": self.model,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "final_result": self.final_result,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExecutionRecording":
        return ExecutionRecording(
            trace_id=d["trace_id"],
            agent_name=d["agent_name"],
            model=d["model"],
            snapshots=[IterationSnapshot.from_dict(s) for s in d.get("snapshots", [])],
            final_result=d.get("final_result", {}),
            metadata=d.get("metadata", {}),
        )
