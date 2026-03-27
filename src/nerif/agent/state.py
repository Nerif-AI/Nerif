from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCallRecord:
    name: str
    arguments: str = ""
    id: Optional[str] = None
    type: Optional[str] = "function"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallRecord":
        return cls(**data)


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)

    def add(self, other: "TokenUsage") -> "TokenUsage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        return self

    @classmethod
    def from_response(cls, response: Any) -> "TokenUsage":
        usage = getattr(response, "usage", None)
        if usage is None:
            return cls()

        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0) or 0
            completion_tokens = usage.get("completion_tokens", 0) or 0
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0

        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


@dataclass
class AgentResult:
    content: str
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    iterations: int = 1
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
            "token_usage": self.token_usage.to_dict(),
            "latency_ms": self.latency_ms,
            "iterations": self.iterations,
            "model": self.model,
        }


@dataclass
class AgentState:
    model: str
    system_prompt: str
    temperature: float
    max_tokens: Optional[int]
    max_iterations: int
    fallback: List[str] = field(default_factory=list)
    memory_state: Optional[Dict[str, Any]] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        return cls(
            model=data["model"],
            system_prompt=data["system_prompt"],
            temperature=data["temperature"],
            max_tokens=data.get("max_tokens"),
            max_iterations=data.get("max_iterations", 10),
            fallback=list(data.get("fallback", [])),
            memory_state=deepcopy(data.get("memory_state")),
            messages=deepcopy(data.get("messages", [])),
            tool_names=list(data.get("tool_names", [])),
        )
