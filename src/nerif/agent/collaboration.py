"""Collaboration primitives for multi-agent systems."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

from .state import AgentResult
from .tool import Tool

if TYPE_CHECKING:
    from .agent import NerifAgent


class SharedWorkspace:
    """Key-value store that agents can read from and write to via tools."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value
        self._history.append({"action": "set", "key": key})

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def keys(self) -> List[str]:
        return list(self._store.keys())

    def snapshot(self) -> Dict[str, Any]:
        return dict(self._store)

    def as_tools(self) -> List[Tool]:
        """Generate Tool objects for reading and writing the workspace."""

        def _write(key: str, value: str) -> str:
            self.set(key, value)
            return f"Stored '{key}'"

        def _read(key: str) -> str:
            val = self.get(key)
            if val is None:
                return f"Key '{key}' not found"
            return str(val)

        def _list_keys() -> str:
            keys = self.keys()
            if not keys:
                return "Workspace is empty"
            return ", ".join(keys)

        write_tool = Tool(
            name="workspace_write",
            description="Write a value to the shared workspace",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The key to store the value under"},
                    "value": {"type": "string", "description": "The value to store"},
                },
                "required": ["key", "value"],
            },
            func=_write,
        )

        read_tool = Tool(
            name="workspace_read",
            description="Read a value from the shared workspace",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The key to read"},
                },
                "required": ["key"],
            },
            func=_read,
        )

        list_tool = Tool(
            name="workspace_list",
            description="List all keys in the shared workspace",
            parameters={"type": "object", "properties": {}},
            func=_list_keys,
        )

        return [write_tool, read_tool, list_tool]


@dataclass
class AgentHandoff:
    """Represents a structured task delegation from one agent to another."""

    from_agent: str
    to_agent: str
    task: str
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert the handoff to a prompt string for the receiving agent."""
        parts = [f"Task: {self.task}"]
        if self.context:
            parts.append(f"Context: {json.dumps(self.context, default=str)}")
        if self.constraints:
            constraints_str = "\n".join(f"- {c}" for c in self.constraints)
            parts.append(f"Constraints:\n{constraints_str}")
        return "\n\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentHandoff":
        return cls(**data)


class AgentMessageBus:
    """Simple message bus for inter-agent communication."""

    def __init__(self) -> None:
        self._agents: Dict[str, "NerifAgent"] = {}
        self._mailboxes: Dict[str, List[Dict[str, Any]]] = {}

    def register(self, name: str, agent: "NerifAgent") -> None:
        self._agents[name] = agent
        self._mailboxes[name] = []

    def send(self, from_name: str, to_name: str, message: str) -> None:
        if to_name not in self._mailboxes:
            raise ValueError(f"Agent '{to_name}' is not registered")
        self._mailboxes[to_name].append({"from": from_name, "message": message})

    def receive(self, name: str) -> List[Dict[str, Any]]:
        msgs = list(self._mailboxes.get(name, []))
        if name in self._mailboxes:
            self._mailboxes[name].clear()
        return msgs

    def send_and_run(self, from_name: str, to_name: str, message: str) -> AgentResult:
        """Send a message to an agent and run it immediately."""
        if to_name not in self._agents:
            raise ValueError(f"Agent '{to_name}' is not registered")
        self.send(from_name, to_name, message)
        return self._agents[to_name].run(message)

    async def asend_and_run(self, from_name: str, to_name: str, message: str) -> AgentResult:
        """Async version of send_and_run."""
        if to_name not in self._agents:
            raise ValueError(f"Agent '{to_name}' is not registered")
        self.send(from_name, to_name, message)
        return await self._agents[to_name].arun(message)

    def as_tools(self, agent_name: str) -> List[Tool]:
        """Generate send/receive tools scoped to a specific agent."""
        other_agents = [n for n in self._agents if n != agent_name]

        def _send_message(to: str, message: str) -> str:
            if to not in self._agents:
                return f"Agent '{to}' not found. Available: {', '.join(other_agents)}"
            self.send(agent_name, to, message)
            return f"Message sent to '{to}'"

        def _check_messages() -> str:
            msgs = self.receive(agent_name)
            if not msgs:
                return "No new messages"
            return "\n".join(f"From {m['from']}: {m['message']}" for m in msgs)

        send_tool = Tool(
            name="send_message",
            description=f"Send a message to another agent. Available agents: {', '.join(other_agents)}",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Name of the agent to send to"},
                    "message": {"type": "string", "description": "The message to send"},
                },
                "required": ["to", "message"],
            },
            func=_send_message,
        )

        receive_tool = Tool(
            name="check_messages",
            description="Check for new messages from other agents",
            parameters={"type": "object", "properties": {}},
            func=_check_messages,
        )

        return [send_tool, receive_tool]
