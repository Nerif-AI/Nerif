"""Tests for Phase 3: Communication & Collaboration."""

from unittest.mock import patch

import pytest

from nerif.agent.agent import NerifAgent
from nerif.agent.collaboration import AgentHandoff, AgentMessageBus, SharedWorkspace
from nerif.agent.state import AgentResult


def _mock_result(content: str) -> AgentResult:
    return AgentResult(content=content, latency_ms=50.0, iterations=1, model="gpt-4o")


class TestSharedWorkspace:
    def test_set_and_get(self):
        ws = SharedWorkspace()
        ws.set("findings", "important data")
        assert ws.get("findings") == "important data"

    def test_get_default(self):
        ws = SharedWorkspace()
        assert ws.get("missing") is None
        assert ws.get("missing", "default") == "default"

    def test_keys(self):
        ws = SharedWorkspace()
        ws.set("a", 1)
        ws.set("b", 2)
        assert set(ws.keys()) == {"a", "b"}

    def test_snapshot(self):
        ws = SharedWorkspace()
        ws.set("key", "val")
        snap = ws.snapshot()
        assert snap == {"key": "val"}
        snap["key"] = "modified"
        assert ws.get("key") == "val"  # original not affected

    def test_as_tools(self):
        ws = SharedWorkspace()
        tools = ws.as_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"workspace_write", "workspace_read", "workspace_list"}

    def test_tool_write_and_read(self):
        ws = SharedWorkspace()
        tools = {t.name: t for t in ws.as_tools()}
        tools["workspace_write"].execute(key="test", value="hello")
        result = tools["workspace_read"].execute(key="test")
        assert result == "hello"

    def test_tool_list(self):
        ws = SharedWorkspace()
        tools = {t.name: t for t in ws.as_tools()}
        assert "empty" in tools["workspace_list"].execute().lower()
        ws.set("x", 1)
        assert "x" in tools["workspace_list"].execute()


class TestAgentHandoff:
    def test_basic_handoff(self):
        h = AgentHandoff(from_agent="researcher", to_agent="writer", task="Write article")
        assert h.from_agent == "researcher"
        assert h.to_agent == "writer"

    def test_to_prompt(self):
        h = AgentHandoff(
            from_agent="a",
            to_agent="b",
            task="Analyze data",
            context={"topic": "AI"},
            constraints=["Be concise", "Use examples"],
        )
        prompt = h.to_prompt()
        assert "Analyze data" in prompt
        assert "AI" in prompt
        assert "Be concise" in prompt

    def test_serialization(self):
        h = AgentHandoff(from_agent="a", to_agent="b", task="Test")
        d = h.to_dict()
        h2 = AgentHandoff.from_dict(d)
        assert h2.from_agent == "a"
        assert h2.task == "Test"


class TestAgentMessageBus:
    def test_register_and_send(self):
        bus = AgentMessageBus()
        a1 = NerifAgent(model="gpt-4o")
        a2 = NerifAgent(model="gpt-4o")
        bus.register("agent1", a1)
        bus.register("agent2", a2)

        bus.send("agent1", "agent2", "hello")
        msgs = bus.receive("agent2")
        assert len(msgs) == 1
        assert msgs[0]["from"] == "agent1"
        assert msgs[0]["message"] == "hello"

    def test_receive_clears_mailbox(self):
        bus = AgentMessageBus()
        a1 = NerifAgent(model="gpt-4o")
        bus.register("a1", a1)
        bus.send("other", "a1", "msg1")

        msgs = bus.receive("a1")
        assert len(msgs) == 1
        assert bus.receive("a1") == []  # cleared

    def test_send_to_unregistered_raises(self):
        bus = AgentMessageBus()
        with pytest.raises(ValueError, match="not registered"):
            bus.send("a", "unknown", "hi")

    def test_send_and_run(self):
        bus = AgentMessageBus()
        agent = NerifAgent(model="gpt-4o")
        bus.register("worker", agent)

        with patch.object(agent, "run", return_value=_mock_result("done")):
            result = bus.send_and_run("boss", "worker", "do this")
            assert result.content == "done"

    def test_as_tools(self):
        bus = AgentMessageBus()
        a1 = NerifAgent(model="gpt-4o")
        a2 = NerifAgent(model="gpt-4o")
        bus.register("a1", a1)
        bus.register("a2", a2)

        tools = bus.as_tools("a1")
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"send_message", "check_messages"}

    def test_tool_send_and_check(self):
        bus = AgentMessageBus()
        a1 = NerifAgent(model="gpt-4o")
        a2 = NerifAgent(model="gpt-4o")
        bus.register("a1", a1)
        bus.register("a2", a2)

        tools_a1 = {t.name: t for t in bus.as_tools("a1")}
        tools_a2 = {t.name: t for t in bus.as_tools("a2")}

        # a1 sends to a2
        result = tools_a1["send_message"].execute(to="a2", message="hello from a1")
        assert "sent" in result.lower()

        # a2 checks messages
        msgs = tools_a2["check_messages"].execute()
        assert "hello from a1" in msgs
