"""Tests for Phase 1: Agent-as-Tool (NerifAgent.as_tool)."""

import asyncio
from unittest.mock import AsyncMock, patch

from nerif.agent.agent import NerifAgent
from nerif.agent.tool import Tool


class TestAsToolBasic:
    def test_as_tool_returns_tool(self):
        agent = NerifAgent(model="gpt-4o")
        t = agent.as_tool(name="researcher", description="Research a topic")
        assert isinstance(t, Tool)
        assert t.name == "researcher"
        assert t.description == "Research a topic"
        assert t.async_func is not None

    def test_as_tool_has_message_parameter(self):
        agent = NerifAgent(model="gpt-4o")
        t = agent.as_tool(name="researcher", description="Research")
        tool_dict = t.to_openai_tool()
        params = tool_dict["function"]["parameters"]
        assert "message" in params["properties"]
        assert params["required"] == ["message"]


class TestAsToolSync:
    def test_sync_call(self):
        agent = NerifAgent(model="gpt-4o")
        t = agent.as_tool(name="researcher", description="Research")

        with patch.object(agent, "run") as mock_run:
            from nerif.agent.state import AgentResult

            mock_run.return_value = AgentResult(content="Research findings here")
            result = t.execute(message="What is quantum computing?")
            assert result == "Research findings here"
            mock_run.assert_called_once_with("What is quantum computing?")


class TestAsToolAsync:
    def test_async_call(self):
        agent = NerifAgent(model="gpt-4o")
        t = agent.as_tool(name="researcher", description="Research")

        async def run():
            with patch.object(agent, "arun", new_callable=AsyncMock) as mock_arun:
                from nerif.agent.state import AgentResult

                mock_arun.return_value = AgentResult(content="Async findings")
                result = await t.aexecute(message="Async question")
                assert result == "Async findings"
                mock_arun.assert_called_once_with("Async question")

        asyncio.run(run())


class TestNestedAgents:
    def test_agent_calls_agent(self):
        """Parent agent can call child agent via as_tool."""
        child = NerifAgent(model="gpt-4o")
        parent = NerifAgent(model="gpt-4o")
        parent.register_tool(child.as_tool(name="child", description="Child agent"))

        assert "child" in parent.tools
        tool_dicts = parent._get_tool_dicts()
        names = [t["function"]["name"] for t in tool_dicts]
        assert "child" in names
