"""Tests for async agent and tool support."""

import asyncio
from unittest.mock import AsyncMock, patch

from nerif.agent.agent import NerifAgent
from nerif.agent.tool import Tool, tool
from nerif.model.model import ToolCallResult


class TestToolAsync:
    def test_tool_with_async_func(self):
        async def async_add(a, b):
            return a + b

        t = Tool(name="add", description="Add", parameters={}, func=lambda a, b: a + b, async_func=async_add)
        assert t.async_func is not None

    def test_tool_without_async_func(self):
        t = Tool(name="add", description="Add", parameters={}, func=lambda a, b: a + b)
        assert t.async_func is None

    def test_aexecute_uses_async_func(self):
        calls = []

        async def async_add(a, b):
            calls.append("async")
            return a + b

        t = Tool(name="add", description="Add", parameters={}, func=lambda a, b: a + b, async_func=async_add)
        result = asyncio.run(t.aexecute(a=1, b=2))
        assert result == 3
        assert calls == ["async"]

    def test_aexecute_falls_back_to_sync(self):
        t = Tool(name="add", description="Add", parameters={}, func=lambda a, b: a + b)
        result = asyncio.run(t.aexecute(a=1, b=2))
        assert result == 3

    def test_execute_still_works(self):
        t = Tool(name="add", description="Add", parameters={}, func=lambda a, b: a + b, async_func=None)
        assert t.execute(a=1, b=2) == 3

    def test_tool_decorator_with_async_func(self):
        async def async_impl(x):
            return x * 2

        @tool(name="double", description="Double", parameters={}, async_func=async_impl)
        def double(x):
            return x * 2

        assert isinstance(double, Tool)
        assert double.async_func is async_impl
        assert double.execute(x=5) == 10

    def test_existing_decorator_still_works(self):
        @tool(name="add", description="Add", parameters={"type": "object"})
        def add(a, b):
            return a + b

        assert isinstance(add, Tool)
        assert add.async_func is None
        assert add.execute(a=1, b=2) == 3


def _make_mock_tool_calls(name="get_weather", args='{"city": "Tokyo"}'):
    return [ToolCallResult(id="call_1", name=name, arguments=args)]


class TestNerifAgentAsync:
    def test_arun_text_response(self):
        agent = NerifAgent(model="gpt-4o")
        agent.register_tool(
            Tool(name="get_weather", description="Get weather", parameters={}, func=lambda city: f"Sunny in {city}")
        )

        async def run():
            with patch.object(agent.model, "achat", new_callable=AsyncMock) as mock_achat:
                mock_achat.return_value = "The weather is sunny."
                result = await agent.arun("What's the weather?")
                assert result.content == "The weather is sunny."
                assert result.iterations == 1

        asyncio.run(run())

    def test_arun_tool_call_then_text(self):
        agent = NerifAgent(model="gpt-4o")
        agent.register_tool(
            Tool(name="get_weather", description="Get weather", parameters={}, func=lambda city: f"Sunny in {city}")
        )

        async def run():
            with (
                patch.object(agent.model, "achat", new_callable=AsyncMock) as mock_achat,
                patch.object(agent.model, "_acontinue_after_tools", new_callable=AsyncMock) as mock_continue,
            ):
                mock_achat.return_value = _make_mock_tool_calls()
                mock_continue.return_value = "It's sunny in Tokyo."
                result = await agent.arun("Weather in Tokyo?")
                assert result.content == "It's sunny in Tokyo."
                assert len(result.tool_calls) == 1
                mock_achat.assert_called_once()
                mock_continue.assert_called_once()

        asyncio.run(run())

    def test_arun_uses_async_tool(self):
        calls = []

        async def async_weather(city):
            calls.append("async")
            return f"Rainy in {city}"

        agent = NerifAgent(model="gpt-4o")
        agent.register_tool(
            Tool(
                name="get_weather",
                description="Get weather",
                parameters={},
                func=lambda city: f"Sunny in {city}",
                async_func=async_weather,
            )
        )

        async def run():
            with (
                patch.object(agent.model, "achat", new_callable=AsyncMock) as mock_achat,
                patch.object(agent.model, "_acontinue_after_tools", new_callable=AsyncMock) as mock_continue,
            ):
                mock_achat.return_value = _make_mock_tool_calls(args='{"city": "London"}')
                mock_continue.return_value = "It's rainy."
                await agent.arun("Weather in London?")
                assert calls == ["async"]

        asyncio.run(run())

    def test_arun_max_iterations(self):
        agent = NerifAgent(model="gpt-4o", max_iterations=2)
        agent.register_tool(Tool(name="noop", description="No-op", parameters={}, func=lambda: "ok"))

        async def run():
            with (
                patch.object(agent.model, "achat", new_callable=AsyncMock) as mock_achat,
                patch.object(agent.model, "_acontinue_after_tools", new_callable=AsyncMock) as mock_continue,
            ):
                mock_achat.return_value = [ToolCallResult(id="c1", name="noop", arguments="{}")]
                mock_continue.return_value = [ToolCallResult(id="c2", name="noop", arguments="{}")]
                result = await agent.arun("Do something")
                assert "maximum iterations" in result.content.lower()

        asyncio.run(run())

    def test_arun_parallel_tool_calls(self):
        call_order = []

        async def slow_tool(city):
            call_order.append(city)
            await asyncio.sleep(0.01)
            return f"Weather in {city}"

        agent = NerifAgent(model="gpt-4o")
        agent.register_tool(
            Tool(name="weather", description="W", parameters={}, func=lambda city: "", async_func=slow_tool)
        )

        async def run():
            with (
                patch.object(agent.model, "achat", new_callable=AsyncMock) as mock_achat,
                patch.object(agent.model, "_acontinue_after_tools", new_callable=AsyncMock) as mock_continue,
            ):
                mock_achat.return_value = [
                    ToolCallResult(id="c1", name="weather", arguments='{"city": "A"}'),
                    ToolCallResult(id="c2", name="weather", arguments='{"city": "B"}'),
                ]
                mock_continue.return_value = "Done"
                result = await agent.arun("Weather in A and B")
                assert result.content == "Done"
                assert len(result.tool_calls) == 2
                assert set(call_order) == {"A", "B"}

        asyncio.run(run())

    def test_arun_tool_not_found(self):
        agent = NerifAgent(model="gpt-4o")

        async def run():
            with (
                patch.object(agent.model, "achat", new_callable=AsyncMock) as mock_achat,
                patch.object(agent.model, "_acontinue_after_tools", new_callable=AsyncMock) as mock_continue,
            ):
                mock_achat.return_value = [ToolCallResult(id="c1", name="nonexistent", arguments="{}")]
                mock_continue.return_value = "Fallback response"
                result = await agent.arun("Do X")
                assert result.content == "Fallback response"
                tool_msgs = [m for m in agent.model.messages if m.get("role") == "tool"]
                assert len(tool_msgs) == 1
                assert "not found" in tool_msgs[0]["content"]

        asyncio.run(run())
