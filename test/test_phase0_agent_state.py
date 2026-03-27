import asyncio
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import patch

from nerif.agent import AgentResult, NerifAgent, SharedMemory, Tool


def _make_text_response(content: str, model: str = "gpt-4o", prompt_tokens: int = 10, completion_tokens: int = 5):
    @dataclass
    class _Usage:
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0

    @dataclass
    class _Message:
        role: str = "assistant"
        content: Optional[str] = None
        tool_calls: Optional[list] = None

    @dataclass
    class _Choice:
        index: int = 0
        message: _Message = field(default_factory=_Message)
        finish_reason: str = "stop"

    @dataclass
    class _Response:
        model: str = ""
        choices: List[_Choice] = field(default_factory=list)
        usage: _Usage = field(default_factory=_Usage)

    return _Response(
        model=model,
        choices=[_Choice(message=_Message(content=content))],
        usage=_Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _make_tool_call_response(
    tool_name: str,
    arguments: str,
    tool_call_id: str = "call_1",
    model: str = "gpt-4o",
    prompt_tokens: int = 8,
    completion_tokens: int = 3,
):
    @dataclass
    class _FunctionCall:
        name: str = ""
        arguments: str = ""

    @dataclass
    class _ToolCall:
        id: str = ""
        type: str = "function"
        function: _FunctionCall = field(default_factory=_FunctionCall)

    @dataclass
    class _Usage:
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0

    @dataclass
    class _Message:
        role: str = "assistant"
        content: Optional[str] = None
        tool_calls: Optional[list] = None

    @dataclass
    class _Choice:
        index: int = 0
        message: _Message = field(default_factory=_Message)
        finish_reason: str = "tool_calls"

    @dataclass
    class _Response:
        model: str = ""
        choices: List[_Choice] = field(default_factory=list)
        usage: _Usage = field(default_factory=_Usage)

    tc = _ToolCall(id=tool_call_id, function=_FunctionCall(name=tool_name, arguments=arguments))
    return _Response(
        model=model,
        choices=[_Choice(message=_Message(content=None, tool_calls=[tc]))],
        usage=_Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class TestPhase0AgentState:
    @patch("nerif.model.model.get_model_response")
    def test_run_returns_structured_agent_result(self, mock_response):
        mock_response.side_effect = [
            _make_tool_call_response("math", '{"x": 2, "y": 3}', prompt_tokens=11, completion_tokens=4),
            _make_text_response("5", prompt_tokens=7, completion_tokens=2),
        ]

        agent = NerifAgent(model="gpt-4o", max_iterations=3)
        agent.register_tool(
            Tool(
                name="math",
                description="Add numbers",
                parameters={"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}},
                func=lambda x, y: x + y,
            )
        )

        result = agent.run("What is 2 + 3?")

        assert isinstance(result, AgentResult)
        assert result.content == "5"
        assert result.iterations == 2
        assert result.model == "gpt-4o"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "math"
        assert result.token_usage.prompt_tokens == 18
        assert result.token_usage.completion_tokens == 6
        assert result.token_usage.total_tokens == 24
        assert result.latency_ms >= 0

    @patch("nerif.model.model.get_model_response_async")
    def test_arun_returns_structured_agent_result(self, mock_response_async):
        async def fake_async_response(messages, **kwargs):
            return _make_text_response("async ok", prompt_tokens=9, completion_tokens=2)

        mock_response_async.side_effect = fake_async_response
        agent = NerifAgent(model="gpt-4o")

        result = asyncio.run(agent.arun("Say async ok"))

        assert isinstance(result, AgentResult)
        assert result.content == "async ok"
        assert result.iterations == 1
        assert result.token_usage.total_tokens == 11

    @patch("nerif.model.model.get_model_response")
    def test_snapshot_and_restore_round_trip(self, mock_response):
        mock_response.return_value = _make_text_response("state saved")

        agent = NerifAgent(model="gpt-4o")
        agent.run("Keep this state")

        state = agent.snapshot()
        restored = NerifAgent(model="gpt-4o-mini")
        restored.restore(state.to_dict())

        assert restored.snapshot().to_dict() == state.to_dict()

    @patch("nerif.model.model.get_model_response")
    def test_shared_memory_is_visible_across_agents(self, mock_response):
        mock_response.side_effect = [
            _make_text_response("stored"),
            _make_text_response("recalled"),
        ]

        shared_memory = SharedMemory()
        agent_a = NerifAgent(model="gpt-4o", shared_memory=shared_memory)
        agent_b = NerifAgent(model="gpt-4o", shared_memory=shared_memory)

        result_a = agent_a.run("Remember blue comet.")
        result_b = agent_b.run("What phrase did I ask you to remember?")

        assert result_a.content == "stored"
        assert result_b.content == "recalled"
        assert agent_a.model.messages is agent_b.model.messages
        assert any(message.get("content") == "Remember blue comet." for message in agent_b.history)
