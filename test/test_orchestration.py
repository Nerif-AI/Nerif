"""Tests for Phase 2: Orchestration patterns."""

import asyncio
from unittest.mock import AsyncMock, patch

from nerif.agent.agent import NerifAgent
from nerif.agent.orchestration import AgentParallel, AgentPipeline, AgentRouter
from nerif.agent.state import AgentResult, TokenUsage


def _mock_result(content: str, model: str = "gpt-4o") -> AgentResult:
    return AgentResult(
        content=content,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        latency_ms=100.0,
        iterations=1,
        model=model,
    )


class TestAgentPipeline:
    def test_sequential_execution(self):
        a1 = NerifAgent(model="gpt-4o")
        a2 = NerifAgent(model="gpt-4o")

        with patch.object(a1, "run", return_value=_mock_result("step1 output")) as m1, patch.object(
            a2, "run", return_value=_mock_result("step2 output")
        ) as m2:
            pipeline = AgentPipeline([("step1", a1), ("step2", a2)])
            result = pipeline.run("initial input")

            assert result.content == "step2 output"
            m1.assert_called_once_with("initial input")
            m2.assert_called_once_with("step1 output")

    def test_token_usage_aggregated(self):
        a1 = NerifAgent(model="gpt-4o")
        a2 = NerifAgent(model="gpt-4o")

        with patch.object(a1, "run", return_value=_mock_result("out1")), patch.object(
            a2, "run", return_value=_mock_result("out2")
        ):
            pipeline = AgentPipeline([("s1", a1), ("s2", a2)])
            result = pipeline.run("input")
            assert result.token_usage.prompt_tokens == 20
            assert result.token_usage.completion_tokens == 10

    def test_async_pipeline(self):
        a1 = NerifAgent(model="gpt-4o")
        a2 = NerifAgent(model="gpt-4o")

        async def run():
            with patch.object(
                a1, "arun", new_callable=AsyncMock, return_value=_mock_result("async1")
            ), patch.object(a2, "arun", new_callable=AsyncMock, return_value=_mock_result("async2")):
                pipeline = AgentPipeline([("s1", a1), ("s2", a2)])
                result = await pipeline.arun("input")
                assert result.content == "async2"

        asyncio.run(run())


class TestAgentParallel:
    def test_sync_all_agents_called(self):
        a1 = NerifAgent(model="gpt-4o")
        a2 = NerifAgent(model="gpt-4o")

        with patch.object(a1, "run", return_value=_mock_result("result1")), patch.object(
            a2, "run", return_value=_mock_result("result2")
        ):
            parallel = AgentParallel(agents=[a1, a2])
            result = parallel.run("question")

            assert "result1" in result.content
            assert "result2" in result.content

    def test_custom_aggregator(self):
        a1 = NerifAgent(model="gpt-4o")
        a2 = NerifAgent(model="gpt-4o")

        def custom_agg(results):
            return " + ".join(r.content for r in results)

        with patch.object(a1, "run", return_value=_mock_result("A")), patch.object(
            a2, "run", return_value=_mock_result("B")
        ):
            parallel = AgentParallel(agents=[a1, a2], aggregator=custom_agg)
            result = parallel.run("q")
            assert result.content == "A + B"

    def test_async_concurrent(self):
        a1 = NerifAgent(model="gpt-4o")
        a2 = NerifAgent(model="gpt-4o")

        async def run():
            with patch.object(
                a1, "arun", new_callable=AsyncMock, return_value=_mock_result("r1")
            ), patch.object(a2, "arun", new_callable=AsyncMock, return_value=_mock_result("r2")):
                parallel = AgentParallel(agents=[a1, a2])
                result = await parallel.arun("q")
                assert "r1" in result.content
                assert "r2" in result.content

        asyncio.run(run())


class TestAgentRouter:
    def test_routes_to_correct_agent(self):
        code_agent = NerifAgent(model="gpt-4o", system_prompt="You write code")
        math_agent = NerifAgent(model="gpt-4o", system_prompt="You solve math")

        router = AgentRouter(
            agents={"code": code_agent, "math": math_agent},
            router_model="gpt-4o",
        )

        with patch.object(router, "_select_agent", return_value="math"), patch.object(
            math_agent, "run", return_value=_mock_result("42")
        ):
            result = router.run("solve 6*7")
            assert result.content == "42"

    def test_fallback_to_first_agent(self):
        a1 = NerifAgent(model="gpt-4o")
        a2 = NerifAgent(model="gpt-4o")

        router = AgentRouter(agents={"first": a1, "second": a2})

        # _select_agent returns "first" -> routes to first agent
        with patch.object(router, "_select_agent", return_value="first"), patch.object(
            a1, "run", return_value=_mock_result("fallback")
        ):
            result = router.run("anything")
            assert result.content == "fallback"
