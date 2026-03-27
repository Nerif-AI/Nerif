"""Orchestration patterns for multi-agent systems."""

from __future__ import annotations

import asyncio
import time as _time
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from .state import AgentResult, TokenUsage

if TYPE_CHECKING:
    from .agent import NerifAgent


class AgentPipeline:
    """Sequential pipeline: each agent's output feeds as input to the next."""

    def __init__(self, stages: List[Tuple[str, "NerifAgent"]]):
        """stages: list of (name, agent) tuples defining the pipeline order."""
        self.stages = stages

    def run(self, message: str) -> AgentResult:
        """Run all stages sequentially. Returns the final AgentResult."""
        current_input = message
        total_usage = TokenUsage()
        all_tool_calls = []
        total_latency = 0.0
        total_iterations = 0
        last_model = ""

        for _name, agent in self.stages:
            result = agent.run(current_input)
            current_input = result.content
            total_usage.add(result.token_usage)
            all_tool_calls.extend(result.tool_calls)
            total_latency += result.latency_ms
            total_iterations += result.iterations
            last_model = result.model

        return AgentResult(
            content=current_input,
            tool_calls=all_tool_calls,
            token_usage=total_usage,
            latency_ms=total_latency,
            iterations=total_iterations,
            model=last_model,
        )

    async def arun(self, message: str) -> AgentResult:
        """Async version of run()."""
        current_input = message
        total_usage = TokenUsage()
        all_tool_calls = []
        total_latency = 0.0
        total_iterations = 0
        last_model = ""

        for _name, agent in self.stages:
            result = await agent.arun(current_input)
            current_input = result.content
            total_usage.add(result.token_usage)
            all_tool_calls.extend(result.tool_calls)
            total_latency += result.latency_ms
            total_iterations += result.iterations
            last_model = result.model

        return AgentResult(
            content=current_input,
            tool_calls=all_tool_calls,
            token_usage=total_usage,
            latency_ms=total_latency,
            iterations=total_iterations,
            model=last_model,
        )


class AgentRouter:
    """Routes messages to the most appropriate sub-agent."""

    def __init__(
        self,
        agents: dict,
        router_model: str = "gpt-4o",
        strategy: str = "llm",
    ):
        self.agents = agents
        self.router_model = router_model
        self.strategy = strategy

    def _select_agent(self, message: str) -> str:
        """Use LLM to select the best agent for the message."""
        from ..model.model import SimpleChatModel

        agent_descriptions = "\n".join(
            f"- {name}: {agent.model.default_prompt[:200]}"
            for name, agent in self.agents.items()
        )
        prompt = (
            f"Given these available agents:\n{agent_descriptions}\n\n"
            f"Which agent should handle this request?\n"
            f'Request: "{message}"\n\n'
            f"Respond with ONLY the agent name, nothing else."
        )
        router = SimpleChatModel(model=self.router_model, temperature=0.0)
        response = router.chat(prompt).strip().lower()

        # Try exact match first, then substring match
        if response in self.agents:
            return response
        for name in self.agents:
            if name.lower() in response:
                return name

        # Fallback to first agent
        return next(iter(self.agents))

    async def _aselect_agent(self, message: str) -> str:
        """Async version of _select_agent."""
        from ..model.model import SimpleChatModel

        agent_descriptions = "\n".join(
            f"- {name}: {agent.model.default_prompt[:200]}"
            for name, agent in self.agents.items()
        )
        prompt = (
            f"Given these available agents:\n{agent_descriptions}\n\n"
            f"Which agent should handle this request?\n"
            f'Request: "{message}"\n\n'
            f"Respond with ONLY the agent name, nothing else."
        )
        router = SimpleChatModel(model=self.router_model, temperature=0.0)
        response = (await router.achat(prompt)).strip().lower()

        if response in self.agents:
            return response
        for name in self.agents:
            if name.lower() in response:
                return name

        return next(iter(self.agents))

    def run(self, message: str) -> AgentResult:
        """Route the message to the selected agent and run it."""
        selected = self._select_agent(message)
        return self.agents[selected].run(message)

    async def arun(self, message: str) -> AgentResult:
        """Async version of run()."""
        selected = await self._aselect_agent(message)
        return await self.agents[selected].arun(message)


class AgentParallel:
    """Run multiple agents concurrently and aggregate results."""

    def __init__(
        self,
        agents: List["NerifAgent"],
        aggregator: Optional[Callable] = None,
    ):
        self.agents = agents
        self.aggregator = aggregator or self._default_aggregator

    @staticmethod
    def _default_aggregator(results: List[AgentResult]) -> str:
        return "\n\n---\n\n".join(r.content for r in results)

    def run(self, message: str) -> AgentResult:
        """Run all agents sequentially (use arun for true parallelism)."""
        started = _time.perf_counter()
        results = [agent.run(message) for agent in self.agents]
        total_usage = TokenUsage()
        all_tool_calls = []
        for r in results:
            total_usage.add(r.token_usage)
            all_tool_calls.extend(r.tool_calls)

        content = self.aggregator(results)
        return AgentResult(
            content=content,
            tool_calls=all_tool_calls,
            token_usage=total_usage,
            latency_ms=(_time.perf_counter() - started) * 1000,
            iterations=sum(r.iterations for r in results),
            model=results[0].model if results else "",
        )

    async def arun(self, message: str) -> AgentResult:
        """Run all agents concurrently via asyncio.gather."""
        started = _time.perf_counter()
        gathered = await asyncio.gather(*[agent.arun(message) for agent in self.agents])
        results = list(gathered)
        total_usage = TokenUsage()
        all_tool_calls = []
        for r in results:
            total_usage.add(r.token_usage)
            all_tool_calls.extend(r.tool_calls)

        content = self.aggregator(results)
        return AgentResult(
            content=content,
            tool_calls=all_tool_calls,
            token_usage=total_usage,
            latency_ms=(_time.perf_counter() - started) * 1000,
            iterations=sum(r.iterations for r in results),
            model=results[0].model if results else "",
        )
