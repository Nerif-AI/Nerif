import asyncio
import json
import logging
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..memory import ConversationMemory, SharedMemory
from ..model.model import SimpleChatModel, ToolCallResult
from .state import AgentResult, AgentState, TokenUsage, ToolCallRecord
from .tool import Tool

if TYPE_CHECKING:
    from ..utils.callbacks import CallbackManager

LOGGER = logging.getLogger("Nerif")


class NerifAgent:
    """
    A simple ReAct-style agent that uses tool calling in a loop.

    The agent:
    1. Sends the user message to the model with available tools
    2. If the model returns tool calls, executes them and feeds results back
    3. Repeats until the model returns a text response or max_iterations is reached
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        system_prompt: str = "You are a helpful assistant with access to tools. Use them when needed to answer questions accurately.",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        max_iterations: int = 10,
        memory: Optional[ConversationMemory] = None,
        shared_memory: Optional[SharedMemory] = None,
        memory_namespace: str = "default",
        fallback: Optional[List[str]] = None,
        callbacks: Optional["CallbackManager"] = None,
    ):
        if memory is None and shared_memory is not None:
            memory = ConversationMemory(shared_memory=shared_memory, namespace=memory_namespace)

        self.model = SimpleChatModel(
            model=model,
            default_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            memory=memory,
            fallback=fallback,
            callbacks=callbacks,
        )
        self.tools: Dict[str, Tool] = {}
        self.max_iterations = max_iterations
        self.callbacks = callbacks

    @property
    def shared_memory(self) -> Optional[SharedMemory]:
        if self.model.memory is None:
            return None
        return self.model.memory.shared_memory

    @property
    def memory_namespace(self) -> Optional[str]:
        if self.model.memory is None:
            return None
        return self.model.memory.namespace

    def register_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def register_tools(self, tools: List[Tool]) -> None:
        for t in tools:
            self.register_tool(t)

    def _get_tool_dicts(self) -> List[Dict[str, Any]]:
        return [t.to_openai_tool() for t in self.tools.values()]

    def _usage_from_model(self) -> TokenUsage:
        if getattr(self.model, "last_response", None) is None:
            return TokenUsage()
        return TokenUsage.from_response(self.model.last_response)

    @staticmethod
    def _to_tool_call_records(tool_calls: List[ToolCallResult]) -> List[ToolCallRecord]:
        return [
            ToolCallRecord(
                id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments,
                type="function",
            )
            for tool_call in tool_calls
        ]

    def _build_result(
        self,
        content: str,
        tool_calls: List[ToolCallRecord],
        token_usage: TokenUsage,
        started_at: float,
        iterations: int,
    ) -> AgentResult:
        return AgentResult(
            content=content,
            tool_calls=tool_calls,
            token_usage=token_usage,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            iterations=iterations,
            model=getattr(self.model, "last_response_model", self.model.model),
        )

    def _execute_tool_call(self, tool_call: ToolCallResult) -> str:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            return json.dumps({"error": f"Tool '{tool_call.name}' not found"})

        try:
            args = json.loads(tool_call.arguments)
        except json.JSONDecodeError:
            return json.dumps({"error": f"Invalid JSON arguments: {tool_call.arguments}"})

        try:
            result = tool.execute(**args)
            if isinstance(result, str):
                return result
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _execute_tool_call_async(self, tool_call: ToolCallResult) -> str:
        """Like _execute_tool_call but awaits async tools."""
        tool = self.tools.get(tool_call.name)
        if tool is None:
            return json.dumps({"error": f"Tool '{tool_call.name}' not found"})

        try:
            args = json.loads(tool_call.arguments)
        except json.JSONDecodeError:
            return json.dumps({"error": f"Invalid JSON arguments: {tool_call.arguments}"})

        try:
            result = await tool.aexecute(**args)
            if isinstance(result, str):
                return result
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def snapshot(self) -> AgentState:
        memory_state = None
        if self.model.memory is not None:
            memory_state = self.model.memory.snapshot()

        fallback = []
        if self.model.fallback_config is not None:
            fallback = list(self.model.fallback_config.models[1:])

        return AgentState(
            model=self.model.model,
            system_prompt=self.model.default_prompt,
            temperature=self.model.temperature,
            max_tokens=self.model.agent_max_tokens,
            max_iterations=self.max_iterations,
            fallback=fallback,
            memory_state=memory_state,
            messages=deepcopy(self.model.messages),
            tool_names=sorted(self.tools.keys()),
        )

    def restore(self, state: AgentState | Dict[str, Any]) -> None:
        if isinstance(state, dict):
            state = AgentState.from_dict(state)

        self.model.model = state.model
        self.model.default_prompt = state.system_prompt
        self.model.temperature = state.temperature
        self.model.agent_max_tokens = state.max_tokens
        self.max_iterations = state.max_iterations

        if state.fallback:
            from ..utils.fallback import FallbackConfig

            self.model.fallback_config = FallbackConfig(models=[state.model] + state.fallback)
        else:
            self.model.fallback_config = None

        if state.memory_state is not None:
            if self.model.memory is None:
                self.model.memory = ConversationMemory()
            self.model.memory.restore(state.memory_state)
            self.model.messages = self.model.memory._messages
        else:
            self.model.memory = None
            self.model.messages = deepcopy(state.messages)

    def run(self, message: str) -> AgentResult:
        """
        Run the agent with a user message and return a structured result.

        The agent will loop through tool calls until it produces a text response
        or hits max_iterations.
        """
        tool_dicts = self._get_tool_dicts() if self.tools else None
        total_usage = TokenUsage()
        all_tool_calls: List[ToolCallRecord] = []
        iterations = 0
        started_at = time.perf_counter()

        result = self.model.chat(
            message,
            append=True,
            tools=tool_dicts,
            tool_choice="auto" if tool_dicts else None,
        )
        iterations += 1
        total_usage.add(self._usage_from_model())

        for _ in range(self.max_iterations):
            if isinstance(result, str):
                return self._build_result(result, all_tool_calls, total_usage, started_at, iterations)

            if not isinstance(result, list):
                return self._build_result(str(result), all_tool_calls, total_usage, started_at, iterations)

            all_tool_calls.extend(self._to_tool_call_records(result))

            for tool_call in result:
                tool_result = self._execute_tool_call(tool_call)
                self.model.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

            result = self.model._continue_after_tools(
                tools=tool_dicts,
                tool_choice="auto" if tool_dicts else None,
            )
            iterations += 1
            total_usage.add(self._usage_from_model())

        LOGGER.warning("Agent reached max_iterations (%d) without producing a final response", self.max_iterations)
        content = result if isinstance(result, str) else "Agent reached maximum iterations without a final response."
        return self._build_result(content, all_tool_calls, total_usage, started_at, iterations)

    async def arun(self, message: str) -> AgentResult:
        """Async version of run(). Uses achat() and aexecute().

        When multiple tool calls are returned in a single response,
        they are executed concurrently via asyncio.gather().
        """
        tool_dicts = self._get_tool_dicts() if self.tools else None
        total_usage = TokenUsage()
        all_tool_calls: List[ToolCallRecord] = []
        iterations = 0
        started_at = time.perf_counter()

        result = await self.model.achat(
            message,
            append=True,
            tools=tool_dicts,
            tool_choice="auto" if tool_dicts else None,
        )
        iterations += 1
        total_usage.add(self._usage_from_model())

        for _ in range(self.max_iterations):
            if isinstance(result, str):
                return self._build_result(result, all_tool_calls, total_usage, started_at, iterations)

            if not isinstance(result, list):
                return self._build_result(str(result), all_tool_calls, total_usage, started_at, iterations)

            all_tool_calls.extend(self._to_tool_call_records(result))

            if len(result) > 1:
                tool_results = await asyncio.gather(*[self._execute_tool_call_async(tc) for tc in result])
                for tool_call, tool_result in zip(result, tool_results):
                    self.model.messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": tool_result}
                    )
            else:
                tool_result = await self._execute_tool_call_async(result[0])
                self.model.messages.append(
                    {"role": "tool", "tool_call_id": result[0].id, "content": tool_result}
                )

            result = await self.model._acontinue_after_tools(
                tools=tool_dicts,
                tool_choice="auto" if tool_dicts else None,
            )
            iterations += 1
            total_usage.add(self._usage_from_model())

        LOGGER.warning("Agent reached max_iterations (%d) without producing a final response", self.max_iterations)
        content = result if isinstance(result, str) else "Agent reached maximum iterations without a final response."
        return self._build_result(content, all_tool_calls, total_usage, started_at, iterations)

    def reset(self) -> None:
        self.model.reset()

    @property
    def history(self) -> List[Any]:
        return self.model.messages
