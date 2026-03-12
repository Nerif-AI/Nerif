import json
import logging
from typing import Any, Dict, List

from ..model.model import SimpleChatModel, ToolCallResult
from .tool import Tool

LOGGER = logging.getLogger("Nerif")


class NerifAgent:
    """
    A simple ReAct-style agent that uses tool calling in a loop.

    The agent:
    1. Sends the user message to the model with available tools
    2. If the model returns tool calls, executes them and feeds results back
    3. Repeats until the model returns a text response or max_iterations is reached

    Attributes:
        model: The underlying SimpleChatModel instance.
        tools: Registered tools available to the agent.
        max_iterations: Maximum number of tool-calling loops.
        history: Full conversation history for inspection.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        system_prompt: str = "You are a helpful assistant with access to tools. Use them when needed to answer questions accurately.",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        max_iterations: int = 10,
    ):
        self.model = SimpleChatModel(
            model=model,
            default_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.tools: Dict[str, Tool] = {}
        self.max_iterations = max_iterations

    def register_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def register_tools(self, tools: List[Tool]) -> None:
        for t in tools:
            self.register_tool(t)

    def _get_tool_dicts(self) -> List[Dict[str, Any]]:
        return [t.to_openai_tool() for t in self.tools.values()]

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

    def run(self, message: str) -> str:
        """
        Run the agent with a user message. Returns the final text response.

        The agent will loop through tool calls until it produces a text response
        or hits max_iterations.
        """
        tool_dicts = self._get_tool_dicts() if self.tools else None

        # Send initial message with append=True to keep history
        result = self.model.chat(
            message,
            append=True,
            tools=tool_dicts,
            tool_choice="auto" if tool_dicts else None,
        )

        for _ in range(self.max_iterations):
            if isinstance(result, str):
                return result

            # result is a list of ToolCallResult
            if not isinstance(result, list):
                return str(result)

            # Execute each tool call and add results to conversation
            for tool_call in result:
                tool_result = self._execute_tool_call(tool_call)
                self.model.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

            # Send tool results back to model
            result = self.model.chat(
                "",  # empty message - the tool results are already in history
                append=True,
                tools=tool_dicts,
                tool_choice="auto" if tool_dicts else None,
            )

        LOGGER.warning("Agent reached max_iterations (%d) without producing a final response", self.max_iterations)
        if isinstance(result, str):
            return result
        return "Agent reached maximum iterations without a final response."

    def reset(self) -> None:
        self.model.reset()

    @property
    def history(self) -> List[Any]:
        return self.model.messages
