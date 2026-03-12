---
sidebar_position: 7
---

# Agent Framework

Nerif provides a ReAct-style agent framework that combines LLM reasoning with tool execution. The agent automatically calls tools in a loop until it produces a final text response.

## Overview

The agent framework consists of two main classes:

- **`NerifAgent`**: The agent that orchestrates the ReAct loop (Reason → Act → Observe → Repeat).
- **`Tool`**: A wrapper around a Python function that the agent can call.

## NerifAgent

### Initialization

```python
from nerif.agent import NerifAgent

agent = NerifAgent(
    model="gpt-4o",               # LLM model to use
    system_prompt="You are a helpful assistant with access to tools.",
    temperature=0.0,               # Response consistency
    max_tokens=None,               # Max tokens per response
    max_iterations=10,             # Max tool-calling loops before stopping
)
```

**Parameters:**
- `model (str)`: The LLM model name (default: `"gpt-4o"`).
- `system_prompt (str)`: System prompt that guides the agent's behavior.
- `temperature (float)`: Temperature for response generation (default: `0.0`).
- `max_tokens (int | None)`: Maximum tokens per response.
- `max_iterations (int)`: Maximum number of tool-calling iterations (default: `10`). Prevents infinite loops.

### Methods

#### `register_tool(tool: Tool)`

Register a single tool with the agent.

```python
agent.register_tool(my_tool)
```

#### `register_tools(tools: List[Tool])`

Register multiple tools at once.

```python
agent.register_tools([tool_a, tool_b, tool_c])
```

#### `run(message: str) -> str`

Run the agent with a user message. Returns the final text response.

The agent will:
1. Send the message to the LLM with available tools.
2. If the LLM returns tool calls, execute them and feed results back.
3. Repeat until the LLM returns a text response or `max_iterations` is reached.

```python
response = agent.run("What is 15 * 23?")
print(response)
```

#### `reset()`

Reset the conversation history.

#### `history` (property)

Access the full conversation history for inspection.

```python
for msg in agent.history:
    print(msg["role"], msg.get("content", ""))
```

## Tool

A `Tool` wraps a Python function so that the agent can call it.

### Initialization

```python
from nerif.agent import Tool

calculator = Tool(
    name="calculator",
    description="Perform basic arithmetic. Supports +, -, *, /.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression to evaluate, e.g. '2 + 3 * 4'",
            },
        },
        "required": ["expression"],
    },
    func=lambda expression: str(eval(expression)),
)
```

**Parameters:**
- `name (str)`: The tool name (used by the LLM to reference the tool).
- `description (str)`: Human-readable description of what the tool does.
- `parameters (dict)`: JSON Schema describing the function parameters (OpenAI function calling format).
- `func (Callable)`: The Python function to execute when the tool is called.

### Methods

- `to_openai_tool() -> dict`: Convert to OpenAI tool format (used internally).
- `execute(**kwargs) -> Any`: Execute the tool's function with the given arguments.

### `@tool` Decorator

You can also create tools using the decorator syntax:

```python
from nerif.agent import tool

@tool(
    name="lookup_capital",
    description="Look up the capital city of a country.",
    parameters={
        "type": "object",
        "properties": {
            "country": {"type": "string", "description": "The country name"},
        },
        "required": ["country"],
    },
)
def lookup_capital(country: str) -> str:
    capitals = {"France": "Paris", "Germany": "Berlin", "Japan": "Tokyo"}
    return capitals.get(country, f"Unknown capital for {country}")
```

## Complete Example

```python
import json
from nerif.agent import NerifAgent, Tool

# Define tools
calculator = Tool(
    name="calculator",
    description="Perform basic arithmetic. Supports +, -, *, /.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression, e.g. '2 + 3 * 4'",
            },
        },
        "required": ["expression"],
    },
    func=lambda expression: str(eval(expression)),
)

lookup = Tool(
    name="lookup_capital",
    description="Look up the capital city of a country.",
    parameters={
        "type": "object",
        "properties": {
            "country": {
                "type": "string",
                "description": "The country name",
            },
        },
        "required": ["country"],
    },
    func=lambda country: json.dumps(
        {"France": "Paris", "Germany": "Berlin", "Japan": "Tokyo"}.get(
            country, f"Unknown capital for {country}"
        )
    ),
)

# Create agent and register tools
agent = NerifAgent(model="gpt-4o", max_iterations=5)
agent.register_tools([calculator, lookup])

# Run the agent
response = agent.run("What is the capital of France, and what is 15 * 23?")
print("Agent response:", response)
```

## How the ReAct Loop Works

```
User message
    │
    ▼
┌──────────────────┐
│  Send to LLM     │◄──────────────────┐
│  (with tools)    │                   │
└──────┬───────────┘                   │
       │                               │
       ▼                               │
   Text response?  ──Yes──► Return     │
       │                               │
       No (tool calls)                 │
       │                               │
       ▼                               │
┌──────────────────┐                   │
│  Execute tools   │                   │
│  Feed results    │───────────────────┘
│  back to LLM     │
└──────────────────┘
```

The loop continues until:
- The LLM produces a final text response, or
- `max_iterations` is reached (a warning is logged and the last available response is returned).
