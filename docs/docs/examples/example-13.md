# Example 13: ReAct Agent with Tool Calling

This example demonstrates the **`NerifAgent`** — a ReAct-style agent that automatically calls tools in a loop until it produces a final text answer.

## Key Concepts

- **`NerifAgent`**: Orchestrates a Reason → Act → Observe loop. It sends the user message to the LLM with available tools, executes any tool calls, feeds results back, and repeats until a text response is produced.
- **`Tool`**: Wraps a Python function with a name, description, and JSON Schema parameters so the LLM can call it.
- **`max_iterations`**: Safety limit on how many tool-calling loops the agent will perform before stopping.

## Code

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
                "description": "A mathematical expression to evaluate, e.g. '2 + 3 * 4'",
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
        {
            "France": "Paris",
            "Germany": "Berlin",
            "Japan": "Tokyo",
            "Brazil": "Brasilia",
        }.get(country, f"Unknown capital for {country}")
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

1. **Reason**: The LLM receives the user message plus descriptions of all available tools.
2. **Act**: If the LLM decides to use tools, it returns tool call requests (function name + arguments).
3. **Observe**: The agent executes the requested tools and feeds the results back to the LLM.
4. **Repeat**: The LLM processes the tool results and either makes more tool calls or produces a final text answer.

The loop continues until:
- The LLM produces a final text response, or
- `max_iterations` is reached (a warning is logged).

## When to Use

- **Multi-step reasoning**: When answering a question requires calling multiple tools or chaining results.
- **External data**: When the LLM needs to query databases, APIs, or compute values it can't do natively.
- **Autonomous workflows**: When you want the model to decide which tools to use and in what order.

For manual (single-step) tool calling without the agent loop, see [Example 10: Tool Calling](./example-10.md).
For details on the agent API, see [Agent Framework](../components/model/nerif-agent-framework.md).
