# Example 10: Tool Calling

This example demonstrates how to use **tool calling** with `SimpleChatModel`. Tool calling allows the LLM to request execution of functions you define, enabling it to interact with external data and services.

## Key Concepts

- **`ToolDefinition`**: Defines a tool using the OpenAI function calling format, with a name, description, and JSON Schema parameters.
- **`ToolCallResult`**: When the model decides to call a tool, it returns a list of `ToolCallResult` objects instead of a text string. Each contains the tool `name`, `arguments` (as a JSON string), and an `id`.

## Code

```python
from nerif.model import SimpleChatModel, ToolDefinition

# Define a tool
weather_tool = ToolDefinition(
    name="get_weather",
    description="Get the current weather for a given location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
        "required": ["location"],
    },
)

# Create model and call with tools
model = SimpleChatModel(model="gpt-4o")
result = model.chat(
    "What's the weather like in San Francisco?",
    tools=[weather_tool],
    tool_choice="auto",
)

print("Result type:", type(result))
if isinstance(result, list):
    for tc in result:
        print(f"Tool call: {tc.name}({tc.arguments})")
else:
    print("Response:", result)
```

## How It Works

1. You define tools with `ToolDefinition`, specifying the function name, description, and parameter schema.
2. Pass the tools to `model.chat()` via the `tools` parameter.
3. Set `tool_choice="auto"` to let the model decide when to use tools.
4. Check the return type: if the model calls a tool, you get a `List[ToolCallResult]`; otherwise, you get a plain text string.

For automatic tool execution in a loop, see [Example 13: ReAct Agent](./example-13.md).
