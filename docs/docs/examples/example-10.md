# Example 10: Tool Calling

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
