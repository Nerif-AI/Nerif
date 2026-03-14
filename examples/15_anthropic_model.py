"""Example: Using Nerif with Anthropic Claude models.

Requires: ANTHROPIC_API_KEY environment variable.
"""
from nerif.model import SimpleChatModel, MultiModalMessage, ToolDefinition

# Basic chat
model = SimpleChatModel(model="anthropic/claude-3-haiku-20240307")
response = model.chat("What is the capital of France?")
print(f"Response: {response}")

# Tool calling
weather_tool = ToolDefinition(
    name="get_weather",
    description="Get the current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
        },
        "required": ["location"],
    },
)

model2 = SimpleChatModel(model="anthropic/claude-3-haiku-20240307")
result = model2.chat(
    "What's the weather in Paris?",
    tools=[weather_tool],
    tool_choice="auto",
)
print(f"Tool call result: {result}")
