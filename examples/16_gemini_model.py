"""Example: Using Nerif with Google Gemini models.

Requires: GOOGLE_API_KEY environment variable.
"""

from nerif.model import SimpleChatModel, ToolDefinition

# Basic chat
model = SimpleChatModel(model="gemini/gemini-2.0-flash")
response = model.chat("Explain quantum computing in one sentence.")
print(f"Response: {response}")

# Structured output (JSON mode)
model2 = SimpleChatModel(model="gemini/gemini-2.0-flash")
response = model2.chat(
    "List 3 programming languages with their year of creation.",
    response_format={"type": "json_object"},
)
print(f"JSON response: {response}")

# Tool calling
calculator_tool = ToolDefinition(
    name="calculate",
    description="Perform a mathematical calculation",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate"},
        },
        "required": ["expression"],
    },
)

model3 = SimpleChatModel(model="gemini/gemini-2.0-flash")
result = model3.chat(
    "What is 42 * 17?",
    tools=[calculator_tool],
    tool_choice="auto",
)
print(f"Tool call result: {result}")
