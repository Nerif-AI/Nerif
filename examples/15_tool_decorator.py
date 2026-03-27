"""Example: Using the @tool() decorator vs Tool class for agent tools."""

from nerif.agent import Tool, tool

# Method 1: Using Tool class directly
add_tool = Tool(
    name="add",
    description="Add two numbers",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        "required": ["a", "b"],
    },
    func=lambda a, b: a + b,
)


# Method 2: Using @tool() decorator
@tool(
    name="multiply",
    description="Multiply two numbers",
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "number", "description": "First number"},
            "y": {"type": "number", "description": "Second number"},
        },
        "required": ["x", "y"],
    },
)
def multiply(x, y):
    return x * y


# Both produce Tool instances
print(f"add_tool type: {type(add_tool)}")
print(f"multiply type: {type(multiply)}")

# Both can be used the same way
print(f"add(2, 3) = {add_tool.execute(a=2, b=3)}")
print(f"multiply(4, 5) = {multiply.execute(x=4, y=5)}")

# Both produce OpenAI-compatible tool definitions
print(f"add_tool schema: {add_tool.to_openai_tool()}")
print(f"multiply schema: {multiply.to_openai_tool()}")
