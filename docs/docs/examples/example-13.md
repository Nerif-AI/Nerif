# Example 13: ReAct Agent with Tool Calling

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
