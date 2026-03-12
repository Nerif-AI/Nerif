# 示例 13：带工具调用的 ReAct 智能体

本示例演示 **`NerifAgent`** —— 一个 ReAct 风格的智能体，能够在循环中自动调用工具，直到生成最终的文本回答。

## 核心概念

- **`NerifAgent`**：编排"推理 -> 行动 -> 观察"循环。它将用户消息连同可用工具一起发送给大语言模型，执行工具调用，将结果反馈给模型，并重复此过程直到产生文本响应。
- **`Tool`**：将 Python 函数封装为包含名称、描述和 JSON Schema 参数的工具，以便大语言模型调用。
- **`max_iterations`**：智能体执行工具调用循环的安全上限。

## 代码

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

## ReAct 循环的工作原理

1. **推理**：大语言模型接收用户消息以及所有可用工具的描述。
2. **行动**：如果模型决定使用工具，它会返回工具调用请求（函数名称 + 参数）。
3. **观察**：智能体执行请求的工具并将结果反馈给大语言模型。
4. **重复**：模型处理工具结果，然后继续调用工具或生成最终的文本回答。

循环会在以下情况下终止：
- 模型产生了最终的文本响应，或者
- 达到 `max_iterations` 上限（此时会记录一条警告日志）。

## 适用场景

- **多步推理**：当回答问题需要调用多个工具或链式使用结果时。
- **外部数据**：当大语言模型需要查询数据库、API 或进行其无法原生完成的计算时。
- **自主工作流**：当你希望模型自行决定使用哪些工具以及使用顺序时。

如需不使用智能体循环的手动（单步）工具调用，请参阅[示例 10：工具调用](./example-10.md)。
如需了解智能体 API 的详细信息，请参阅[智能体框架](../components/model/nerif-agent-framework.md)。
