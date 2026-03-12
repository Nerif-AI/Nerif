# 示例 10：工具调用

本示例演示如何在 `SimpleChatModel` 中使用**工具调用**。工具调用允许大语言模型请求执行你定义的函数，使其能够与外部数据和服务进行交互。

## 核心概念

- **`ToolDefinition`**：使用 OpenAI 函数调用格式定义工具，包含名称、描述和 JSON Schema 参数。
- **`ToolCallResult`**：当模型决定调用工具时，它会返回一个 `ToolCallResult` 对象列表，而不是文本字符串。每个对象包含工具的 `name`（名称）、`arguments`（参数，JSON 字符串格式）和 `id`。

## 代码

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

## 工作原理

1. 使用 `ToolDefinition` 定义工具，指定函数名称、描述和参数模式。
2. 通过 `tools` 参数将工具传递给 `model.chat()`。
3. 设置 `tool_choice="auto"` 让模型自行决定何时使用工具。
4. 检查返回类型：如果模型调用了工具，你会得到 `List[ToolCallResult]`；否则会得到纯文本字符串。

如需自动循环执行工具调用，请参阅 [示例 13：ReAct 智能体](./example-13.md)。
