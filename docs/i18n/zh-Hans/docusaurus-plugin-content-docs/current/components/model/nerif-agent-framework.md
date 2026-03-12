---
sidebar_position: 7
---

# Agent 框架

Nerif 提供了一个 ReAct 风格的 Agent 框架，将 LLM 推理与工具执行相结合。Agent 会在循环中自动调用工具，直到生成最终的文本响应。

## 概述

Agent 框架由两个主要类组成：

- **`NerifAgent`**：编排 ReAct 循环（推理 → 行动 → 观察 → 重复）的 Agent。
- **`Tool`**：Python 函数的包装器，使 Agent 能够调用该函数。

## NerifAgent

### 初始化

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

**参数：**
- `model (str)`：LLM 模型名称（默认值：`"gpt-4o"`）。
- `system_prompt (str)`：引导 Agent 行为的系统提示词。
- `temperature (float)`：响应生成的温度（默认值：`0.0`）。
- `max_tokens (int | None)`：每次响应的最大 token 数。
- `max_iterations (int)`：工具调用的最大迭代次数（默认值：`10`）。用于防止无限循环。

### 方法

#### `register_tool(tool: Tool)`

向 Agent 注册单个工具。

```python
agent.register_tool(my_tool)
```

#### `register_tools(tools: List[Tool])`

一次注册多个工具。

```python
agent.register_tools([tool_a, tool_b, tool_c])
```

#### `run(message: str) -> str`

使用用户消息运行 Agent。返回最终的文本响应。

Agent 将：
1. 将消息连同可用工具发送给 LLM。
2. 如果 LLM 返回工具调用，执行这些工具并将结果反馈给 LLM。
3. 重复上述过程，直到 LLM 返回文本响应或达到 `max_iterations`。

```python
response = agent.run("What is 15 * 23?")
print(response)
```

#### `reset()`

重置对话历史。

#### `history`（属性）

访问完整的对话历史记录以进行检查。

```python
for msg in agent.history:
    print(msg["role"], msg.get("content", ""))
```

## Tool

`Tool` 将 Python 函数包装起来，使 Agent 能够调用它。

### 初始化

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

**参数：**
- `name (str)`：工具名称（LLM 通过此名称引用工具）。
- `description (str)`：工具功能的可读描述。
- `parameters (dict)`：描述函数参数的 JSON Schema（OpenAI 函数调用格式）。
- `func (Callable)`：工具被调用时执行的 Python 函数。

### 方法

- `to_openai_tool() -> dict`：转换为 OpenAI 工具格式（内部使用）。
- `execute(**kwargs) -> Any`：使用给定参数执行工具的函数。

### `@tool` 装饰器

你也可以使用装饰器语法创建工具：

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

## 完整示例

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

## ReAct 循环工作原理

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

循环持续进行，直到：
- LLM 生成最终的文本响应，或者
- 达到 `max_iterations`（此时会记录警告信息，并返回最后可用的响应）。
