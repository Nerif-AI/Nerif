---
sidebar_position: 7
---

# Agent 框架

Nerif 提供了一个 ReAct 风格的 Agent 框架，支持结构化结果、工具执行、编排辅助、协作原语以及可选的可观测性钩子。

## NerifAgent

### 初始化

```python
from nerif.agent import NerifAgent

agent = NerifAgent(
    model="gpt-4o",
    system_prompt="You are a helpful assistant with access to tools.",
    temperature=0.0,
    max_tokens=None,
    max_iterations=10,
)
```

### 关键方法

#### `run(message: str) -> AgentResult`
#### `arun(message: str) -> AgentResult`

同步和异步执行都会返回结构化的 `AgentResult`，而不是纯文本字符串。

`AgentResult` 包含：
- `content`
- `tool_calls`
- `token_usage`
- `latency_ms`
- `iterations`
- `model`

```python
result = agent.run("What is 15 * 23?")
print(result.content)
print(result.latency_ms)
print(result.token_usage.total_tokens)
```

#### `register_tool(tool: Tool)`
注册单个工具。

#### `register_tools(tools: list[Tool])`
注册多个工具。

#### `snapshot() -> AgentState`
序列化当前 agent 状态。

#### `restore(state: AgentState | dict)`
恢复之前的快照。

#### `as_tool(name: str, description: str) -> Tool`
将一个 agent 暴露为另一个 agent 可调用的工具。

#### `reset()`
重置对话状态。

#### `history`
访问当前消息历史。

## 共享记忆

你可以挂载共享记忆，让多个 agent 读写同一份会话状态。

```python
from nerif.agent import NerifAgent, SharedMemory

shared = SharedMemory()
researcher = NerifAgent(shared_memory=shared, memory_namespace="project")
writer = NerifAgent(shared_memory=shared, memory_namespace="project")
```

## 编排辅助

Nerif 内置了以下编排模式：

- `AgentPipeline`
- `AgentRouter`
- `AgentParallel`

```python
from nerif.agent import AgentPipeline

pipeline = AgentPipeline([
    ("research", researcher),
    ("write", writer),
])

result = pipeline.run("Write a short note about tracing")
print(result.content)
```

## 协作原语

对于更复杂的多 agent 工作流，Nerif 还提供：

- `SharedWorkspace`
- `AgentHandoff`
- `AgentMessageBus`

## Tool

`Tool` 用于包装 Python 函数，使 agent 可以调用它。

```python
from nerif.agent import Tool

calculator = Tool(
    name="calculator",
    description="Perform arithmetic.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
        },
        "required": ["expression"],
    },
    func=lambda expression: str(eval(expression)),
)
```

## 可观测性

如果附加 `CallbackManager`，agent 运行会发出 LLM、tool、fallback、retry 和 memory 事件。如果启用了 `nerif.observability` tracing，agent 和 orchestrator 运行还可以生成 trace。
