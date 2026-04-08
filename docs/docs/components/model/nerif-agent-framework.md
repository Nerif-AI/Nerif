---
sidebar_position: 7
---

# Agent Framework

Nerif provides a ReAct-style agent framework with structured results, tool execution, orchestration helpers, collaboration primitives, and optional observability hooks.

## NerifAgent

### Initialization

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

### Key methods

#### `run(message: str) -> AgentResult`
#### `arun(message: str) -> AgentResult`

Both sync and async execution return a structured `AgentResult`, not just plain text.

`AgentResult` includes:
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
Register one tool.

#### `register_tools(tools: list[Tool])`
Register multiple tools.

#### `snapshot() -> AgentState`
Serialize the current agent state.

#### `restore(state: AgentState | dict)`
Restore a prior snapshot.

#### `as_tool(name: str, description: str) -> Tool`
Expose an agent as a tool so another agent can call it.

#### `reset()`
Reset the conversation state.

#### `history`
Access the current message history.

## Shared memory

You can attach shared memory so multiple agents read/write the same conversation state.

```python
from nerif.agent import NerifAgent, SharedMemory

shared = SharedMemory()
researcher = NerifAgent(shared_memory=shared, memory_namespace="project")
writer = NerifAgent(shared_memory=shared, memory_namespace="project")
```

## Orchestration helpers

Nerif includes built-in orchestration patterns:

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

## Collaboration primitives

For more complex multi-agent workflows, Nerif also exposes:

- `SharedWorkspace`
- `AgentHandoff`
- `AgentMessageBus`

## Tool

A `Tool` wraps a Python function so the agent can call it.

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

## Observability

If you attach a `CallbackManager`, agent runs can emit LLM, tool, fallback, retry, and memory events. If `nerif.observability` tracing is active, agent and orchestrator runs can also produce traces.
