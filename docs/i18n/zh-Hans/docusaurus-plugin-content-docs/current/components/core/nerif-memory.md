---
sidebar_position: 4
---

# 对话记忆

`ConversationMemory` 通过滑动窗口、自动摘要和持久化功能管理对话历史。

## 基本用法

```python
from nerif.memory import ConversationMemory
from nerif.model import SimpleChatModel

memory = ConversationMemory(max_messages=20)
model = SimpleChatModel(memory=memory)

model.chat("Hello!", append=True)
model.chat("What did I just say?", append=True)  # 模型记得上下文
```

## 滑动窗口

按消息数量或 token 数量限制历史记录：

```python
# 保留最近 10 条消息
memory = ConversationMemory(max_messages=10)

# 保留约 4000 个 token 以内
memory = ConversationMemory(max_tokens=4000)
```

## 自动摘要

当超出窗口限制时，较旧的消息将通过 LLM 自动摘要：

```python
memory = ConversationMemory(
    max_messages=10,
    summarize=True,
    summarize_model="gpt-4o-mini",
)
```

## 持久化

保存和加载对话：

```python
memory.save("conversation.json")
loaded = ConversationMemory.load("conversation.json")
model = SimpleChatModel(memory=loaded)
```

## 结合可观测性

通过传入计数器来跟踪摘要成本：

```python
from nerif.utils import NerifTokenCounter

counter = NerifTokenCounter()
memory = ConversationMemory(
    max_messages=10,
    summarize=True,
    counter=counter,  # 跟踪摘要 LLM 调用
)
model = SimpleChatModel(counter=counter, memory=memory)
```

## 参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `max_messages` | `int` | `None` | 保留的最大非系统消息数 |
| `max_tokens` | `int` | `None` | 最大近似 token 数量 |
| `summarize` | `bool` | `False` | 是否对丢弃的消息进行摘要 |
| `summarize_model` | `str` | `gpt-4o-mini` | 用于摘要的模型 |
| `summary_prompt` | `str` | (默认) | 自定义摘要提示词 |
| `counter` | `NerifTokenCounter` | `None` | 跟踪摘要成本 |
