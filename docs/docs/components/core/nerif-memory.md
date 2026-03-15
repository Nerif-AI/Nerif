---
sidebar_position: 4
---

# Conversation Memory

`ConversationMemory` manages conversation history with sliding window, auto-summarization, and persistence.

## Basic Usage

```python
from nerif.memory import ConversationMemory
from nerif.model import SimpleChatModel

memory = ConversationMemory(max_messages=20)
model = SimpleChatModel(memory=memory)

model.chat("Hello!", append=True)
model.chat("What did I just say?", append=True)  # Model remembers context
```

## Sliding Window

Limit history by message count or token count:

```python
# Keep last 10 messages
memory = ConversationMemory(max_messages=10)

# Keep within ~4000 tokens
memory = ConversationMemory(max_tokens=4000)
```

## Auto-Summarization

When the window is exceeded, older messages are summarized via LLM:

```python
memory = ConversationMemory(
    max_messages=10,
    summarize=True,
    summarize_model="gpt-4o-mini",
)
```

## Persistence

Save and load conversations:

```python
memory.save("conversation.json")
loaded = ConversationMemory.load("conversation.json")
model = SimpleChatModel(memory=loaded)
```

## Combined with Observability

Track summarization costs by passing a counter:

```python
from nerif.utils import NerifTokenCounter

counter = NerifTokenCounter()
memory = ConversationMemory(
    max_messages=10,
    summarize=True,
    counter=counter,  # Tracks summarization LLM calls
)
model = SimpleChatModel(counter=counter, memory=memory)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_messages` | `int` | `None` | Max non-system messages to retain |
| `max_tokens` | `int` | `None` | Max approximate token count |
| `summarize` | `bool` | `False` | Summarize dropped messages |
| `summarize_model` | `str` | `gpt-4o-mini` | Model for summarization |
| `summary_prompt` | `str` | (default) | Custom summarization prompt |
| `counter` | `NerifTokenCounter` | `None` | Track summarization costs |
