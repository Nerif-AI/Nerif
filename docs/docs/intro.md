---
sidebar_position: 1
---

# QuickStart

:::tip[TL;DR]

With Nerif, you can program with both "**Human language**" and "**Python**".

Example:
```python
from nerif.core import nerif, nerif_match_string
from nerif.model import SimpleChatModel

model = SimpleChatModel()

# Use nerif to judge "natural language statement"
if nerif("the sky is blue"):
    print("True")
else:
    print("No", end=", ")
    print(model.chat("what is the color of the sky?"))

# Use nerif_match to select from options
choices = ["sunny", "rainy", "cloudy"]
idx = nerif_match_string(selections=choices, text="The weather is warm and bright")
print(f"Best match: {choices[idx]}")
```

You can get started with [examples](./category/start-with-examples)
:::

## Intro

Using Large Language Models (LLMs) can often yield unpredictable results. They may produce excessively polite responses that, while technically correct, don't always align with the context. When faced with uncertainty, LLMs might display human-like confusion, offering detailed explanations of their thought processes instead of clear answers—an approach that isn't always useful for end-users.

To tackle these challenges, we introduce **Nerif**. This tool seamlessly integrates LLMs with Python programming, providing a reliable bridge between natural language and structured code.

### Key Features

- **Natural language judgment** — `nerif("statement")` returns `True` or `False`
- **Option matching** — `nerif_match_string(selections, text)` returns the best match index
- **Format verification** — Extract `int`, `float`, `list`, JSON, and more from LLM outputs
- **Multi-modal input** — Send images, audio, and video alongside text using `MultiModalMessage`
- **Tool calling** — Define tools with `ToolDefinition` and let the model call them
- **Structured output** — Get JSON responses with `response_format={"type": "json_object"}`
- **Agent framework** — `NerifAgent` runs ReAct-style tool-calling loops automatically
- **Token tracking** — Monitor token usage with `NerifTokenCounter`
- **Batch processing** — Process large volumes of requests with the OpenAI-compatible Batch API
- **Streaming responses** — `stream_chat()` yields tokens in real-time for interactive applications
- **Async support** — `achat()`, `aembed()`, `astream_chat()` for concurrent async/await usage
- **Retry with backoff** — Automatic retry with exponential backoff via `RetryConfig`
- **Pydantic output** — `response_model=MyModel` returns validated Pydantic objects directly
- **Optional embedding** — `nerif()` works without embedding model via text fallback
- **Conversation memory** — Sliding window and token-based context management with auto-summarization and persistence
- **Enhanced observability** — Request metrics plus opt-in tracing, trace storage, recording, budget controls, and exports via `nerif.observability`
- **Lightweight RAG** — Abstract vector store interface with built-in NumPy implementation
- **Custom exceptions** — Structured error hierarchy (`NerifError`, `ProviderError`, `FormatError`, `ConfigError`)
- **CLI tools** — `nerif check`, `nerif test-model`, `nerif models` commands
- **Type hints** — `py.typed` marker for full IDE and mypy support

## Installation

```bash
pip install nerif
```

## Project Structure

Nerif uses a standard Python `src/` layout:

```
Nerif/
├── src/nerif/          # Python source package
│   ├── core/           # nerif(), nerif_match(), Nerification
│   ├── model/          # SimpleChatModel, LogitsChatModel, VisionModel, VideoModel
│   ├── agent/          # NerifAgent, Tool - ReAct-style tool calling agent
│   ├── batch/          # Batch API processing
│   ├── utils/          # Format verification, token counting, logging
│   └── cli/            # Command-line utilities
├── docs/               # Docusaurus documentation site
├── test/               # Test suite
├── examples/           # Usage examples
└── pyproject.toml      # Build configuration
```
