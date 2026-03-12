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

You can get start with [examples](./category/start-with-examples)
:::

## Intro

Using Large Language Models (LLMs) can often yield unpredictable results. They may produce excessively polite responses that, while technically correct, don't always align with the context. When faced with uncertainty, LLMs might display human-like confusion, offering detailed explanations of their thought processes instead of clear answers—an approach that isn't always useful for end-users.

To tackle these challenges, we're excited to introduce Nerif. This tool seamlessly integrates LLMs with Python programming, aiming to set a new standard in AI-driven development. Our goal is not only to match but potentially surpass existing frameworks like Langchain, Dify, and other leading AI agent programming methodologies.

Our objectives with Nerif are clear:

- Empower developers to harness LLMs in their projects exactly as they envision.
- Ensure that LLM outputs are properly formatted into programmable types for seamless integration.
- Introduce innovative metrics like token_cost and pass_rate to help developers refine the quality of their AI agents' prompts.
- Provide multi-modal support (vision, audio) and tool calling for building AI agents.

By focusing on these key points, we aim to significantly enhance the utility and efficiency of LLMs in programming environments, making them more accessible and effective for developers worldwide.

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
│   ├── model/          # SimpleChatModel, LogitsChatModel, VisionModel
│   ├── agent/          # Tool calling and agent framework
│   ├── batch/          # Batch API processing
│   ├── utils/          # Format verification, token counting, logging
│   └── cli/            # Command-line utilities
├── docs/               # Docusaurus documentation site
├── test/               # Test suite
├── examples/           # Usage examples
└── pyproject.toml      # Build configuration
```

