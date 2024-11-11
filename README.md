# Nerif

Program with both **Python** and **Natural Language**.

Using Large Language Models (LLMs) can often yield unpredictable results. They may produce excessively polite responses that, while technically correct, don't always align with the context. When faced with uncertainty, LLMs might display human-like confusion, offering detailed explanations of their thought processes instead of clear answers—an approach that isn't always useful for end-users.

To tackle these challenges, we're excited to introduce Nerif. This tool seamlessly integrates LLMs with Python programming, aiming to set a new standard in AI-driven development. Our goal is not only to match but potentially surpass existing frameworks like Langchain, Dify, and other leading AI agent programming methodologies.

Our objectives with Nerif are clear:

- Empower developers to harness LLMs in their projects exactly as they envision.
- Ensure that LLM outputs are properly formatted into programmable types for seamless integration.
- Introduce innovative metrics like token_cost and pass_rate to help developers refine the quality of their AI agents' prompts.

By focusing on these key points, we aim to significantly enhance the utility and efficiency of LLMs in programming environments, making them more accessible and effective for developers worldwide.

## How to install

### Pre-requisite

- Python >= 3.9

### Install

```
pip install nerif
```

## QuickStart

```python
from nerif.agent import SimpleChatAgent
from nerif.core import nerif

# Use nerif judge "natural language statement"
if nerif("the sky is blue"):
    print("True")
else:
    print("No", end=", ")
    # Call a simple agent
    agent = SimpleChatAgent()
    print(agent.chat("what is the color of the sky?"))
```

## Documentation

More detailed documentation is on the [official website](https://nerif-ai.com).

## License

Nerif is licensed under the [GNU General Public License v3.0](LICENSE).
