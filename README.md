# Nerif

Program with both **Python** and **Natural Language**.

LLMs can be tricky to work with. They sometimes give overly formal responses or get confused when they're unsure, making them challenging to use in real applications.

That's why we built Nerif - a simple tool that connects LLMs with Python code. We want to make it better than existing tools like Langchain and Dify.

Nerif helps you:

- Control exactly how LLMs work in your code
- Convert LLM responses into usable data formats
- Track performance with metrics like cost and success rate

Our goal is to make LLMs easier to use for developers, turning complex AI capabilities into practical programming tools.

## How to install

### Pre-requisite

- Python >= 3.9

### Install

```
pip install nerif
```

## QuickStart

```python
from nerif.core import nerif
from nerif.model import SimpleChatModel

agent = SimpleChatModel()

# Use nerif judge "natural language statement"
if nerif("the sky is blue"):
    print("True")
else:
    # Call a simple agent
    print("No", end=", ")
    print(agent.chat("what is the color of the sky?"))
```

## Documentation

More detailed documentation is on the [official website](https://nerif-ai.com).

## License

Nerif is licensed under the [GNU General Public License v3.0](LICENSE).
