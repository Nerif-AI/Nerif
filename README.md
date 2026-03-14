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
- Environment variable `OPENAI_API_KEY` or other LLM API keys, for more details, see [here](https://nerif-ai.com/docs/nerif-environment)
- Set default model and embedding model with `NERIF_DEFAULT_LLM_MODEL` and `NERIF_DEFAULT_EMBEDDING_MODEL`, for more details, see [here](https://nerif-ai.com/docs/nerif-environment)

Example:

```bash
export OPENAI_API_KEY=`your_api_key`
export NERIF_DEFAULT_LLM_MODEL=gpt-4o
export NERIF_DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
```

### Install

```bash
pip install nerif
```

Optional feature groups:

```bash
pip install "nerif[img-gen]"
pip install "nerif[asr]"
pip install "nerif[tts]"
pip install "nerif[pydantic]"
```

## QuickStart

```python
from nerif.core import nerif
from nerif.model import SimpleChatModel

model = SimpleChatModel()
# Default model is `gpt-4o`

# Use nerif judge "natural language statement"
if nerif("the sky is blue"):
    print("True")
else:
    # Call a simple model
    print("No", end=", ")
    print(model.chat("what is the color of the sky?"))
```

## v1.1 Features

### Streaming

```python
from nerif.model import SimpleChatModel

model = SimpleChatModel()
for chunk in model.stream_chat("Tell me a story"):
    print(chunk, end="", flush=True)
```

### Async Support

```python
import asyncio
from nerif.model import SimpleChatModel

async def main():
    model = SimpleChatModel()
    result = await model.achat("Hello!")
    print(result)

asyncio.run(main())
```

### Pydantic Structured Output

```python
from pydantic import BaseModel
from nerif.model import SimpleChatModel

class City(BaseModel):
    name: str
    country: str
    population: int

model = SimpleChatModel()
city = model.chat("Tell me about Tokyo.", response_model=City)
print(f"{city.name}: {city.population:,}")
```

### Retry Configuration

```python
from nerif.model import SimpleChatModel
from nerif.utils import RetryConfig

model = SimpleChatModel(retry_config=RetryConfig(max_retries=5))
```

### Optional Embedding

```python
from nerif.core import Nerif

# Works without embedding model
judge = Nerif(model="gpt-4o", embed_model=None)
result = judge.judge("the sky is blue")
```

## Documentation

More detailed documentation is on the [official website](https://nerif-ai.com).

## License

Nerif is licensed under the [GNU General Public License v3.0](LICENSE).
