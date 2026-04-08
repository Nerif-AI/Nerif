---
sidebar_position: 1
---

# Nerif Model

## Model Classes

### SimpleChatModel

The main chat model class in Nerif. It supports text and multi-modal input, tool calling, structured output, retry, fallback, callbacks, rate limiting, and optional conversation memory.

### Constructor

```python
from nerif.model import SimpleChatModel

model = SimpleChatModel(
    model="gpt-4o",
    default_prompt="You are a helpful assistant.",
    temperature=0.0,
    counter=None,
    max_tokens=None,
    retry_config=None,
    memory=None,
    fallback=None,
    callbacks=None,
    rate_limiter=None,
)
```

### Key methods

- `chat(...)`
- `achat(...)`
- `stream_chat(...)`
- `astream_chat(...)`
- `reset(prompt=None)`
- `set_max_tokens(max_tokens=None)`

### `chat()`

```python
result = model.chat(
    "Summarize this message.",
    append=True,
    tools=None,
    tool_choice=None,
    response_format=None,
    response_model=None,
)
```

Returns either:
- `str`
- `list[ToolCallResult]`
- a validated Pydantic model when `response_model=` is used

### Memory, fallback, and callbacks

- `memory=ConversationMemory(...)` enables managed conversation history
- `fallback=[...]` enables automatic model fallback on transient failures
- `callbacks=CallbackManager()` enables LLM, tool, retry, fallback, and memory event hooks
- `rate_limiter=RateLimiter(...)` enables request throttling

### MultiModalMessage

Use `MultiModalMessage` to build text + image/audio/video requests.

### Tool calling

Use `ToolDefinition` for OpenAI-compatible tool/function calling. Tool calls return `ToolCallResult` objects.

### Structured output

Use `response_format={...}` or `response_model=MyModel` for structured responses.

### Streaming

Use `stream_chat()` or `astream_chat()` for incremental output.
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### Retry Configuration

Configure automatic retry with exponential backoff:

```python
from nerif.model import SimpleChatModel
from nerif.utils import RetryConfig, NO_RETRY, AGGRESSIVE_RETRY

# Custom retry
model = SimpleChatModel(retry_config=RetryConfig(max_retries=5, base_delay=0.5))

# No retry
model_fast = SimpleChatModel(retry_config=NO_RETRY)
```

### Pydantic Structured Output

Use `response_model` for type-safe structured output:

```python
from pydantic import BaseModel
from nerif.model import SimpleChatModel

class City(BaseModel):
    name: str
    country: str
    population: int

model = SimpleChatModel()
city = model.chat("Tell me about Tokyo.", response_model=City)
print(f"{city.name}, {city.country}: {city.population:,}")
```

---

### SimpleEmbeddingModel

A simple model class for embedding text. Converts text strings into numerical vector representations.

Attributes:

- `model (str)`: The name of the embedding model to use.
- `counter (NerifTokenCounter)`: Optional counter for tracking token usage.

Methods:

- `embed(string: str) -> List[float]`: Encodes a string into an embedding vector.
- `aembed(string: str) -> List[float]`: Async version of embed().

Init:
```python
def __init__(
    self,
    model: str = "text-embedding-3-small",
    counter: Optional[NerifTokenCounter] = None,
)
```

Example:

```python
from nerif.model import SimpleEmbeddingModel

embedding_model = SimpleEmbeddingModel()
print(embedding_model.embed("What is the capital of the moon?"))
```

---

### LogitsChatModel

A model for fetching logits (log probabilities) from an LLM. Used internally by Nerif Core for logits-based judgment.

Attributes:

- `model (str)`: The name of the model to use.
- `default_prompt (str)`: The default system prompt for the chat.
- `temperature (float)`: The temperature setting for response generation.
- `counter (NerifTokenCounter)`: Token counter instance.
- `max_tokens (int)`: The maximum number of tokens to generate.

Methods:

- `reset()`: Resets the conversation history.
- `set_max_tokens(max_tokens: None|int)`: Sets the maximum tokens limit.
- `chat(message: str, max_tokens: None|int, logprobs: bool = True, top_logprobs: int = 5) -> Any`:
    Sends a message and gets a response with logit probabilities.

Init:

```python
def __init__(
    self,
    model: str = NERIF_DEFAULT_LLM_MODEL,
    default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
    temperature: float = 0.0,
    counter: Optional[NerifTokenCounter] = None,
    max_tokens: int | None = None,
)
```

Example:

```python
from nerif.model import LogitsChatModel

logits_model = LogitsChatModel()
print(logits_model.chat("What is the capital of the moon?"))
```

---

### OllamaEmbeddingModel

An embedding model for use with Ollama local models.

Attributes:

- `model (str)`: Ollama model name (default: `"ollama/mxbai-embed-large"`).
- `url (str)`: Ollama API URL (default: `"http://localhost:11434/v1/"`).
- `counter (NerifTokenCounter)`: Optional counter for tracking token usage.

Methods:

- `embed(string: str) -> List[float]`: Encodes a string into an embedding vector.

```python
from nerif.model import OllamaEmbeddingModel

model = OllamaEmbeddingModel(model="ollama/mxbai-embed-large")
embedding = model.embed("Hello world")
```

---

### VideoModel

A model for video understanding tasks. Wraps `SimpleChatModel` with `MultiModalMessage` for video input.

Methods:

- `analyze_url(video_url: str, prompt: str = "Describe this video.", max_tokens: int | None = None) -> str`: Analyze a video from a URL.
- `analyze_path(video_path: str, prompt: str = "Describe this video.", max_tokens: int | None = None) -> str`: Analyze a video from a local file.
- `reset()`: Reset conversation history.

```python
from nerif.model import VideoModel

video_model = VideoModel(model="gpt-4o")
result = video_model.analyze_url("https://example.com/video.mp4", "What happens in this video?")
print(result)
```

---

### VisionModel

A vision model for image analysis. For new code, prefer using `SimpleChatModel` with `MultiModalMessage`.

See [Vision Model](./vision-model.md) for detailed documentation.
