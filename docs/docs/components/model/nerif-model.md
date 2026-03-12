---
sidebar_position: 1
---

# Nerif Model

## Model Classes

### SimpleChatModel

The main model class in Nerif. Supports text and multi-modal input (images, audio, video) as well as tool calling and structured output (JSON mode).

Attributes:

- `model (str)`: The name of the model to use (default: `NERIF_DEFAULT_LLM_MODEL`).
- `default_prompt (str)`: The default system prompt for the chat.
- `temperature (float)`: The temperature setting for response generation.
- `counter (NerifTokenCounter)`: Token counter instance.
- `messages (List[Any])`: The conversation history.
- `max_tokens (int)`: The maximum number of tokens to generate in the response.

Methods:

- `reset(prompt=None)`: Resets the conversation history. Optionally sets a new system prompt.
- `set_max_tokens(max_tokens=None|int)`: Sets the maximum tokens limit.
- `chat(message, append=False, max_tokens=None, tools=None, tool_choice=None, response_format=None)`: Sends a message and gets a response.

Init:

```python
def __init__(
    self,
    model: str = NERIF_DEFAULT_LLM_MODEL,
    default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
    temperature: float = 0.0,
    counter: NerifTokenCounter = None,
    max_tokens: None | int = None,
)
```

#### `chat()` Method

```python
def chat(
    self,
    message: Union[str, MultiModalMessage],
    append: bool = False,
    max_tokens: None | int = None,
    tools: Optional[List[Union[Dict, ToolDefinition]]] = None,
    tool_choice: Optional[Any] = None,
    response_format: Optional[Any] = None,
) -> Union[str, List[ToolCallResult]]:
```

**Parameters:**
- `message`: Text string or `MultiModalMessage` for multi-modal input.
- `append`: If `True`, keep conversation history; if `False`, reset after response.
- `max_tokens`: Override max tokens for this request.
- `tools`: List of tool definitions for function calling (see [Tool Calling](#tool-calling)).
- `tool_choice`: Tool choice parameter (e.g. `"auto"`, `"none"`, or a specific tool).
- `response_format`: Response format (e.g. `{"type": "json_object"}` for JSON mode).

**Returns:** Text response string, or `List[ToolCallResult]` if tools were called.

Example:

```python
from nerif.model import SimpleChatModel

model = SimpleChatModel()

print(model.chat("What is the capital of the moon?"))
print(model.chat("What is the capital of the moon?", max_tokens=10))
```

---

### MultiModalMessage

Helper class for building multi-modal messages with text, images, audio, and video. Uses a fluent (chainable) API.

Methods:

- `add_text(text: str)`: Add a text part.
- `add_image_url(url: str)`: Add an image from a URL.
- `add_image_path(path: str)`: Add an image from a local file path (auto base64-encoded).
- `add_image_base64(b64: str, media_type="image/jpeg")`: Add a base64-encoded image.
- `add_audio_url(url: str)`: Add audio from a URL.
- `add_audio_path(path: str, format="wav")`: Add audio from a local file (auto base64-encoded).
- `add_audio_base64(b64: str, format="wav")`: Add base64-encoded audio.
- `add_video_url(url: str)`: Add a video from a URL.
- `add_video_path(path: str)`: Add a video from a local file (auto base64-encoded).
- `to_content()`: Returns the list of content parts for the API.

All `add_*` methods return `self`, so calls can be chained:

```python
from nerif.model import SimpleChatModel, MultiModalMessage

model = SimpleChatModel(model="gpt-4o")

msg = (
    MultiModalMessage()
    .add_text("What do you see in this image?")
    .add_image_url("https://example.com/photo.jpg")
)

result = model.chat(msg)
print(result)
```

---

### Tool Calling

Nerif supports OpenAI-compatible tool calling through `ToolDefinition` and `ToolCallResult`.

#### ToolDefinition

Helper for defining tools in OpenAI function calling format.

```python
from nerif.model import SimpleChatModel, ToolDefinition

weather_tool = ToolDefinition(
    name="get_weather",
    description="Get the current weather for a given location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
        "required": ["location"],
    },
)

model = SimpleChatModel(model="gpt-4o")
result = model.chat(
    "What's the weather like in San Francisco?",
    tools=[weather_tool],
    tool_choice="auto",
)
```

#### ToolCallResult

Represents a tool call returned by the model. Returned when the model decides to call a tool instead of generating text.

Attributes:
- `id (str)`: The unique ID for this tool call.
- `name (str)`: The name of the function to call.
- `arguments (str)`: JSON string of the function arguments.

```python
if isinstance(result, list):
    for tc in result:
        print(f"Tool: {tc.name}, Args: {tc.arguments}")
```

---

### Structured Output (JSON Mode)

Use `response_format` to get structured JSON output from the model:

```python
from nerif.model import SimpleChatModel
from nerif.utils import NerifFormat

model = SimpleChatModel(model="gpt-4o")
result = model.chat(
    "List three programming languages with their year of creation. "
    "Respond in JSON format as an array of objects with 'name' and 'year' fields.",
    response_format={"type": "json_object"},
)

# Parse robustly with NerifFormat
parsed = NerifFormat.json_parse(result)
```

---

### SimpleEmbeddingModel

A simple model class for embedding text. Converts text strings into numerical vector representations.

Attributes:

- `model (str)`: The name of the embedding model to use.
- `counter (NerifTokenCounter)`: Optional counter for tracking token usage.

Methods:

- `embed(string: str) -> List[float]`: Encodes a string into an embedding vector.

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
