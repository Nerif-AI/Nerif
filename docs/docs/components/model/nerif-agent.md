---
sidebar_position: 1
---

# Nerif Model

## Model Class

### SimpleChatModel

A simple model class for the Nerif project.
This class implements a simple chat model for the Nerif project.
It uses OpenAI's GPT models to generate responses to user inputs.

Attributes:

- `proxy_url (str)`: The URL of the proxy server for API requests.
- `api_key (str)`: The API key for authentication.
- `model (str)`: The name of the GPT model to use.
- `default_prompt (str)`: The default system prompt for the chat.
- `temperature (float)`: The temperature setting for response generation.
- `counter (NerifTokenCounter)`: Token counter instance.
- `messages (List[Any])`: The conversation history.
- `max_tokens (int)`: The maximum number of tokens to generate in the response.

Methods:

- `reset(prompt=None)`: Resets the conversation history.
- `set_max_tokens(max_tokens=None|int)`: Sets the maximum tokens limit.
- `chat(message, append=False, max_tokens=None|int)`: Sends a message and gets a response.

Init:

```python
    def __init__(
        self,
        proxy_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = NERIF_DEFAULT_LLM_MODEL,
        default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
        temperature: float = 0.0,
        counter: NerifTokenCounter = None,
        max_tokens: None | int = None,
    )
```

Example:

```python
import nerif

model = nerif.model.SimpleChatModel()

print(model.chat("What is the capital of the moon?"))
print(model.chat("What is the capital of the moon?", max_tokens=10))
```

### SimpleEmbeddingModel

A simple model class for embedding text in the Nerif project.
This class provides text embedding functionality using OpenAI's embedding models.
It converts text strings into numerical vector representations.

Attributes:

- `proxy_url (str)`: The URL of the proxy server for API requests.
- `api_key (str)`: The API key for authentication.
- `model (str)`: The name of the embedding model to use.
- `counter (NerifTokenCounter)`: Optional counter for tracking token usage.

Methods:

- `encode(string: str) -> List[float]`: Encodes a string into an embedding vector.

Init:
```python
def __init__(
    self,
    proxy_url: Optional[str] = None,
    # base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "text-embedding-3-small",
    counter: Optional[NerifTokenCounter] = None,
):
```

Example:

```python
import nerif

embedding_model = nerif.model.SimpleEmbeddingModel()
print(embedding_model.encode("What is the capital of the moon?"))

```

### LogitsModel
A simple model for fetching logits from a model. This class provides functionality to get logit probabilities along with model responses.

Attributes:

- `proxy_url (str)`: The URL of the proxy server for API requests.
- `api_key (str)`: The API key for authentication.
- `model (str)`: The name of the model to use.
- `default_prompt (str)`: The default system prompt for the chat.
- `temperature (float)`: The temperature setting for response generation.
- `counter (NerifTokenCounter)`: Token counter instance.
- `messages (List[Any])`: The conversation history.
- `max_tokens (int)`: The maximum number of tokens to generate in the response.

Methods:

- `reset()`: Resets the conversation history.
- `set_max_tokens(max_tokens: None|int)`: Sets the maximum tokens limit.
- `chat(message: str, max_tokens: None|int, logprobs: bool = True, top_logprobs: int = 5) -> Any`: 
    Sends a message and gets a response with logits. The logprobs parameter enables logit probabilities, 
    and top_logprobs controls how many top probabilities to return.

Init:

```python
def __init__(
    self,
    proxy_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = NERIF_DEFAULT_LLM_MODEL,
    default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
    temperature: float = 0.0,
    counter: Optional[NerifTokenCounter] = None,
    max_tokens: int | None = None,
):
```

Example:

```python
import nerif

logits_model = nerif.model.LogitsModel()
print(logits_model.chat("What is the capital of the moon?"))
```

### VisionModel
A simple model for vision tasks in the Nerif project.
This class implements a vision-capable model that can process both text and images.
It uses OpenAI's GPT-4 Vision models to generate responses to user inputs that include images.

Attributes:

- `proxy_url (str)`: The URL of the proxy server for API requests.
- `api_key (str)`: The API key for authentication.
- `model (str)`: The name of the GPT model to use.
- `default_prompt (str)`: The default system prompt for the chat.
- `temperature (float)`: The temperature setting for response generation.
- `counter (NerifTokenCounter)`: Token counter instance.
- `max_tokens (int)`: Maximum tokens to generate in responses.

Methods:

- `append_message(message_type, content)`: Adds an image or text message to the conversation.
- `reset()`: Resets the conversation history.
- `set_max_tokens(max_tokens)`: Sets the maximum response token length.
- `chat(input: List[Any], append: bool, max_tokens: int|None) -> str`: Processes the input and returns a response.

Init:

```python
def __init__(
    self,
    proxy_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = NERIF_DEFAULT_LLM_MODEL,
    default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
    temperature: float = 0.0,
    counter: Optional[NerifTokenCounter] = None,
    max_tokens: int | None = None,
):
```

Example:

```python
from nerif.model import MessageType, VisionModel

if __name__ == "__main__":
    vision_model = VisionModel(model="openrouter/openai/gpt-4o-2024-08-06")
    vision_model.append_message(
        MessageType.IMAGE_URL,
        "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png",
    )
    vision_model.append_message(MessageType.TEXT, "what is in this image?")
    response = vision_model.chat()
    print(response)
```