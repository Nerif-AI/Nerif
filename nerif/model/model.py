import base64
from typing import Any, List, Optional

from ..utils import (
    LOGGER,
    NERIF_DEFAULT_LLM_MODEL,
    OPENAI_MODEL,
    MessageType,
    NerifTokenCounter,
    get_litellm_embedding,
    get_litellm_response,
    get_ollama_response,
    get_sllm_response,
    get_vllm_response,
)


class SimpleChatModel:
    """
    A simple agent class for the Nerif project.
    This class implements a simple chat agent for the Nerif project.
    It uses OpenAI's GPT models to generate responses to user inputs.

    Attributes:
        model (str): The name of the GPT model to use.
        default_prompt (str): The default system prompt for the chat.
        temperature (float): The temperature setting for response generation.
        counter (NerifTokenCounter): Token counter instance.
        messages (List[Any]): The conversation history.
        max_tokens (int): The maximum number of tokens to generate in the response

    Methods:
        reset(prompt=None): Resets the conversation history.
        chat(message, append=False, max_tokens=None|int): Sends a message and gets a response.
    """

    def __init__(
        self,
        model: str = NERIF_DEFAULT_LLM_MODEL,
        default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
        temperature: float = 0.0,
        counter: NerifTokenCounter = None,
        max_tokens: None | int = None,
    ):
        # Set the model, temperature
        self.model = model
        self.temperature = temperature

        # Set the default prompt and initialize the conversation history
        self.default_prompt = default_prompt
        self.messages: List[Any] = [
            {"role": "system", "content": default_prompt},
        ]
        self.counter = counter
        self.agent_max_tokens = max_tokens

    def reset(self, prompt: Optional[str] = None) -> None:
        # Reset the conversation history
        if prompt is None:
            prompt = self.default_prompt

        self.messages: List[Any] = [{"role": "system", "content": prompt}]

    def set_max_tokens(self, max_tokens: None | int = None):
        self.agent_max_tokens = max_tokens

    def chat(self, message: str, append: bool = False, max_tokens: None | int = None) -> str:
        # Append the user's message to the conversation history
        new_message = {"role": "user", "content": message}
        self.messages.append(new_message)

        req_max_tokens = self.agent_max_tokens if max_tokens is None else max_tokens

        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": req_max_tokens,
        }

        if self.counter is not None:
            kwargs["counter"] = self.counter

        LOGGER.debug("requested with message: %s", self.messages)
        LOGGER.debug("arguments of request: %s", kwargs)

        if self.model in OPENAI_MODEL:
            result = get_litellm_response(self.messages, **kwargs)
        elif self.model.startswith("openrouter"):
            result = get_litellm_response(self.messages, **kwargs)
        elif self.model.startswith("ollama"):
            result = get_ollama_response(self.messages, **kwargs)
        elif self.model.startswith("vllm"):
            result = get_vllm_response(self.messages, **kwargs)
        elif self.model.startswith("sllm"):
            result = get_sllm_response(self.messages, **kwargs)
        else:
            raise ValueError(f"Model {self.model} not supported")

        text_result = result.choices[0].message.content
        if append:
            self.messages.append({"role": "system", "content": text_result})
        else:
            self.reset()
        return text_result


class SimpleEmbeddingModel:
    """
    A simple agent for embedding text.

    Attributes:
        model (str): The name of the embedding model to use.
        counter (NerifTokenCounter): Token counter instance.

    Methods:
        embed(string: str) -> List[float]: Encodes a string into an embedding.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        counter: Optional[NerifTokenCounter] = None,
    ):
        self.model = model
        self.counter = counter

    def embed(self, string: str) -> List[float]:
        result = get_litellm_embedding(
            messages=string,
            model=self.model,
            counter=self.counter,
        )

        return result.data[0]["embedding"]


class LogitsChatModel:
    """
    A simple agent for fetching logits from a model.

    Attributes:
        model (str): The name of the model to use.
        default_prompt (str): The default system prompt for the chat.
        temperature (float): The temperature setting for response generation.
        counter (NerifTokenCounter): Token counter instance.
        messages (List[Any]): The conversation history.
        max_tokens (int): The maximum number of tokens to generate in the response

    Methods:
        chat(message, max_tokens=None|int, logprobs=True, top_logprobs=5) -> Any:
            Sends a message and gets a response with logits.
    """

    def __init__(
        self,
        model: str = NERIF_DEFAULT_LLM_MODEL,
        default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
        temperature: float = 0.0,
        counter: Optional[NerifTokenCounter] = None,
        max_tokens: int | None = None,
    ):
        self.model = model
        self.temperature = temperature

        # Set the default prompt and initialize the conversation history
        self.default_prompt = default_prompt
        self.messages: List[Any] = [
            {"role": "system", "content": default_prompt},
        ]
        self.counter = counter
        self.agent_max_tokens = max_tokens

    def reset(self):
        self.messages = [{"role": "system", "content": self.default_prompt}]

    def set_max_tokens(self, max_tokens: None | int = None):
        self.agent_max_tokens = max_tokens

    def chat(
        self,
        message: str,
        max_tokens: int | None = None,
        logprobs: bool = True,
        top_logprobs: int = 5,
    ) -> Any:
        # Append the user's message to the conversation history
        new_message = {"role": "user", "content": message}
        self.messages.append(new_message)

        req_max_tokens = self.agent_max_tokens if max_tokens is None else max_tokens

        if self.model in OPENAI_MODEL:
            result = get_litellm_response(
                self.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=req_max_tokens,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                counter=self.counter,
            )
        else:
            raise ValueError(f"Model {self.model} not supported")

        return result


class VisionModel:
    """
    A simple agent for vision tasks.
    This class implements a vision-capable agent for the Nerif project.
    It uses OpenAI's GPT-4 Vision models to generate responses to user inputs that include images.

    Attributes:
        model (str): The name of the GPT model to use.
        default_prompt (str): The default system prompt for the chat.
        temperature (float): The temperature setting for response generation.
        counter (NerifTokenCounter): Token counter instance.
        max_tokens (int): Maximum tokens to generate in responses.

    Methods:
        append_message(message_type, content): Adds an image or text message to the conversation.
        reset(): Resets the conversation history.
        set_max_tokens(max_tokens): Sets the maximum response token length.
    """

    def __init__(
        self,
        model: str = NERIF_DEFAULT_LLM_MODEL,
        default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
        temperature: float = 0.0,
        counter: Optional[NerifTokenCounter] = None,
        max_tokens: int | None = None,
    ):
        self.model = model
        self.temperature = temperature

        # Set the default prompt and initialize the conversation history
        self.default_prompt = default_prompt
        self.messages: List[Any] = [
            {"role": "system", "content": default_prompt},
        ]
        self.content_cache = []
        self.cost_count = {"input_token_count": 0, "output_token_count": 0}
        self.couter = counter
        self.agent_max_tokens = max_tokens

    def append_message(self, message_type: MessageType, content: str):
        if message_type == MessageType.IMAGE_PATH:
            content = f"data:image/jpeg;base64,{base64.b64encode(open(content, 'rb').read()).decode('utf-8')}"
            self.content_cache.append({"type": "image_url", "image_url": {"url": content}})
        elif message_type == MessageType.IMAGE_URL:
            self.content_cache.append({"type": "image_url", "image_url": {"url": content}})
        elif message_type == MessageType.IMAGE_BASE64:
            self.content_cache.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{content}"},
                }
            )
        elif message_type == MessageType.TEXT:
            self.content_cache.append({"type": "text", "text": content})
        else:
            raise ValueError(f"Message type {message_type} not supported")

    def reset(self):
        self.messages = [{"role": "system", "content": self.default_prompt}]
        self.content_cache = []

    def set_max_tokens(self, max_tokens: None | int = None):
        self.agent_max_tokens = max_tokens

    def chat(
        self,
        input: List[Any] = None,
        append: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        if input is None:
            # combine cache and new message
            content = self.content_cache
        else:
            content = self.messages + input

        message = {
            "role": "user",
            "content": content,
        }
        self.messages.append(message)

        req_max_tokens = self.agent_max_tokens if max_tokens is None else max_tokens

        result = get_litellm_response(
            self.messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=req_max_tokens,
            counter=self.couter,
        )
        text_result = result.choices[0].message.content
        if append:
            self.content_cache.append(text_result)
        else:
            self.reset()
        return text_result
