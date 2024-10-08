import base64
import logging
import os
from enum import Enum, auto
from typing import Any, List, Optional, Union

import litellm

from .token_counter import NerifTokenCounter

# OpenAI Models
OPENAI_MODEL: List[str] = [
    "gpt-3.5-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-05-13-preview",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-08-06-preview",
    "gpt-4o-2024-09-13",
    "gpt-4o-2024-09-13-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4",
    "gpt-4-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-2024-04-09-preview",
]
OPENAI_EMBEDDING_MODEL: List[str] = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_PROXY_URL = os.environ.get("OPENAI_PROXY_URL")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OR_SITE_URL = os.environ.get("OR_SITE_URL")
OR_APP_NAME = os.environ.get("OR_APP_NAME")

NERIF_DEFAULT_LLM_MODEL = os.environ.get("NERIF_DEFAULT_LLM_MODEL", "gpt-4o")
NERIF_DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "NERIF_DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small"
)

LOGGER = logging.getLogger("Nerif")


class MessageType(Enum):
    IMAGE_PATH = auto()
    IMAGE_URL = auto()
    IMAGE_BASE64 = auto()
    TEXT = auto()


def get_litellm_embedding(
    messages: str,
    model: str = NERIF_DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    counter: Optional[NerifTokenCounter] = None,
) -> Any:
    if model in OPENAI_EMBEDDING_MODEL:
        if api_key is None or api_key == "":
            api_key = OPENAI_API_KEY
        if base_url is None or base_url == "":
            base_url = OPENAI_API_BASE
        kargs = {
            "model": model,
            "input": messages,
            "api_key": api_key,
            "base_url": base_url,
        }
    elif model.startswith("openrouter"):
        kargs = {
            "model": model,
            "input": messages,
        }

    response = litellm.embedding(**kargs)

    if counter is not None:
        counter.set_parser_based_on_model(model)
        counter.count_from_response(response)

    return response


def get_litellm_response(
    messages: List[Any],
    model: str = NERIF_DEFAULT_LLM_MODEL,
    temperature: float = 0,
    max_tokens: int = 300,
    stream: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    logprobs: bool = False,
    top_logprobs: int = 5,
    counter: Optional[NerifTokenCounter] = None,
) -> Any:
    """
    Get a text response from an litellm model.

    Parameters:
    - messages (list): The list of messages to send to the model.
    - model (str): Te name of the OpenAI model to use. Default is "gpt-3.5-turbo".
    - temperature (float): The temperature setting for response generation. Default is 0.
    - max_tokens (int): The maximum number of tokens to generate in the response. Default is 300.
    - stream (bool): Whether to stream the response. Default is False.
    - api_key (str): The API key for accessing the OpenAI API. Default is None.
    - batch_size (int): The number of predictions to make in a single request. Default is 1.

    Returns:
    - list: A list of generated text responses if messages is a list, otherwise a single text response.
    """
    if model in OPENAI_MODEL:
        if api_key is None or api_key == "":
            api_key = OPENAI_API_KEY
        if base_url is None or base_url == "":
            base_url = OPENAI_API_BASE
        kargs = {
            "model": model,
            "messages": messages,
            "api_key": api_key,
            "base_url": base_url,
        }
    elif model.startswith("openrouter"):
        kargs = {
            "model": model,
            "messages": messages,
        }
    else:
        raise ValueError(f"Model {model} not supported")

    kargs["stream"] = stream
    kargs["temperature"] = temperature
    kargs["max_tokens"] = max_tokens

    if logprobs:
        kargs["logprobs"] = logprobs
        kargs["top_logprobs"] = top_logprobs

    responses = litellm.completion(**kargs)

    if counter is not None:
        counter.set_parser_based_on_model(model)
        counter.count_from_response(responses)

    return responses


def get_ollama_response(
    prompt: Union[str, List[str]],
    url: str = "http://localhost:11434/v1/",
    model: str = "llama3.1",
    max_tokens: int = 300,
    temperature: float = 0,
    stream: bool = False,
    api_key: Optional[str] = "ollama",
    counter: Optional[NerifTokenCounter] = None,
) -> Union[str, List[str]]:
    """
    Get a text response from an Ollama model.

    Parameters:
    - prompt (str or list): The input prompt(s) for the model.
    - url (str): The URL of the Ollama API. Default is "http://localhost:11434/api/generate".
    - model (str): The name of the Ollama model. Default is "llama3.1".
    - max_tokens (int): The maximum number of tokens to generate in the response. Default is 300.
    - temperature (float): The temperature setting for response generation. Default is 0.
    - stream (bool): Whether to stream the response. Default is False.
    - api_key (str): The API key for accessing the Ollama API. Default is None.
    - batch_size (int): The number of predictions to make in a single request. Default is 1.

    Returns:
    - str or list: The generated text response(s).
    """

    # todo: support batch ollama inference

    response = get_litellm_response(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        api_key=api_key,
        base_url=url,
    )

    if counter is not None:
        counter.set_parser_based_on_model(model)
        counter.count_from_response(response)

    return response


class SimpleChatAgent:
    """
    A simple agent class for the Nerif project.
    This class implements a simple chat agent for the Nerif project.
    It uses OpenAI's GPT models to generate responses to user inputs.

    Attributes:
        proxy_url (str): The URL of the proxy server for API requests.
        api_key (str): The API key for authentication.
        model (str): The name of the GPT model to use.
        default_prompt (str): The default system prompt for the chat.
        temperature (float): The temperature setting for response generation.
        messages (List[Any]): The conversation history.

    Methods:
        reset(prompt=None): Resets the conversation history.
        chat(message, append=False, max_tokens=300): Sends a message and gets a response.
    """

    def __init__(
        self,
        proxy_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = NERIF_DEFAULT_LLM_MODEL,
        default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
        temperature: float = 0.0,
        counter: NerifTokenCounter = None,
    ):
        # Set the proxy URL and API key
        if proxy_url is None or proxy_url == "":
            proxy_url = OPENAI_PROXY_URL
        elif proxy_url[-1] == "/":
            proxy_url = proxy_url[:-1]
        if api_key is None or api_key == "":
            api_key = OPENAI_API_KEY

        # Set the model, temperature, proxy URL, and API key
        self.model = model
        self.temperature = temperature
        self.proxy_url = proxy_url
        self.api_key = api_key

        # Set the default prompt and initialize the conversation history
        self.default_prompt = default_prompt
        self.messages: List[Any] = [
            {"role": "system", "content": default_prompt},
        ]
        self.counter = counter

    def reset(self, prompt: Optional[str] = None) -> None:
        # Reset the conversation history
        if prompt is None:
            prompt = self.default_prompt

        self.messages: List[Any] = [{"role": "system", "content": prompt}]

    def chat(self, message: str, append: bool = False, max_tokens: int = 300) -> str:
        # Append the user's message to the conversation history
        new_message = {"role": "user", "content": message}
        self.messages.append(new_message)

        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }

        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }

        if self.counter is not None:
            kwargs["counter"] = self.counter

        if self.model.startswith("ollama"):
            # ??? why is here a model_name never used
            # Model name is used to count tokens(price) @2024-10-05
            # model_name = self.model.split("/")[1]

            LOGGER.debug(
                "requested with following:\n\tmessage: <dict> %s </dict> \n\targuments of request: <dict> %s </dict>",
                self.messages,
                kwargs,
            )
            result = get_ollama_response(self.messages, **kwargs)
        elif self.model.startswith("openrouter"):
            LOGGER.debug(
                "requested with following:\n\tmessage: <dict> %s </dict> \n\targuments of request: <dict> %s </dict>",
                self.messages,
                kwargs,
            )
            result = get_litellm_response(self.messages, **kwargs)
        elif self.model in OPENAI_MODEL:
            LOGGER.debug(
                "requested with following:\n\tmessage: <dict> %s </dict> \n\targuments of request: <dict> %s </dict>",
                self.messages,
                kwargs,
            )
            result = get_litellm_response(self.messages, **kwargs)

        else:
            raise ValueError(f"Model {self.model} not supported")

        text_result = result.choices[0].message.content
        if append:
            self.messages.append({"role": "system", "content": text_result})
        else:
            self.reset()
        return text_result


class SimpleEmbeddingAgent:
    # TODO: support ollama embedding model
    """
    A simple agent for embedding text.

    Attributes:
        proxy_url (str): The URL of the proxy server for API requests.
        api_key (str): The API key for authentication.
        model (str): The name of the embedding model to use.

    Methods:
        encode(string: str) -> List[float]: Encodes a string into an embedding.
    """

    def __init__(
        self,
        proxy_url: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        counter: Optional[NerifTokenCounter] = None,
    ):
        if proxy_url is None or proxy_url == "":
            proxy_url = OPENAI_PROXY_URL
        elif proxy_url[-1] == "/":
            proxy_url = proxy_url[:-1]
        if api_key is None or api_key == "":
            api_key = OPENAI_API_KEY

        self.model = model
        self.proxy_url = proxy_url
        self.api_key = api_key
        self.counter = counter

    def encode(self, string: str) -> List[float]:
        result = get_litellm_embedding(
            messages=string,
            model=self.model,
            api_key=self.api_key,
            counter=self.counter,
        )

        return result.data[0]["embedding"]


class LogitsAgent:
    # TODO: support ollama logits model
    """
    A simple agent for fetching logits from a model.

    Attributes:
        proxy_url (str): The URL of the proxy server for API requests.
        api_key (str): The API key for authentication.
        model (str): The name of the model to use.
        default_prompt (str): The default system prompt for the chat.
        temperature (float): The temperature setting for response generation.
        messages (List[Any]): The conversation history.
        cost_count (dict): Tracks token usage for input and output.

    Methods:
        chat(message, max_tokens=300, logprobs=True, top_logprobs=5) -> Any:
            Sends a message and gets a response with logits.
    """

    def __init__(
        self,
        proxy_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = NERIF_DEFAULT_LLM_MODEL,
        default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
        temperature: float = 0.0,
        counter: Optional[NerifTokenCounter] = None,
    ):
        if proxy_url is None or proxy_url == "":
            proxy_url = OPENAI_PROXY_URL
        elif proxy_url[-1] == "/":
            proxy_url = proxy_url[:-1]
        if api_key is None or api_key == "":
            api_key = OPENAI_API_KEY

        self.model = model
        self.proxy_url = proxy_url
        self.api_key = api_key
        self.temperature = temperature

        # Set the default prompt and initialize the conversation history
        self.default_prompt = default_prompt
        self.messages: List[Any] = [
            {"role": "system", "content": default_prompt},
        ]
        self.counter = counter

    def reset(self):
        self.messages = [{"role": "system", "content": self.default_prompt}]

    def chat(
        self,
        message: str,
        max_tokens: int = 300,
        logprobs: bool = True,
        top_logprobs: int = 5,
    ) -> Any:
        # Append the user's message to the conversation history
        new_message = {"role": "user", "content": message}
        self.messages.append(new_message)

        if self.model in OPENAI_MODEL:
            result = get_litellm_response(
                self.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                counter=self.counter,
            )
        else:
            raise ValueError(f"Model {self.model} not supported")

        return result


class VisionAgent:
    """
    A simple agent for vision tasks.
    """

    def __init__(
        self,
        proxy_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = NERIF_DEFAULT_LLM_MODEL,
        default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
        temperature: float = 0.0,
        counter: Optional[NerifTokenCounter] = None,
    ):
        if proxy_url is None or proxy_url == "":
            proxy_url = OPENAI_PROXY_URL
        elif proxy_url[-1] == "/":
            proxy_url = proxy_url[:-1]
        if api_key is None or api_key == "":
            api_key = OPENAI_API_KEY

        self.model = model
        self.proxy_url = proxy_url
        self.api_key = api_key
        self.temperature = temperature

        # Set the default prompt and initialize the conversation history
        self.default_prompt = default_prompt
        self.messages: List[Any] = [
            {"role": "system", "content": default_prompt},
        ]
        self.content_cache = []
        self.cost_count = {"input_token_count": 0, "output_token_count": 0}
        self.couter = counter

    def append_message(self, message_type: MessageType, content: str):
        if message_type == MessageType.IMAGE_PATH:
            content = f"data:image/jpeg;base64,{base64.b64encode(open(content, 'rb').read()).decode('utf-8')}"
            self.content_cache.append(
                {"type": "image_url", "image_url": {"url": content}}
            )
        elif message_type == MessageType.IMAGE_URL:
            self.content_cache.append(
                {"type": "image_url", "image_url": {"url": content}}
            )
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

    def chat(
        self, input: List[Any] = None, append: bool = False, max_tokens: int = 1000
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

        result = get_litellm_response(
            self.messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            counter=self.couter,
        )
        text_result = result.choices[0].message.content
        if append:
            self.content_cache.append(text_result)
        else:
            self.reset()
        return text_result
