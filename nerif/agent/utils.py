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
    "gpt-o1",
    "gpt-o1-preview",
    "gpt-o1-mini",
]
OPENAI_EMBEDDING_MODEL: List[str] = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OR_SITE_URL = os.environ.get("OR_SITE_URL")
OR_APP_NAME = os.environ.get("OR_APP_NAME")

OLLAMA_URL = os.environ.get("OLLAMA_URL")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")

VLLM_URL = os.environ.get("VLLM_URL")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY")

SLLM_URL = os.environ.get("SLLM_URL")
SLLM_API_KEY = os.environ.get("SLLM_API_KEY")

NERIF_DEFAULT_LLM_MODEL = os.environ.get("NERIF_DEFAULT_LLM_MODEL", "gpt-4o")
NERIF_DEFAULT_EMBEDDING_MODEL = os.environ.get("NERIF_DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")

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
    elif model.startswith("ollama"):
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
    print(model)
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
    elif model.startswith("ollama"):
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
