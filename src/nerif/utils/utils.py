import logging
import os
from enum import Enum, auto
from typing import Any, List, Optional, Union

import litellm
import numpy as np
from openai import OpenAI

from .token_counter import NerifTokenCounter

# Environment variables

# OpenAI configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_PROXY_URL = os.environ.get("OPENAI_PROXY_URL")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")

# Default model settings
NERIF_DEFAULT_LLM_MODEL = os.environ.get("NERIF_DEFAULT_LLM_MODEL", "gpt-4o")
NERIF_DEFAULT_EMBEDDING_MODEL = os.environ.get("NERIF_DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")

# OpenRouter configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OR_SITE_URL = os.environ.get("OR_SITE_URL")
OR_APP_NAME = os.environ.get("OR_APP_NAME")

# Ollama configuration
OLLAMA_URL = os.environ.get("OLLAMA_URL")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")

# vLLM configuration
VLLM_URL = os.environ.get("VLLM_URL")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY")

# SLLM configuration
SLLM_URL = os.environ.get("SLLM_URL")
SLLM_API_KEY = os.environ.get("SLLM_API_KEY")

# Anthropic configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Google configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


def similarity_dist(vec1, vec2, func="cosine"):
    if type(vec1) is list:
        vec1 = np.array(vec1)
    if type(vec2) is list:
        vec2 = np.array(vec2)
    if func == "cosine":
        return 1 - (vec1 @ vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    else:
        return np.linalg.norm(vec1 - vec2)


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


LOGGER = logging.getLogger("Nerif")


class MessageType(Enum):
    IMAGE_PATH = auto()
    IMAGE_URL = auto()
    IMAGE_BASE64 = auto()
    TEXT = auto()
    AUDIO_PATH = auto()
    AUDIO_URL = auto()
    AUDIO_BASE64 = auto()
    VIDEO_PATH = auto()
    VIDEO_URL = auto()


def _build_model_kwargs(
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict:
    """Build kwargs dict for litellm based on model name prefix."""
    kwargs = {"model": model}

    if model.startswith("custom_openai/"):
        kwargs["api_key"] = api_key if (api_key and api_key != "") else OPENAI_API_KEY
        kwargs["base_url"] = base_url if (base_url and base_url != "") else OPENAI_API_BASE
    elif model in OPENAI_MODEL or model in OPENAI_EMBEDDING_MODEL:
        kwargs["api_key"] = api_key if (api_key and api_key != "") else OPENAI_API_KEY
        kwargs["base_url"] = base_url if (base_url and base_url != "") else OPENAI_API_BASE
    elif model.startswith("anthropic/"):
        kwargs["api_key"] = api_key if (api_key and api_key != "") else ANTHROPIC_API_KEY
    elif model.startswith("gemini/"):
        kwargs["api_key"] = api_key if (api_key and api_key != "") else GOOGLE_API_KEY
    elif model.startswith("openrouter"):
        pass  # litellm handles openrouter natively
    elif model.startswith("ollama"):
        pass  # litellm handles ollama/ prefix natively
    elif model.startswith("vllm"):
        pass  # litellm handles vllm/ prefix natively
    elif model.startswith("sllm"):
        pass  # litellm handles sllm/ prefix natively
    else:
        # default: pass through api_key/base_url if provided
        if api_key and api_key != "":
            kwargs["api_key"] = api_key
        if base_url and base_url != "":
            kwargs["base_url"] = base_url

    return kwargs


def get_litellm_embedding(
    messages: str,
    model: str = NERIF_DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    counter: Optional[NerifTokenCounter] = None,
) -> Any:
    kargs = _build_model_kwargs(model, api_key, base_url)
    kargs["input"] = messages

    # For embedding, override base_url if explicitly provided
    if base_url and base_url != "":
        kargs["base_url"] = base_url

    response = litellm.embedding(**kargs)

    if counter is not None:
        counter.set_parser_based_on_model(model)
        counter.count_from_response(response)

    return response


def get_model_response(
    messages: List[Any],
    model: str = NERIF_DEFAULT_LLM_MODEL,
    temperature: float = 0,
    max_tokens: int | None = None,
    stream: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    logprobs: bool = False,
    top_logprobs: int = 5,
    counter: Optional[NerifTokenCounter] = None,
    tools: Optional[List[Any]] = None,
    tool_choice: Optional[Any] = None,
    response_format: Optional[Any] = None,
) -> Any:
    """
    Unified model response function. Routes to the correct backend based on model name prefix.
    Supports all backends that litellm supports (OpenAI, Anthropic, Gemini, Ollama, vLLM, etc.).

    Parameters:
    - messages (list): The list of messages to send to the model.
    - model (str): The model name (with optional prefix like 'anthropic/', 'gemini/', 'ollama/').
    - temperature (float): The temperature setting for response generation.
    - max_tokens (int): The maximum number of tokens to generate.
    - stream (bool): Whether to stream the response.
    - api_key (str): Override API key.
    - base_url (str): Override base URL.
    - logprobs (bool): Whether to return log probabilities.
    - top_logprobs (int): Number of top log probabilities to return.
    - counter: Token counter instance.
    - tools (list): Tool definitions for function calling.
    - tool_choice: Tool choice parameter for function calling.
    - response_format: Response format (e.g. {"type": "json_object"}).

    Returns:
    - The model response object.
    """
    kargs = _build_model_kwargs(model, api_key, base_url)
    kargs["messages"] = messages
    kargs["stream"] = stream
    kargs["temperature"] = temperature
    kargs["max_tokens"] = max_tokens

    if logprobs:
        kargs["logprobs"] = logprobs
        kargs["top_logprobs"] = top_logprobs

    if tools is not None:
        kargs["tools"] = tools
    if tool_choice is not None:
        kargs["tool_choice"] = tool_choice
    if response_format is not None:
        kargs["response_format"] = response_format

    # For custom_openai/ prefix, bypass litellm and use OpenAI client directly.
    # This is useful for proxies that only support /v1/chat/completions
    # (e.g. when litellm routes GPT-5 models to /v1/responses).
    if model.startswith("custom_openai/"):
        actual_model = model[len("custom_openai/"):]
        client_api_key = kargs.get("api_key") or OPENAI_API_KEY
        client_base_url = kargs.get("base_url") or OPENAI_API_BASE
        client = OpenAI(api_key=client_api_key, base_url=client_base_url)
        oai_kwargs = {
            "model": actual_model,
            "messages": messages,
            "stream": stream,
        }
        if temperature is not None:
            oai_kwargs["temperature"] = temperature
        if max_tokens is not None:
            oai_kwargs["max_tokens"] = max_tokens
        if tools is not None:
            oai_kwargs["tools"] = tools
        if tool_choice is not None:
            oai_kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            oai_kwargs["response_format"] = response_format
        responses = client.chat.completions.create(**oai_kwargs)
    else:
        responses = litellm.completion(**kargs)

    if counter is not None:
        counter.set_parser_based_on_model(model)
        counter.count_from_response(responses)

    return responses


# Backward-compatible alias
def get_litellm_response(
    messages: List[Any],
    model: str = NERIF_DEFAULT_LLM_MODEL,
    temperature: float = 0,
    max_tokens: int | None = None,
    stream: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    logprobs: bool = False,
    top_logprobs: int = 5,
    counter: Optional[NerifTokenCounter] = None,
) -> Any:
    """Backward-compatible wrapper around get_model_response()."""
    return get_model_response(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        api_key=api_key,
        base_url=base_url,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        counter=counter,
    )


def get_ollama_response(
    messages: List[Any],
    url: str = OLLAMA_URL,
    model: str = "llama3.1",
    max_tokens: int | None = None,
    temperature: float = 0,
    stream: bool = False,
    api_key: Optional[str] = None,
    counter: Optional[NerifTokenCounter] = None,
) -> Union[str, List[str]]:
    """Backward-compatible Ollama wrapper."""
    if url is None or url == "":
        url = "http://localhost:11434/v1/"

    return get_model_response(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        api_key=api_key,
        base_url=url,
        counter=counter,
    )


def get_vllm_response(
    messages: List[Any],
    url: str = VLLM_URL,
    model: str = "llama3.1",
    max_tokens: int | None = None,
    temperature: float = 0,
    stream: bool = False,
    api_key: Optional[str] = None,
    counter: Optional[NerifTokenCounter] = None,
) -> Union[str, List[str]]:
    """Backward-compatible vLLM wrapper. Uses OpenAI client directly for vLLM."""
    if url is None or url == "":
        url = "http://localhost:8000/v1"
    if api_key is None or api_key == "":
        api_key = "token-abc123"

    model = "/".join(model.split("/")[1:])

    client = OpenAI(
        base_url=url,
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )

    if counter is not None:
        counter.set_parser_based_on_model(model)
        counter.count_from_response(response)

    return response


def get_sllm_response(
    messages: List[Any],
    url: str = SLLM_URL,
    model: str = "llama3.1",
    max_tokens: int | None = None,
    temperature: float = 0,
    stream: bool = False,
    api_key: Optional[str] = None,
    counter: Optional[NerifTokenCounter] = None,
) -> Union[str, List[str]]:
    """Backward-compatible SLLM wrapper. Uses OpenAI client directly for SLLM."""
    if url is None or url == "":
        url = "http://localhost:8343/v1"
    if api_key is None or api_key == "":
        api_key = "token-abc123"

    model = "/".join(model.split("/")[1:])

    client = OpenAI(
        base_url=url,
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )

    if counter is not None:
        counter.set_parser_based_on_model(model)
        counter.count_from_response(response)

    return response
