import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

import httpx
import numpy as np

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

# Default httpx timeout (30 seconds connect, 120 seconds read for LLM responses)
_DEFAULT_TIMEOUT = httpx.Timeout(30.0, read=120.0)


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


# ---------------------------------------------------------------------------
# Lightweight response dataclasses that mirror OpenAI/litellm response shapes
# ---------------------------------------------------------------------------


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _FunctionCall:
    name: str = ""
    arguments: str = ""


@dataclass
class _ToolCall:
    id: str = ""
    type: str = "function"
    function: _FunctionCall = field(default_factory=_FunctionCall)


@dataclass
class _Message:
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[_ToolCall]] = None


@dataclass
class _LogprobContent:
    token: str = ""
    logprob: float = 0.0
    bytes: Optional[List[int]] = None
    top_logprobs: Optional[List[Dict[str, Any]]] = None


@dataclass
class _Logprobs:
    content: Optional[List[Dict[str, Any]]] = None


@dataclass
class _Choice:
    index: int = 0
    message: _Message = field(default_factory=_Message)
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


@dataclass
class ChatCompletionResponse:
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[_Choice] = field(default_factory=list)
    usage: _Usage = field(default_factory=_Usage)


@dataclass
class _EmbeddingData:
    object: str = "embedding"
    embedding: List[float] = field(default_factory=list)
    index: int = 0


@dataclass
class EmbeddingResponse:
    object: str = "list"
    model: str = ""
    data: List[Dict[str, Any]] = field(default_factory=list)
    usage: _Usage = field(default_factory=_Usage)


# ---------------------------------------------------------------------------
# Provider routing helpers
# ---------------------------------------------------------------------------


def _resolve_endpoint(
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> tuple:
    """Return (effective_base_url, effective_api_key, effective_model, provider) for a model string."""

    provider = "openai"

    if model.startswith("custom_openai/"):
        effective_model = model[len("custom_openai/"):]
        effective_key = api_key or OPENAI_API_KEY or ""
        effective_base = base_url or OPENAI_API_BASE or "https://api.openai.com/v1"
        provider = "openai_compat"
    elif model.startswith("anthropic/"):
        effective_model = model[len("anthropic/"):]
        effective_key = api_key or ANTHROPIC_API_KEY or ""
        effective_base = base_url or "https://api.anthropic.com"
        provider = "anthropic"
    elif model.startswith("gemini/"):
        effective_model = model[len("gemini/"):]
        effective_key = api_key or GOOGLE_API_KEY or ""
        effective_base = base_url or "https://generativelanguage.googleapis.com"
        provider = "gemini"
    elif model.startswith("openrouter/"):
        effective_model = model[len("openrouter/"):]
        effective_key = api_key or OPENROUTER_API_KEY or ""
        effective_base = base_url or "https://openrouter.ai/api/v1"
        provider = "openai_compat"
    elif model.startswith("ollama/"):
        effective_model = model[len("ollama/"):]
        effective_key = api_key or OLLAMA_API_KEY or "ollama"
        effective_base = base_url or OLLAMA_URL or "http://localhost:11434/v1"
        provider = "openai_compat"
    elif model.startswith("vllm/"):
        effective_model = model[len("vllm/"):]
        effective_key = api_key or VLLM_API_KEY or "token-abc123"
        effective_base = base_url or VLLM_URL or "http://localhost:8000/v1"
        provider = "openai_compat"
    elif model.startswith("sllm/"):
        effective_model = model[len("sllm/"):]
        effective_key = api_key or SLLM_API_KEY or "token-abc123"
        effective_base = base_url or SLLM_URL or "http://localhost:8343/v1"
        provider = "openai_compat"
    elif model in OPENAI_MODEL or model in OPENAI_EMBEDDING_MODEL:
        effective_model = model
        effective_key = api_key or OPENAI_API_KEY or ""
        effective_base = base_url or OPENAI_API_BASE or "https://api.openai.com/v1"
        provider = "openai"
    else:
        # Default: treat as OpenAI-compatible
        effective_model = model
        effective_key = api_key or OPENAI_API_KEY or ""
        effective_base = base_url or OPENAI_API_BASE or "https://api.openai.com/v1"
        provider = "openai"

    # Ensure base URL doesn't end with trailing slash for consistent joining
    effective_base = effective_base.rstrip("/")

    return effective_base, effective_key, effective_model, provider


# ---------------------------------------------------------------------------
# Anthropic-specific helpers
# ---------------------------------------------------------------------------


def _anthropic_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Any],
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    tools: Optional[List[Any]] = None,
    tool_choice: Optional[Any] = None,
) -> ChatCompletionResponse:
    """Call the Anthropic Messages API and return a ChatCompletionResponse."""
    url = f"{base_url}/v1/messages"

    # Separate system message from conversation
    system_text = None
    conversation = []
    for msg in messages:
        if msg.get("role") == "system":
            system_text = msg.get("content", "")
        else:
            conversation.append(msg)

    body: Dict[str, Any] = {
        "model": model,
        "messages": conversation,
        "max_tokens": max_tokens or 4096,
    }
    if system_text:
        body["system"] = system_text
    if temperature is not None:
        body["temperature"] = temperature

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    resp = httpx.post(url, json=body, headers=headers, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    # Map Anthropic response to ChatCompletionResponse
    content_text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            content_text += block.get("text", "")

    usage_data = data.get("usage", {})
    return ChatCompletionResponse(
        id=data.get("id", ""),
        model=data.get("model", model),
        choices=[
            _Choice(
                index=0,
                message=_Message(role="assistant", content=content_text),
                finish_reason=data.get("stop_reason", "end_turn"),
            )
        ],
        usage=_Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        ),
    )


# ---------------------------------------------------------------------------
# Gemini-specific helpers
# ---------------------------------------------------------------------------


def _gemini_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Any],
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    stream: bool = False,
) -> ChatCompletionResponse:
    """Call the Google Gemini API and return a ChatCompletionResponse."""
    url = f"{base_url}/v1beta/models/{model}:generateContent?key={api_key}"

    # Convert OpenAI-style messages to Gemini format
    system_instruction = None
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_instruction = content
        else:
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": content}]})

    body: Dict[str, Any] = {"contents": contents}
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    generation_config: Dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens
    if generation_config:
        body["generationConfig"] = generation_config

    headers = {"content-type": "application/json"}

    resp = httpx.post(url, json=body, headers=headers, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    # Map Gemini response to ChatCompletionResponse
    content_text = ""
    candidates = data.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            content_text += part.get("text", "")

    usage_meta = data.get("usageMetadata", {})
    return ChatCompletionResponse(
        id="",
        model=model,
        choices=[
            _Choice(
                index=0,
                message=_Message(role="assistant", content=content_text),
                finish_reason="stop",
            )
        ],
        usage=_Usage(
            prompt_tokens=usage_meta.get("promptTokenCount", 0),
            completion_tokens=usage_meta.get("candidatesTokenCount", 0),
            total_tokens=usage_meta.get("totalTokenCount", 0),
        ),
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible completion/embedding (covers OpenAI, OpenRouter, Ollama, vLLM, sllm, custom)
# ---------------------------------------------------------------------------


def _parse_chat_response(data: dict, model_name: str) -> ChatCompletionResponse:
    """Parse an OpenAI-compatible JSON response into ChatCompletionResponse."""
    choices = []
    for c in data.get("choices", []):
        msg_data = c.get("message", {})

        # Parse tool calls if present
        tool_calls = None
        if msg_data.get("tool_calls"):
            tool_calls = []
            for tc in msg_data["tool_calls"]:
                func = tc.get("function", {})
                tool_calls.append(
                    _ToolCall(
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                        function=_FunctionCall(
                            name=func.get("name", ""),
                            arguments=func.get("arguments", ""),
                        ),
                    )
                )

        # Parse logprobs if present
        logprobs_data = c.get("logprobs")

        choices.append(
            _Choice(
                index=c.get("index", 0),
                message=_Message(
                    role=msg_data.get("role", "assistant"),
                    content=msg_data.get("content"),
                    tool_calls=tool_calls,
                ),
                finish_reason=c.get("finish_reason"),
                logprobs=logprobs_data,
            )
        )

    usage_data = data.get("usage", {})
    return ChatCompletionResponse(
        id=data.get("id", ""),
        object=data.get("object", "chat.completion"),
        created=data.get("created", 0),
        model=data.get("model", model_name),
        choices=choices,
        usage=_Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        ),
    )


def _openai_compatible_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Any],
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    logprobs: bool = False,
    top_logprobs: int = 5,
    tools: Optional[List[Any]] = None,
    tool_choice: Optional[Any] = None,
    response_format: Optional[Any] = None,
) -> ChatCompletionResponse:
    """Make a chat completion request to an OpenAI-compatible endpoint."""
    url = f"{base_url}/chat/completions"

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,  # We don't support streaming in this implementation
    }
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = top_logprobs
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if response_format is not None:
        body["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = httpx.post(url, json=body, headers=headers, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    return _parse_chat_response(data, model)


def _openai_compatible_embedding(
    base_url: str,
    api_key: str,
    model: str,
    input_text: Union[str, List[str]],
) -> EmbeddingResponse:
    """Make an embedding request to an OpenAI-compatible endpoint."""
    url = f"{base_url}/embeddings"

    body: Dict[str, Any] = {
        "model": model,
        "input": input_text,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = httpx.post(url, json=body, headers=headers, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    embedding_data = []
    for item in data.get("data", []):
        embedding_data.append({"embedding": item.get("embedding", []), "index": item.get("index", 0)})

    usage_data = data.get("usage", {})
    return EmbeddingResponse(
        model=data.get("model", model),
        data=embedding_data,
        usage=_Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        ),
    )


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def get_embedding(
    messages: str,
    model: str = NERIF_DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    counter: Optional[NerifTokenCounter] = None,
) -> Any:
    effective_base, effective_key, effective_model, provider = _resolve_endpoint(model, api_key, base_url)

    # For embedding, override base_url if explicitly provided
    if base_url and base_url != "":
        effective_base = base_url.rstrip("/")

    response = _openai_compatible_embedding(effective_base, effective_key, effective_model, messages)

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
    effective_base, effective_key, effective_model, provider = _resolve_endpoint(model, api_key, base_url)

    if provider == "anthropic":
        responses = _anthropic_completion(
            base_url=effective_base,
            api_key=effective_key,
            model=effective_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
        )
    elif provider == "gemini":
        responses = _gemini_completion(
            base_url=effective_base,
            api_key=effective_key,
            model=effective_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
    else:
        # OpenAI-compatible (openai, openai_compat, openrouter, ollama, vllm, sllm)
        responses = _openai_compatible_completion(
            base_url=effective_base,
            api_key=effective_key,
            model=effective_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

    if counter is not None:
        counter.set_parser_based_on_model(model)
        counter.count_from_response(responses)

    return responses


# Backward-compatible aliases
get_litellm_embedding = get_embedding


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


def get_response(
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
    """Alias for get_model_response()."""
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
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
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
    """Backward-compatible vLLM wrapper."""
    if url is None or url == "":
        url = "http://localhost:8000/v1"
    if api_key is None or api_key == "":
        api_key = "token-abc123"

    actual_model = "/".join(model.split("/")[1:])

    effective_base = url.rstrip("/")
    response = _openai_compatible_completion(
        base_url=effective_base,
        api_key=api_key,
        model=actual_model,
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
    """Backward-compatible SLLM wrapper."""
    if url is None or url == "":
        url = "http://localhost:8343/v1"
    if api_key is None or api_key == "":
        api_key = "token-abc123"

    actual_model = "/".join(model.split("/")[1:])

    effective_base = url.rstrip("/")
    response = _openai_compatible_completion(
        base_url=effective_base,
        api_key=api_key,
        model=actual_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )

    if counter is not None:
        counter.set_parser_based_on_model(model)
        counter.count_from_response(response)

    return response
