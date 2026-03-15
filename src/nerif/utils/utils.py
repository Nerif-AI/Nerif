import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx

from nerif.exceptions import ProviderError

from .retry import DEFAULT_RETRY, RetryConfig, retry_async, retry_sync
from .token_counter import NerifTokenCounter

# Environment variables

# OpenAI configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_PROXY_URL = os.environ.get("OPENAI_PROXY_URL")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")

# Default model settings
NERIF_DEFAULT_LLM_MODEL = os.environ.get("NERIF_DEFAULT_LLM_MODEL", "gpt-4o")
NERIF_DEFAULT_EMBEDDING_MODEL = os.environ.get("NERIF_DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")

# Allow disabling embedding via empty string
if NERIF_DEFAULT_EMBEDDING_MODEL == "":
    NERIF_DEFAULT_EMBEDDING_MODEL = None

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
    """Compute distance between two vectors. Pure Python — no numpy required."""
    if not isinstance(vec1, (list, tuple)):
        vec1 = list(vec1)
    if not isinstance(vec2, (list, tuple)):
        vec2 = list(vec2)
    if func == "cosine":
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return 1 - dot / (norm1 * norm2)
    else:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


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
class StreamChunk:
    """A single chunk from a streaming response."""

    content: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[_Usage] = None


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
        effective_model = model[len("custom_openai/") :]
        effective_key = api_key or OPENAI_API_KEY or ""
        effective_base = base_url or OPENAI_API_BASE or "https://api.openai.com/v1"
        provider = "openai_compat"
    elif model.startswith("anthropic/"):
        effective_model = model[len("anthropic/") :]
        effective_key = api_key or ANTHROPIC_API_KEY or ""
        effective_base = base_url or "https://api.anthropic.com"
        provider = "anthropic"
    elif model.startswith("gemini/"):
        effective_model = model[len("gemini/") :]
        effective_key = api_key or GOOGLE_API_KEY or ""
        effective_base = base_url or "https://generativelanguage.googleapis.com"
        provider = "gemini"
    elif model.startswith("openrouter/"):
        effective_model = model[len("openrouter/") :]
        effective_key = api_key or OPENROUTER_API_KEY or ""
        effective_base = base_url or "https://openrouter.ai/api/v1"
        provider = "openai_compat"
    elif model.startswith("ollama/"):
        effective_model = model[len("ollama/") :]
        effective_key = api_key or OLLAMA_API_KEY or "ollama"
        effective_base = base_url or OLLAMA_URL or "http://localhost:11434/v1"
        provider = "openai_compat"
    elif model.startswith("vllm/"):
        effective_model = model[len("vllm/") :]
        effective_key = api_key or VLLM_API_KEY or "token-abc123"
        effective_base = base_url or VLLM_URL or "http://localhost:8000/v1"
        provider = "openai_compat"
    elif model.startswith("sllm/"):
        effective_model = model[len("sllm/") :]
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


def _convert_content_to_anthropic(content: Any) -> Any:
    """Convert OpenAI-style message content to Anthropic format.

    Handles:
    - Plain strings (passed through)
    - Content arrays with text, image_url, etc.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content

    anthropic_parts = []
    for part in content:
        part_type = part.get("type", "")
        if part_type == "text":
            anthropic_parts.append({"type": "text", "text": part.get("text", "")})
        elif part_type == "image_url":
            image_url = part.get("image_url", {}).get("url", "")
            if image_url.startswith("data:"):
                # Parse data URI: data:<media_type>;base64,<data>
                header, _, b64_data = image_url.partition(",")
                media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
                anthropic_parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        },
                    }
                )
            else:
                anthropic_parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url,
                        },
                    }
                )
        # Skip unsupported types (audio, video) silently
    return anthropic_parts if anthropic_parts else content


def _convert_tools_to_anthropic(tools: List[Any]) -> List[Dict[str, Any]]:
    """Convert OpenAI tool format to Anthropic tool format.

    OpenAI: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    Anthropic: {"name": ..., "description": ..., "input_schema": ...}
    """
    anthropic_tools = []
    for tool in tools:
        func = tool.get("function", {})
        anthropic_tools.append(
            {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return anthropic_tools


def _convert_tool_choice_to_anthropic(tool_choice: Any) -> Dict[str, Any]:
    """Convert OpenAI tool_choice to Anthropic tool_choice format."""
    if tool_choice is None:
        return {"type": "auto"}
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "none":
            return {"type": "auto"}  # Anthropic doesn't have "none", use auto
        elif tool_choice == "required":
            return {"type": "any"}
        else:
            return {"type": "auto"}
    if isinstance(tool_choice, dict):
        # OpenAI: {"type": "function", "function": {"name": "..."}}
        func = tool_choice.get("function", {})
        if func.get("name"):
            return {"type": "tool", "name": func["name"]}
    return {"type": "auto"}


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
    response_format: Optional[Any] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
) -> ChatCompletionResponse:
    """Call the Anthropic Messages API and return a ChatCompletionResponse."""
    url = f"{base_url}/v1/messages"

    # Separate system message and convert content format
    system_text = None
    conversation = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_text = content if isinstance(content, str) else str(content)
        elif role == "tool":
            # Convert OpenAI tool result to Anthropic tool_result format
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": content if isinstance(content, str) else str(content),
                        }
                    ],
                }
            )
        elif role == "assistant":
            # Check if this message has tool_calls (OpenAI format)
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                anthropic_content = []
                if content:
                    anthropic_content.append({"type": "text", "text": content})
                for tc in tool_calls:
                    func = (
                        tc.get("function", {})
                        if isinstance(tc, dict)
                        else {"name": tc.function.name, "arguments": tc.function.arguments}
                    )
                    try:
                        input_data = json.loads(func.get("arguments", "{}"))
                    except (ValueError, TypeError):
                        input_data = {}
                    tc_id = tc.get("id", "") if isinstance(tc, dict) else tc.id
                    anthropic_content.append(
                        {
                            "type": "tool_use",
                            "id": tc_id,
                            "name": func.get("name", ""),
                            "input": input_data,
                        }
                    )
                conversation.append({"role": "assistant", "content": anthropic_content})
            else:
                conversation.append({"role": "assistant", "content": _convert_content_to_anthropic(content)})
        else:
            conversation.append({"role": role, "content": _convert_content_to_anthropic(content)})

    body: Dict[str, Any] = {
        "model": model,
        "messages": conversation,
        "max_tokens": max_tokens or 4096,
    }
    if system_text:
        # If JSON mode requested, add instruction to system prompt
        if response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
            system_text += "\n\nYou must respond with valid JSON only. No other text."
        body["system"] = system_text
    elif response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
        body["system"] = "You must respond with valid JSON only. No other text."
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if top_k is not None:
        body["top_k"] = top_k
    if stop_sequences is not None:
        body["stop_sequences"] = stop_sequences
    if tools is not None:
        body["tools"] = _convert_tools_to_anthropic(tools)
        body["tool_choice"] = _convert_tool_choice_to_anthropic(tool_choice)

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
    tool_calls_list = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            content_text += block.get("text", "")
        elif block.get("type") == "tool_use":
            tool_calls_list.append(
                _ToolCall(
                    id=block.get("id", ""),
                    type="function",
                    function=_FunctionCall(
                        name=block.get("name", ""),
                        arguments=json.dumps(block.get("input", {})),
                    ),
                )
            )

    message = _Message(
        role="assistant",
        content=content_text if content_text else None,
        tool_calls=tool_calls_list if tool_calls_list else None,
    )

    # Map stop_reason to finish_reason
    stop_reason = data.get("stop_reason", "end_turn")
    finish_reason = "tool_calls" if stop_reason == "tool_use" else "stop"

    usage_data = data.get("usage", {})
    return ChatCompletionResponse(
        id=data.get("id", ""),
        model=data.get("model", model),
        choices=[
            _Choice(
                index=0,
                message=message,
                finish_reason=finish_reason,
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


def _convert_content_to_gemini_parts(content: Any) -> List[Dict[str, Any]]:
    """Convert OpenAI-style message content to Gemini parts format."""
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts = []
    for part in content:
        part_type = part.get("type", "")
        if part_type == "text":
            parts.append({"text": part.get("text", "")})
        elif part_type == "image_url":
            image_url = part.get("image_url", {}).get("url", "")
            if image_url.startswith("data:"):
                # Parse data URI: data:<media_type>;base64,<data>
                header, _, b64_data = image_url.partition(",")
                mime_type = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
                parts.append({"inlineData": {"mimeType": mime_type, "data": b64_data}})
            else:
                # Gemini supports file URIs, not HTTP URLs directly in inline data
                parts.append({"text": f"[Image: {image_url}]"})
    return parts if parts else [{"text": ""}]


def _convert_tools_to_gemini(tools: List[Any]) -> List[Dict[str, Any]]:
    """Convert OpenAI tool format to Gemini functionDeclarations format.

    OpenAI: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    Gemini: {"functionDeclarations": [{"name": ..., "description": ..., "parameters": ...}]}
    """
    declarations = []
    for tool in tools:
        func = tool.get("function", {})
        decl: Dict[str, Any] = {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
        }
        params = func.get("parameters")
        if params:
            decl["parameters"] = params
        declarations.append(decl)
    return declarations


def _convert_tool_choice_to_gemini(tool_choice: Any) -> Dict[str, Any]:
    """Convert OpenAI tool_choice to Gemini toolConfig format."""
    if tool_choice is None or tool_choice == "auto":
        return {"functionCallingConfig": {"mode": "AUTO"}}
    if tool_choice == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    if tool_choice == "required":
        return {"functionCallingConfig": {"mode": "ANY"}}
    if isinstance(tool_choice, dict):
        func = tool_choice.get("function", {})
        if func.get("name"):
            return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [func["name"]]}}
    return {"functionCallingConfig": {"mode": "AUTO"}}


def _gemini_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Any],
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    tools: Optional[List[Any]] = None,
    tool_choice: Optional[Any] = None,
    response_format: Optional[Any] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
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
            system_instruction = content if isinstance(content, str) else str(content)
        elif role == "tool":
            # Convert tool result to Gemini function response
            tool_call_id = msg.get("tool_call_id", "")
            try:
                response_data = json.loads(content) if isinstance(content, str) else content
            except (ValueError, TypeError):
                response_data = {"result": content}
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": tool_call_id,
                                "response": response_data
                                if isinstance(response_data, dict)
                                else {"result": str(response_data)},
                            }
                        }
                    ],
                }
            )
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                parts = []
                if content:
                    parts.append({"text": content})
                for tc in tool_calls:
                    func = (
                        tc.get("function", {})
                        if isinstance(tc, dict)
                        else {"name": tc.function.name, "arguments": tc.function.arguments}
                    )
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except (ValueError, TypeError):
                        args = {}
                    parts.append({"functionCall": {"name": func.get("name", ""), "args": args}})
                contents.append({"role": "model", "parts": parts})
            else:
                gemini_parts = _convert_content_to_gemini_parts(content)
                contents.append({"role": "model", "parts": gemini_parts})
        else:
            gemini_parts = _convert_content_to_gemini_parts(content)
            contents.append({"role": "user", "parts": gemini_parts})

    body: Dict[str, Any] = {"contents": contents}
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    generation_config: Dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens
    if top_p is not None:
        generation_config["topP"] = top_p
    if top_k is not None:
        generation_config["topK"] = top_k
    if stop_sequences is not None:
        generation_config["stopSequences"] = stop_sequences
    if response_format and isinstance(response_format, dict):
        if response_format.get("type") == "json_object":
            generation_config["responseMimeType"] = "application/json"
        elif response_format.get("type") == "json_schema":
            generation_config["responseMimeType"] = "application/json"
            schema = response_format.get("json_schema", {}).get("schema")
            if schema:
                generation_config["responseSchema"] = schema
    if generation_config:
        body["generationConfig"] = generation_config

    if tools is not None:
        body["tools"] = [{"functionDeclarations": _convert_tools_to_gemini(tools)}]
        body["toolConfig"] = _convert_tool_choice_to_gemini(tool_choice)

    headers = {"content-type": "application/json"}

    resp = httpx.post(url, json=body, headers=headers, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    # Map Gemini response to ChatCompletionResponse
    content_text = ""
    tool_calls_list = []
    candidates = data.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            if "text" in part:
                content_text += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls_list.append(
                    _ToolCall(
                        id=fc.get("name", ""),  # Gemini doesn't provide IDs, use name
                        type="function",
                        function=_FunctionCall(
                            name=fc.get("name", ""),
                            arguments=json.dumps(fc.get("args", {})),
                        ),
                    )
                )

    message = _Message(
        role="assistant",
        content=content_text if content_text else None,
        tool_calls=tool_calls_list if tool_calls_list else None,
    )

    finish_reason = "tool_calls" if tool_calls_list else "stop"

    usage_meta = data.get("usageMetadata", {})
    return ChatCompletionResponse(
        id="",
        model=model,
        choices=[
            _Choice(
                index=0,
                message=message,
                finish_reason=finish_reason,
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
# Streaming helpers
# ---------------------------------------------------------------------------


def _parse_sse_line(line: str) -> Optional[dict]:
    """Parse a single SSE data line, return parsed JSON or None."""
    line = line.strip()
    if not line or not line.startswith("data: "):
        return None
    data = line[6:]  # Remove "data: " prefix
    if data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _openai_compatible_completion_stream(
    base_url,
    api_key,
    model,
    messages,
    temperature=0,
    max_tokens=None,
    tools=None,
    tool_choice=None,
    response_format=None,
):
    """Stream chat completion from an OpenAI-compatible endpoint. Yields StreamChunk."""
    url = f"{base_url}/chat/completions"
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
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

    with httpx.stream("POST", url, json=body, headers=headers, timeout=_DEFAULT_TIMEOUT) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            data = _parse_sse_line(line)
            if data is None:
                continue
            # Check for usage in the final chunk
            usage = None
            usage_data = data.get("usage")
            if usage_data:
                usage = _Usage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )
            choices = data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                finish_reason = choices[0].get("finish_reason")
                if content or finish_reason or usage:
                    yield StreamChunk(
                        content=content or "",
                        finish_reason=finish_reason,
                        usage=usage,
                    )
            elif usage:
                yield StreamChunk(content="", usage=usage)


def _anthropic_completion_stream(
    base_url,
    api_key,
    model,
    messages,
    temperature=0,
    max_tokens=None,
    tools=None,
    tool_choice=None,
    response_format=None,
):
    """Stream from Anthropic Messages API. Yields StreamChunk."""
    url = f"{base_url}/v1/messages"

    # Reuse message conversion from _anthropic_completion
    system_text = None
    conversation = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_text = content if isinstance(content, str) else str(content)
        else:
            conversation.append({"role": role, "content": _convert_content_to_anthropic(content)})

    body: Dict[str, Any] = {
        "model": model,
        "messages": conversation,
        "max_tokens": max_tokens or 4096,
        "stream": True,
    }
    if system_text:
        if response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
            system_text += "\n\nYou must respond with valid JSON only. No other text."
        body["system"] = system_text
    if temperature is not None:
        body["temperature"] = temperature

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    input_tokens = 0
    output_tokens = 0
    with httpx.stream("POST", url, json=body, headers=headers, timeout=_DEFAULT_TIMEOUT) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = data.get("type", "")
            if event_type == "message_start":
                usage_info = data.get("message", {}).get("usage", {})
                input_tokens = usage_info.get("input_tokens", 0)
            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                text = delta.get("text", "")
                if text:
                    yield StreamChunk(content=text)
            elif event_type == "message_delta":
                delta_usage = data.get("usage", {})
                output_tokens = delta_usage.get("output_tokens", 0)
                stop_reason = data.get("delta", {}).get("stop_reason")
                yield StreamChunk(
                    content="",
                    finish_reason="stop" if stop_reason else None,
                    usage=_Usage(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    ),
                )


def _gemini_completion_stream(
    base_url,
    api_key,
    model,
    messages,
    temperature=0,
    max_tokens=None,
    tools=None,
    tool_choice=None,
    response_format=None,
):
    """Stream from Gemini API. Yields StreamChunk."""
    url = f"{base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse&key={api_key}"

    # Reuse message conversion from _gemini_completion
    system_instruction = None
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_instruction = content if isinstance(content, str) else str(content)
        elif role == "assistant":
            gemini_parts = _convert_content_to_gemini_parts(content)
            contents.append({"role": "model", "parts": gemini_parts})
        else:
            gemini_parts = _convert_content_to_gemini_parts(content)
            contents.append({"role": "user", "parts": gemini_parts})

    body: Dict[str, Any] = {"contents": contents}
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    generation_config: Dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens
    if response_format and isinstance(response_format, dict):
        if response_format.get("type") == "json_object":
            generation_config["responseMimeType"] = "application/json"
    if generation_config:
        body["generationConfig"] = generation_config

    headers = {"content-type": "application/json"}

    with httpx.stream("POST", url, json=body, headers=headers, timeout=_DEFAULT_TIMEOUT) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            data = _parse_sse_line(line)
            if data is None:
                continue
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = ""
                for part in parts:
                    if "text" in part:
                        text += part["text"]
                finish_reason = candidates[0].get("finishReason")

                usage = None
                usage_meta = data.get("usageMetadata")
                if usage_meta:
                    usage = _Usage(
                        prompt_tokens=usage_meta.get("promptTokenCount", 0),
                        completion_tokens=usage_meta.get("candidatesTokenCount", 0),
                        total_tokens=usage_meta.get("totalTokenCount", 0),
                    )

                if text or finish_reason or usage:
                    yield StreamChunk(
                        content=text,
                        finish_reason="stop" if finish_reason == "STOP" else None,
                        usage=usage,
                    )


def get_model_response_stream(
    messages,
    model=NERIF_DEFAULT_LLM_MODEL,
    temperature=0,
    max_tokens=None,
    api_key=None,
    base_url=None,
    counter=None,
    tools=None,
    tool_choice=None,
    response_format=None,
    retry_config=None,
):
    """Stream model response. Yields StreamChunk objects."""
    effective_base, effective_key, effective_model, provider = _resolve_endpoint(model, api_key, base_url)

    if provider == "anthropic":
        stream_fn = _anthropic_completion_stream
    elif provider == "gemini":
        stream_fn = _gemini_completion_stream
    else:
        stream_fn = _openai_compatible_completion_stream

    kwargs: Dict[str, Any] = {
        "base_url": effective_base,
        "api_key": effective_key,
        "model": effective_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        kwargs["response_format"] = response_format

    yield from stream_fn(**kwargs)


# ---------------------------------------------------------------------------
# Async provider functions
# ---------------------------------------------------------------------------


async def _openai_compatible_completion_async(
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
    """Async version of _openai_compatible_completion."""
    url = f"{base_url}/chat/completions"

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
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

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    return _parse_chat_response(data, model)


async def _anthropic_completion_async(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Any],
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    tools: Optional[List[Any]] = None,
    tool_choice: Optional[Any] = None,
    response_format: Optional[Any] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
) -> ChatCompletionResponse:
    """Async version of _anthropic_completion."""
    url = f"{base_url}/v1/messages"

    system_text = None
    conversation = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_text = content if isinstance(content, str) else str(content)
        elif role == "tool":
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": content if isinstance(content, str) else str(content),
                        }
                    ],
                }
            )
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                anthropic_content = []
                if content:
                    anthropic_content.append({"type": "text", "text": content})
                for tc in tool_calls:
                    func = (
                        tc.get("function", {})
                        if isinstance(tc, dict)
                        else {"name": tc.function.name, "arguments": tc.function.arguments}
                    )
                    try:
                        input_data = json.loads(func.get("arguments", "{}"))
                    except (ValueError, TypeError):
                        input_data = {}
                    tc_id = tc.get("id", "") if isinstance(tc, dict) else tc.id
                    anthropic_content.append(
                        {
                            "type": "tool_use",
                            "id": tc_id,
                            "name": func.get("name", ""),
                            "input": input_data,
                        }
                    )
                conversation.append({"role": "assistant", "content": anthropic_content})
            else:
                conversation.append({"role": "assistant", "content": _convert_content_to_anthropic(content)})
        else:
            conversation.append({"role": role, "content": _convert_content_to_anthropic(content)})

    body: Dict[str, Any] = {
        "model": model,
        "messages": conversation,
        "max_tokens": max_tokens or 4096,
    }
    if system_text:
        if response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
            system_text += "\n\nYou must respond with valid JSON only. No other text."
        body["system"] = system_text
    elif response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
        body["system"] = "You must respond with valid JSON only. No other text."
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if top_k is not None:
        body["top_k"] = top_k
    if stop_sequences is not None:
        body["stop_sequences"] = stop_sequences
    if tools is not None:
        body["tools"] = _convert_tools_to_anthropic(tools)
        body["tool_choice"] = _convert_tool_choice_to_anthropic(tool_choice)

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    content_text = ""
    tool_calls_list = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            content_text += block.get("text", "")
        elif block.get("type") == "tool_use":
            tool_calls_list.append(
                _ToolCall(
                    id=block.get("id", ""),
                    type="function",
                    function=_FunctionCall(
                        name=block.get("name", ""),
                        arguments=json.dumps(block.get("input", {})),
                    ),
                )
            )

    message = _Message(
        role="assistant",
        content=content_text if content_text else None,
        tool_calls=tool_calls_list if tool_calls_list else None,
    )

    stop_reason = data.get("stop_reason", "end_turn")
    finish_reason = "tool_calls" if stop_reason == "tool_use" else "stop"

    usage_data = data.get("usage", {})
    return ChatCompletionResponse(
        id=data.get("id", ""),
        model=data.get("model", model),
        choices=[
            _Choice(
                index=0,
                message=message,
                finish_reason=finish_reason,
            )
        ],
        usage=_Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        ),
    )


async def _gemini_completion_async(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Any],
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    tools: Optional[List[Any]] = None,
    tool_choice: Optional[Any] = None,
    response_format: Optional[Any] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
) -> ChatCompletionResponse:
    """Async version of _gemini_completion."""
    url = f"{base_url}/v1beta/models/{model}:generateContent?key={api_key}"

    system_instruction = None
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_instruction = content if isinstance(content, str) else str(content)
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            try:
                response_data = json.loads(content) if isinstance(content, str) else content
            except (ValueError, TypeError):
                response_data = {"result": content}
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": tool_call_id,
                                "response": response_data
                                if isinstance(response_data, dict)
                                else {"result": str(response_data)},
                            }
                        }
                    ],
                }
            )
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                parts = []
                if content:
                    parts.append({"text": content})
                for tc in tool_calls:
                    func = (
                        tc.get("function", {})
                        if isinstance(tc, dict)
                        else {"name": tc.function.name, "arguments": tc.function.arguments}
                    )
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except (ValueError, TypeError):
                        args = {}
                    parts.append({"functionCall": {"name": func.get("name", ""), "args": args}})
                contents.append({"role": "model", "parts": parts})
            else:
                gemini_parts = _convert_content_to_gemini_parts(content)
                contents.append({"role": "model", "parts": gemini_parts})
        else:
            gemini_parts = _convert_content_to_gemini_parts(content)
            contents.append({"role": "user", "parts": gemini_parts})

    body: Dict[str, Any] = {"contents": contents}
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    generation_config: Dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens
    if top_p is not None:
        generation_config["topP"] = top_p
    if top_k is not None:
        generation_config["topK"] = top_k
    if stop_sequences is not None:
        generation_config["stopSequences"] = stop_sequences
    if response_format and isinstance(response_format, dict):
        if response_format.get("type") == "json_object":
            generation_config["responseMimeType"] = "application/json"
        elif response_format.get("type") == "json_schema":
            generation_config["responseMimeType"] = "application/json"
            schema = response_format.get("json_schema", {}).get("schema")
            if schema:
                generation_config["responseSchema"] = schema
    if generation_config:
        body["generationConfig"] = generation_config

    if tools is not None:
        body["tools"] = [{"functionDeclarations": _convert_tools_to_gemini(tools)}]
        body["toolConfig"] = _convert_tool_choice_to_gemini(tool_choice)

    headers = {"content-type": "application/json"}

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    content_text = ""
    tool_calls_list = []
    candidates = data.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            if "text" in part:
                content_text += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls_list.append(
                    _ToolCall(
                        id=fc.get("name", ""),
                        type="function",
                        function=_FunctionCall(
                            name=fc.get("name", ""),
                            arguments=json.dumps(fc.get("args", {})),
                        ),
                    )
                )

    message = _Message(
        role="assistant",
        content=content_text if content_text else None,
        tool_calls=tool_calls_list if tool_calls_list else None,
    )

    finish_reason = "tool_calls" if tool_calls_list else "stop"

    usage_meta = data.get("usageMetadata", {})
    return ChatCompletionResponse(
        id="",
        model=model,
        choices=[
            _Choice(
                index=0,
                message=message,
                finish_reason=finish_reason,
            )
        ],
        usage=_Usage(
            prompt_tokens=usage_meta.get("promptTokenCount", 0),
            completion_tokens=usage_meta.get("candidatesTokenCount", 0),
            total_tokens=usage_meta.get("totalTokenCount", 0),
        ),
    )


async def _openai_compatible_embedding_async(
    base_url: str,
    api_key: str,
    model: str,
    input_text: Union[str, List[str]],
) -> EmbeddingResponse:
    """Async version of _openai_compatible_embedding."""
    url = f"{base_url}/embeddings"

    body: Dict[str, Any] = {
        "model": model,
        "input": input_text,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        resp = await client.post(url, json=body, headers=headers)
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
# Async streaming helpers
# ---------------------------------------------------------------------------


async def _openai_compatible_completion_stream_async(
    base_url,
    api_key,
    model,
    messages,
    temperature=0,
    max_tokens=None,
    tools=None,
    tool_choice=None,
    response_format=None,
) -> AsyncGenerator:
    """Async streaming from an OpenAI-compatible endpoint. Yields StreamChunk."""
    url = f"{base_url}/chat/completions"
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
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

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                data = _parse_sse_line(line)
                if data is None:
                    continue
                usage = None
                usage_data = data.get("usage")
                if usage_data:
                    usage = _Usage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0),
                    )
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    finish_reason = choices[0].get("finish_reason")
                    if content or finish_reason or usage:
                        yield StreamChunk(
                            content=content or "",
                            finish_reason=finish_reason,
                            usage=usage,
                        )
                elif usage:
                    yield StreamChunk(content="", usage=usage)


async def _anthropic_completion_stream_async(
    base_url,
    api_key,
    model,
    messages,
    temperature=0,
    max_tokens=None,
    tools=None,
    tool_choice=None,
    response_format=None,
) -> AsyncGenerator:
    """Async streaming from Anthropic Messages API. Yields StreamChunk."""
    url = f"{base_url}/v1/messages"

    system_text = None
    conversation = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_text = content if isinstance(content, str) else str(content)
        else:
            conversation.append({"role": role, "content": _convert_content_to_anthropic(content)})

    body: Dict[str, Any] = {
        "model": model,
        "messages": conversation,
        "max_tokens": max_tokens or 4096,
        "stream": True,
    }
    if system_text:
        if response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
            system_text += "\n\nYou must respond with valid JSON only. No other text."
        body["system"] = system_text
    if temperature is not None:
        body["temperature"] = temperature

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    input_tokens = 0
    output_tokens = 0
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = data.get("type", "")
                if event_type == "message_start":
                    usage_info = data.get("message", {}).get("usage", {})
                    input_tokens = usage_info.get("input_tokens", 0)
                elif event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    text = delta.get("text", "")
                    if text:
                        yield StreamChunk(content=text)
                elif event_type == "message_delta":
                    delta_usage = data.get("usage", {})
                    output_tokens = delta_usage.get("output_tokens", 0)
                    stop_reason = data.get("delta", {}).get("stop_reason")
                    yield StreamChunk(
                        content="",
                        finish_reason="stop" if stop_reason else None,
                        usage=_Usage(
                            prompt_tokens=input_tokens,
                            completion_tokens=output_tokens,
                            total_tokens=input_tokens + output_tokens,
                        ),
                    )


async def _gemini_completion_stream_async(
    base_url,
    api_key,
    model,
    messages,
    temperature=0,
    max_tokens=None,
    tools=None,
    tool_choice=None,
    response_format=None,
) -> AsyncGenerator:
    """Async streaming from Gemini API. Yields StreamChunk."""
    url = f"{base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse&key={api_key}"

    system_instruction = None
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_instruction = content if isinstance(content, str) else str(content)
        elif role == "assistant":
            gemini_parts = _convert_content_to_gemini_parts(content)
            contents.append({"role": "model", "parts": gemini_parts})
        else:
            gemini_parts = _convert_content_to_gemini_parts(content)
            contents.append({"role": "user", "parts": gemini_parts})

    body: Dict[str, Any] = {"contents": contents}
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    generation_config: Dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens
    if response_format and isinstance(response_format, dict):
        if response_format.get("type") == "json_object":
            generation_config["responseMimeType"] = "application/json"
    if generation_config:
        body["generationConfig"] = generation_config

    headers = {"content-type": "application/json"}

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                data = _parse_sse_line(line)
                if data is None:
                    continue
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = ""
                    for part in parts:
                        if "text" in part:
                            text += part["text"]
                    finish_reason = candidates[0].get("finishReason")

                    usage = None
                    usage_meta = data.get("usageMetadata")
                    if usage_meta:
                        usage = _Usage(
                            prompt_tokens=usage_meta.get("promptTokenCount", 0),
                            completion_tokens=usage_meta.get("candidatesTokenCount", 0),
                            total_tokens=usage_meta.get("totalTokenCount", 0),
                        )

                    if text or finish_reason or usage:
                        yield StreamChunk(
                            content=text,
                            finish_reason="stop" if finish_reason == "STOP" else None,
                            usage=usage,
                        )


async def get_model_response_stream_async(
    messages,
    model=NERIF_DEFAULT_LLM_MODEL,
    temperature=0,
    max_tokens=None,
    api_key=None,
    base_url=None,
    counter=None,
    tools=None,
    tool_choice=None,
    response_format=None,
    retry_config=None,
) -> AsyncGenerator:
    """Async streaming model response. Yields StreamChunk objects."""
    effective_base, effective_key, effective_model, provider = _resolve_endpoint(model, api_key, base_url)

    if provider == "anthropic":
        stream_fn = _anthropic_completion_stream_async
    elif provider == "gemini":
        stream_fn = _gemini_completion_stream_async
    else:
        stream_fn = _openai_compatible_completion_stream_async

    kwargs: Dict[str, Any] = {
        "base_url": effective_base,
        "api_key": effective_key,
        "model": effective_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        kwargs["response_format"] = response_format

    async for chunk in stream_fn(**kwargs):
        yield chunk


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def get_embedding(
    messages: str,
    model: str = NERIF_DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    counter: Optional[NerifTokenCounter] = None,
    retry_config: Optional[RetryConfig] = None,
) -> Any:
    effective_base, effective_key, effective_model, provider = _resolve_endpoint(model, api_key, base_url)

    # For embedding, override base_url if explicitly provided
    if base_url and base_url != "":
        effective_base = base_url.rstrip("/")

    def _call():
        return _openai_compatible_embedding(effective_base, effective_key, effective_model, messages)

    try:
        response = retry_sync(_call, retry_config=retry_config or DEFAULT_RETRY)
    except httpx.HTTPStatusError as exc:
        raise ProviderError(
            str(exc),
            provider=provider,
            status_code=exc.response.status_code,
            response=exc.response,
        ) from exc

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
    retry_config: Optional[RetryConfig] = None,
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
    - retry_config: Retry configuration (default: DEFAULT_RETRY with 3 retries).

    Returns:
    - The model response object.
    """
    effective_base, effective_key, effective_model, provider = _resolve_endpoint(model, api_key, base_url)

    def _call():
        if provider == "anthropic":
            return _anthropic_completion(
                base_url=effective_base,
                api_key=effective_key,
                model=effective_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            )
        elif provider == "gemini":
            return _gemini_completion(
                base_url=effective_base,
                api_key=effective_key,
                model=effective_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            )
        else:
            # OpenAI-compatible (openai, openai_compat, openrouter, ollama, vllm, sllm)
            return _openai_compatible_completion(
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

    _start = time.monotonic()
    try:
        responses = retry_sync(_call, retry_config=retry_config or DEFAULT_RETRY)
    except httpx.HTTPStatusError as exc:
        _latency = (time.monotonic() - _start) * 1000.0
        if counter is not None:
            counter.record_request(model, latency_ms=_latency, success=False, error=exc)
        raise ProviderError(
            str(exc),
            provider=provider,
            status_code=exc.response.status_code,
            response=exc.response,
        ) from exc

    _latency = (time.monotonic() - _start) * 1000.0
    if counter is not None:
        counter.set_parser_based_on_model(model)
        counter.count_from_response(responses)
        _pt = responses.usage.prompt_tokens if hasattr(responses, "usage") else 0
        _ct = responses.usage.completion_tokens if hasattr(responses, "usage") else 0
        counter.record_request(model, latency_ms=_latency, success=True, prompt_tokens=_pt, completion_tokens=_ct)

    return responses


async def get_model_response_async(
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
    retry_config: Optional[RetryConfig] = None,
) -> Any:
    """Async version of get_model_response."""
    effective_base, effective_key, effective_model, provider = _resolve_endpoint(model, api_key, base_url)

    async def _call():
        if provider == "anthropic":
            return await _anthropic_completion_async(
                base_url=effective_base,
                api_key=effective_key,
                model=effective_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            )
        elif provider == "gemini":
            return await _gemini_completion_async(
                base_url=effective_base,
                api_key=effective_key,
                model=effective_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            )
        else:
            return await _openai_compatible_completion_async(
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

    _start = time.monotonic()
    try:
        responses = await retry_async(_call, retry_config=retry_config or DEFAULT_RETRY)
    except httpx.HTTPStatusError as exc:
        _latency = (time.monotonic() - _start) * 1000.0
        if counter is not None:
            counter.record_request(model, latency_ms=_latency, success=False, error=exc)
        raise ProviderError(
            str(exc),
            provider=provider,
            status_code=exc.response.status_code,
            response=exc.response,
        ) from exc

    _latency = (time.monotonic() - _start) * 1000.0
    if counter is not None:
        counter.set_parser_based_on_model(model)
        counter.count_from_response(responses)
        _pt = responses.usage.prompt_tokens if hasattr(responses, "usage") else 0
        _ct = responses.usage.completion_tokens if hasattr(responses, "usage") else 0
        counter.record_request(model, latency_ms=_latency, success=True, prompt_tokens=_pt, completion_tokens=_ct)

    return responses


async def get_embedding_async(
    messages: str,
    model: str = NERIF_DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    counter: Optional[NerifTokenCounter] = None,
    retry_config: Optional[RetryConfig] = None,
) -> Any:
    """Async version of get_embedding."""
    effective_base, effective_key, effective_model, provider = _resolve_endpoint(model, api_key, base_url)

    if base_url and base_url != "":
        effective_base = base_url.rstrip("/")

    async def _call():
        return await _openai_compatible_embedding_async(effective_base, effective_key, effective_model, messages)

    try:
        response = await retry_async(_call, retry_config=retry_config or DEFAULT_RETRY)
    except httpx.HTTPStatusError as exc:
        raise ProviderError(
            str(exc),
            provider=provider,
            status_code=exc.response.status_code,
            response=exc.response,
        ) from exc

    if counter is not None:
        counter.set_parser_based_on_model(model)
        counter.count_from_response(response)

    return response


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
    retry_config: Optional[RetryConfig] = None,
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
        retry_config=retry_config,
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
