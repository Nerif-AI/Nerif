import json
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
                anthropic_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    },
                })
            else:
                anthropic_parts.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": image_url,
                    },
                })
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
        anthropic_tools.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        })
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
            conversation.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content if isinstance(content, str) else str(content),
                }],
            })
        elif role == "assistant":
            # Check if this message has tool_calls (OpenAI format)
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                anthropic_content = []
                if content:
                    anthropic_content.append({"type": "text", "text": content})
                for tc in tool_calls:
                    func = tc.get("function", {}) if isinstance(tc, dict) else {"name": tc.function.name, "arguments": tc.function.arguments}
                    try:
                        input_data = json.loads(func.get("arguments", "{}"))
                    except (ValueError, TypeError):
                        input_data = {}
                    tc_id = tc.get("id", "") if isinstance(tc, dict) else tc.id
                    anthropic_content.append({
                        "type": "tool_use",
                        "id": tc_id,
                        "name": func.get("name", ""),
                        "input": input_data,
                    })
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
            contents.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": tool_call_id,
                        "response": response_data if isinstance(response_data, dict) else {"result": str(response_data)},
                    }
                }],
            })
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                parts = []
                if content:
                    parts.append({"text": content})
                for tc in tool_calls:
                    func = tc.get("function", {}) if isinstance(tc, dict) else {"name": tc.function.name, "arguments": tc.function.arguments}
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
            response_format=response_format,
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
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
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
