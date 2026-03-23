import base64
import time as _time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

from ..utils import (
    LOGGER,
    NERIF_DEFAULT_LLM_MODEL,
    OPENAI_MODEL,
    MessageType,
    NerifTokenCounter,
    get_embedding,
    get_embedding_async,
    get_model_response,
    get_model_response_async,
    get_model_response_stream,
    get_model_response_stream_async,
    get_response,
)
from ..utils.callbacks import CallbackManager, LLMEndEvent, LLMErrorEvent, LLMStartEvent
from ..utils.fallback import FallbackConfig
from ..utils.rate_limit import RateLimiter
from ..utils.retry import RetryConfig


@dataclass
class MultiModalMessage:
    """Helper class for building multi-modal messages with text, images, audio, and video."""

    parts: List[Dict[str, Any]] = field(default_factory=list)

    def add_text(self, text: str) -> "MultiModalMessage":
        self.parts.append({"type": "text", "text": text})
        return self

    def add_image_url(self, url: str) -> "MultiModalMessage":
        self.parts.append({"type": "image_url", "image_url": {"url": url}})
        return self

    def add_image_path(self, path: str) -> "MultiModalMessage":
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        self.parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        return self

    def add_image_base64(self, b64: str, media_type: str = "image/jpeg") -> "MultiModalMessage":
        self.parts.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}})
        return self

    def add_audio_url(self, url: str) -> "MultiModalMessage":
        self.parts.append({"type": "input_audio", "input_audio": {"url": url}})
        return self

    def add_audio_path(self, path: str, format: str = "wav") -> "MultiModalMessage":
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        self.parts.append({"type": "input_audio", "input_audio": {"data": b64, "format": format}})
        return self

    def add_audio_base64(self, b64: str, format: str = "wav") -> "MultiModalMessage":
        self.parts.append({"type": "input_audio", "input_audio": {"data": b64, "format": format}})
        return self

    def add_video_url(self, url: str) -> "MultiModalMessage":
        self.parts.append({"type": "video_url", "video_url": {"url": url}})
        return self

    def add_video_path(self, path: str) -> "MultiModalMessage":
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        self.parts.append({"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}})
        return self

    def to_content(self) -> List[Dict[str, Any]]:
        return self.parts


@dataclass
class ToolDefinition:
    """Helper for defining tools in OpenAI function calling format."""

    name: str
    description: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCallResult:
    """Represents a tool call returned by the model."""

    id: str
    name: str
    arguments: str

    def __repr__(self) -> str:
        return f"ToolCallResult(id={self.id!r}, name={self.name!r}, arguments={self.arguments!r})"


class SimpleChatModel:
    """
    A simple agent class for the Nerif project.
    Supports text and multi-modal input (images, audio, video) as well as
    tool calling and structured output (JSON mode).

    Attributes:
        model (str): The name of the model to use.
        default_prompt (str): The default system prompt for the chat.
        temperature (float): The temperature setting for response generation.
        counter (NerifTokenCounter): Token counter instance.
        messages (List[Any]): The conversation history.
        max_tokens (int): The maximum number of tokens to generate in the response.
        memory: Optional ConversationMemory for managed conversation history.
    """

    def __init__(
        self,
        model: str = NERIF_DEFAULT_LLM_MODEL,
        default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
        temperature: float = 0.0,
        counter: Optional[NerifTokenCounter] = None,
        max_tokens: None | int = None,
        retry_config: Optional[RetryConfig] = None,
        memory: Optional[Any] = None,
        fallback: Optional[List[str]] = None,
        callbacks: Optional[CallbackManager] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.default_prompt = default_prompt
        self.counter = counter
        self.agent_max_tokens = max_tokens
        self.retry_config = retry_config
        self.memory = memory
        self.callbacks = callbacks
        self.rate_limiter = rate_limiter
        self.fallback_config = None
        if fallback:
            self.fallback_config = FallbackConfig(models=[model] + fallback)
        self.last_response = None
        self.last_response_model = model
        self.last_latency_ms = 0.0

        if memory is not None:
            # Use the memory's internal list as the canonical message store.
            # Seed the system message via memory so window management is aware.
            if not memory._messages:
                memory.add_message("system", default_prompt)
            self.messages: List[Any] = memory._messages
        else:
            self.messages: List[Any] = [
                {"role": "system", "content": default_prompt},
            ]

    def reset(self, prompt: Optional[str] = None) -> None:
        if prompt is None:
            prompt = self.default_prompt
        if self.memory is not None:
            self.memory.clear()
            self.memory.add_message("system", prompt)
            self.messages = self.memory._messages
        else:
            self.messages: List[Any] = [{"role": "system", "content": prompt}]

    def set_max_tokens(self, max_tokens: None | int = None):
        self.agent_max_tokens = max_tokens

    def _call_with_fallback(self, call_fn, kwargs):
        """Try primary model, then each fallback in order."""
        if self.fallback_config is None:
            return call_fn(**kwargs)
        last_exception = None
        for model_name in self.fallback_config.models:
            try:
                kwargs["model"] = model_name
                return call_fn(**kwargs)
            except Exception as e:
                last_exception = e
                if not self.fallback_config.should_fallback(e):
                    raise
                LOGGER.warning("Model %s failed, trying next fallback: %s", model_name, e)
        raise last_exception

    async def _call_with_fallback_async(self, call_fn, kwargs):
        """Async version of _call_with_fallback."""
        if self.fallback_config is None:
            return await call_fn(**kwargs)
        last_exception = None
        for model_name in self.fallback_config.models:
            try:
                kwargs["model"] = model_name
                return await call_fn(**kwargs)
            except Exception as e:
                last_exception = e
                if not self.fallback_config.should_fallback(e):
                    raise
                LOGGER.warning("Model %s failed, trying next fallback: %s", model_name, e)
        raise last_exception

    def _prepare_chat_kwargs(self, message, max_tokens, tools, tool_choice, response_format, response_model):
        """Build message and kwargs for chat/achat. Appends user message to history."""
        if isinstance(message, MultiModalMessage):
            new_message = {"role": "user", "content": message.to_content()}
        else:
            new_message = {"role": "user", "content": message}
        if self.memory is not None:
            self.memory.add_message(new_message["role"], new_message["content"])
        else:
            self.messages.append(new_message)

        req_max_tokens = self.agent_max_tokens if max_tokens is None else max_tokens

        tool_dicts = None
        if tools is not None:
            tool_dicts = [t.to_dict() if isinstance(t, ToolDefinition) else t for t in tools]

        if response_model is not None:
            schema = response_model.model_json_schema()
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": response_model.__name__, "schema": schema},
            }

        kwargs = {"model": self.model, "temperature": self.temperature, "max_tokens": req_max_tokens}
        if self.counter is not None:
            kwargs["counter"] = self.counter
        if tool_dicts is not None:
            kwargs["tools"] = tool_dicts
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.retry_config is not None:
            kwargs["retry_config"] = self.retry_config
        return kwargs

    def _process_chat_result(self, result, append, response_model=None):
        """Process API result: handle tool calls, history, response_model parsing."""
        choice = result.choices[0]

        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            tool_results = [
                ToolCallResult(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                for tc in choice.message.tool_calls
            ]
            if append:
                if self.memory is not None:
                    self.memory.add_message("assistant", None)
                    self.memory._messages[-1]["tool_calls"] = choice.message.tool_calls
                else:
                    self.messages.append(
                        {"role": "assistant", "content": None, "tool_calls": choice.message.tool_calls}
                    )
            else:
                self.reset()
            return tool_results

        text_result = choice.message.content
        if append:
            if self.memory is not None:
                self.memory.add_message("assistant", text_result)
            else:
                self.messages.append({"role": "assistant", "content": text_result})
        else:
            self.reset()

        if response_model is not None:
            from ..utils.format import FormatVerifierPydantic

            verifier = FormatVerifierPydantic(response_model)
            return verifier(text_result)

        return text_result

    def _record_response_metadata(self, result, model_name: str, latency_ms: float) -> None:
        self.last_response = result
        self.last_response_model = getattr(result, "model", model_name)
        self.last_latency_ms = latency_ms

    def chat(
        self,
        message: Union[str, MultiModalMessage],
        append: bool = False,
        max_tokens: None | int = None,
        tools: Optional[List[Union[Dict[str, Any], ToolDefinition]]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Any] = None,
        response_model: Optional[Any] = None,
    ) -> Union[str, List[ToolCallResult]]:
        """Send a message and get a response."""
        kwargs = self._prepare_chat_kwargs(message, max_tokens, tools, tool_choice, response_format, response_model)

        LOGGER.debug("requested with message: %s", self.messages)
        LOGGER.debug("arguments of request: %s", kwargs)

        llm_start_time = _time.time()
        if self.callbacks:
            self.callbacks.fire(
                "on_llm_start",
                LLMStartEvent(
                    model=kwargs["model"],
                    messages=list(self.messages),
                    timestamp=llm_start_time,
                    kwargs=kwargs,
                ),
            )

        if self.rate_limiter is not None:
            self.rate_limiter.acquire()

        try:
            result = self._call_with_fallback(lambda **kw: get_model_response(self.messages, **kw), kwargs)
        except Exception as e:
            if self.rate_limiter is not None:
                self.rate_limiter.release()
            if self.callbacks:
                self.callbacks.fire(
                    "on_llm_error",
                    LLMErrorEvent(
                        model=kwargs["model"],
                        error=e,
                        latency_ms=(_time.time() - llm_start_time) * 1000,
                        will_retry=False,
                    ),
                )
            raise

        if self.rate_limiter is not None:
            self.rate_limiter.release()

        if self.callbacks:
            self.callbacks.fire(
                "on_llm_end",
                LLMEndEvent(
                    model=kwargs["model"],
                    response=str(result.choices[0].message.content or "")[:200],
                    latency_ms=(_time.time() - llm_start_time) * 1000,
                    prompt_tokens=getattr(getattr(result, "usage", None), "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(getattr(result, "usage", None), "completion_tokens", 0) or 0,
                    cost_usd=0.0,
                ),
            )

        self._record_response_metadata(result, kwargs["model"], (_time.time() - llm_start_time) * 1000)
        return self._process_chat_result(result, append, response_model)

    def stream_chat(
        self,
        message: Union[str, MultiModalMessage],
        append: bool = False,
        max_tokens: Optional[int] = None,
        response_format: Optional[Any] = None,
    ) -> Generator[str, None, None]:
        """Stream response chunks. Yields text strings.

        Note: Tool calling is not supported in streaming mode.
        """
        if isinstance(message, MultiModalMessage):
            new_message = {"role": "user", "content": message.to_content()}
        else:
            new_message = {"role": "user", "content": message}
        if self.memory is not None:
            self.memory.add_message(new_message["role"], new_message["content"])
        else:
            self.messages.append(new_message)

        req_max_tokens = self.agent_max_tokens if max_tokens is None else max_tokens

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": req_max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.counter is not None:
            kwargs["counter"] = self.counter
        if self.retry_config is not None:
            kwargs["retry_config"] = self.retry_config

        full_response = []
        for chunk in get_model_response_stream(self.messages, **kwargs):
            if chunk.content:
                full_response.append(chunk.content)
                yield chunk.content

        # Manage history
        complete_text = "".join(full_response)
        if append:
            if self.memory is not None:
                self.memory.add_message("assistant", complete_text)
            else:
                self.messages.append({"role": "assistant", "content": complete_text})
        else:
            self.reset()

    async def achat(
        self,
        message: Union[str, "MultiModalMessage"],
        append: bool = False,
        max_tokens: None | int = None,
        tools: Optional[List[Union[Dict[str, Any], "ToolDefinition"]]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Any] = None,
        response_model: Optional[Any] = None,
    ) -> Union[str, List["ToolCallResult"]]:
        """Async version of chat()."""
        kwargs = self._prepare_chat_kwargs(message, max_tokens, tools, tool_choice, response_format, response_model)

        LOGGER.debug("requested with message: %s", self.messages)
        LOGGER.debug("arguments of request: %s", kwargs)

        llm_start_time = _time.time()
        if self.callbacks:
            self.callbacks.fire(
                "on_llm_start",
                LLMStartEvent(
                    model=kwargs["model"],
                    messages=list(self.messages),
                    timestamp=llm_start_time,
                    kwargs=kwargs,
                ),
            )

        if self.rate_limiter is not None:
            await self.rate_limiter.aacquire()

        try:
            result = await self._call_with_fallback_async(
                lambda **kw: get_model_response_async(self.messages, **kw), kwargs
            )
        except Exception as e:
            if self.rate_limiter is not None:
                self.rate_limiter.arelease()
            if self.callbacks:
                self.callbacks.fire(
                    "on_llm_error",
                    LLMErrorEvent(
                        model=kwargs["model"],
                        error=e,
                        latency_ms=(_time.time() - llm_start_time) * 1000,
                        will_retry=False,
                    ),
                )
            raise

        if self.rate_limiter is not None:
            self.rate_limiter.arelease()

        if self.callbacks:
            self.callbacks.fire(
                "on_llm_end",
                LLMEndEvent(
                    model=kwargs["model"],
                    response=str(result.choices[0].message.content or "")[:200],
                    latency_ms=(_time.time() - llm_start_time) * 1000,
                    prompt_tokens=getattr(getattr(result, "usage", None), "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(getattr(result, "usage", None), "completion_tokens", 0) or 0,
                    cost_usd=0.0,
                ),
            )

        self._record_response_metadata(result, kwargs["model"], (_time.time() - llm_start_time) * 1000)
        return self._process_chat_result(result, append, response_model)

    def _continue_after_tools(self, tools=None, tool_choice=None):
        """Continue conversation after tool results without appending a user message.

        Used by NerifAgent when tool results are already in self.messages.
        Preserves fallback, callback, and rate-limiter behavior.
        """
        kwargs = {"model": self.model, "temperature": self.temperature,
                  "max_tokens": self.agent_max_tokens}
        if self.counter is not None:
            kwargs["counter"] = self.counter
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if self.retry_config is not None:
            kwargs["retry_config"] = self.retry_config

        llm_start_time = _time.time()
        if self.callbacks:
            self.callbacks.fire("on_llm_start", LLMStartEvent(
                model=kwargs["model"], messages=list(self.messages),
                timestamp=llm_start_time, kwargs=kwargs,
            ))

        if self.rate_limiter is not None:
            self.rate_limiter.acquire()

        try:
            result = self._call_with_fallback(
                lambda **kw: get_model_response(self.messages, **kw), kwargs
            )
        except Exception as e:
            if self.rate_limiter is not None:
                self.rate_limiter.release()
            if self.callbacks:
                self.callbacks.fire("on_llm_error", LLMErrorEvent(
                    model=kwargs["model"], error=e,
                    latency_ms=(_time.time() - llm_start_time) * 1000,
                    will_retry=False,
                ))
            raise

        if self.rate_limiter is not None:
            self.rate_limiter.release()

        if self.callbacks:
            self.callbacks.fire("on_llm_end", LLMEndEvent(
                model=kwargs["model"],
                response=str(result.choices[0].message.content or "")[:200],
                latency_ms=(_time.time() - llm_start_time) * 1000,
                prompt_tokens=getattr(getattr(result, "usage", None), "prompt_tokens", 0) or 0,
                completion_tokens=getattr(getattr(result, "usage", None), "completion_tokens", 0) or 0,
                cost_usd=0.0,
            ))

        self._record_response_metadata(result, kwargs["model"], (_time.time() - llm_start_time) * 1000)
        return self._process_chat_result(result, append=True)

    async def _acontinue_after_tools(self, tools=None, tool_choice=None):
        """Async version of _continue_after_tools."""
        kwargs = {"model": self.model, "temperature": self.temperature,
                  "max_tokens": self.agent_max_tokens}
        if self.counter is not None:
            kwargs["counter"] = self.counter
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if self.retry_config is not None:
            kwargs["retry_config"] = self.retry_config

        llm_start_time = _time.time()
        if self.callbacks:
            self.callbacks.fire("on_llm_start", LLMStartEvent(
                model=kwargs["model"], messages=list(self.messages),
                timestamp=llm_start_time, kwargs=kwargs,
            ))

        if self.rate_limiter is not None:
            await self.rate_limiter.aacquire()

        try:
            result = await self._call_with_fallback_async(
                lambda **kw: get_model_response_async(self.messages, **kw), kwargs
            )
        except Exception as e:
            if self.rate_limiter is not None:
                self.rate_limiter.arelease()
            if self.callbacks:
                self.callbacks.fire("on_llm_error", LLMErrorEvent(
                    model=kwargs["model"], error=e,
                    latency_ms=(_time.time() - llm_start_time) * 1000,
                    will_retry=False,
                ))
            raise

        if self.rate_limiter is not None:
            self.rate_limiter.arelease()

        if self.callbacks:
            self.callbacks.fire("on_llm_end", LLMEndEvent(
                model=kwargs["model"],
                response=str(result.choices[0].message.content or "")[:200],
                latency_ms=(_time.time() - llm_start_time) * 1000,
                prompt_tokens=getattr(getattr(result, "usage", None), "prompt_tokens", 0) or 0,
                completion_tokens=getattr(getattr(result, "usage", None), "completion_tokens", 0) or 0,
                cost_usd=0.0,
            ))

        self._record_response_metadata(result, kwargs["model"], (_time.time() - llm_start_time) * 1000)
        return self._process_chat_result(result, append=True)

    async def astream_chat(
        self,
        message: Union[str, "MultiModalMessage"],
        append: bool = False,
        max_tokens: Optional[int] = None,
        response_format: Optional[Any] = None,
    ) -> AsyncGenerator[str, None]:
        """Async streaming version of stream_chat(). Yields text strings."""
        if isinstance(message, MultiModalMessage):
            new_message = {"role": "user", "content": message.to_content()}
        else:
            new_message = {"role": "user", "content": message}
        if self.memory is not None:
            self.memory.add_message(new_message["role"], new_message["content"])
        else:
            self.messages.append(new_message)

        req_max_tokens = self.agent_max_tokens if max_tokens is None else max_tokens

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": req_max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.counter is not None:
            kwargs["counter"] = self.counter
        if self.retry_config is not None:
            kwargs["retry_config"] = self.retry_config

        full_response = []
        async for chunk in get_model_response_stream_async(self.messages, **kwargs):
            if chunk.content:
                full_response.append(chunk.content)
                yield chunk.content

        complete_text = "".join(full_response)
        if append:
            if self.memory is not None:
                self.memory.add_message("assistant", complete_text)
            else:
                self.messages.append({"role": "assistant", "content": complete_text})
        else:
            self.reset()


class SimpleEmbeddingModel:
    """
    A simple agent for embedding text.

    Attributes:
        model (str): The name of the embedding model to use.
        counter (NerifTokenCounter): Token counter instance.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        counter: Optional[NerifTokenCounter] = None,
    ):
        self.model = model
        self.counter = counter

    def embed(self, string: str) -> List[float]:
        result = get_embedding(
            messages=string,
            model=self.model,
            counter=self.counter,
        )

        return result.data[0]["embedding"]

    async def aembed(self, string: str) -> List[float]:
        """Async version of embed()."""
        result = await get_embedding_async(
            messages=string,
            model=self.model,
            counter=self.counter,
        )
        return result.data[0]["embedding"]


class LogitsChatModel:
    """
    A simple agent for fetching logits from a model.
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
        new_message = {"role": "user", "content": message}
        self.messages.append(new_message)

        req_max_tokens = self.agent_max_tokens if max_tokens is None else max_tokens

        if self.model in OPENAI_MODEL:
            result = get_response(
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


class OllamaEmbeddingModel:
    """
    A simple agent for Ollama embedding models.
    """

    def __init__(
        self,
        model: str = "ollama/mxbai-embed-large",
        url: str = "http://localhost:11434/v1/",
        counter: Optional[NerifTokenCounter] = None,
    ):
        self.model = model
        self.url = url
        self.counter = counter

    def embed(self, string: str) -> List[float]:
        result = get_embedding(
            messages=string,
            model=self.model,
            base_url=self.url,
            counter=self.counter,
        )

        return result.data[0]["embedding"]


class VisionModel:
    """
    A simple agent for vision tasks. Backward-compatible wrapper.
    For new code, use SimpleChatModel with MultiModalMessage.
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
            self.content_cache.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{content}"}})
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
            content = self.content_cache
        else:
            content = self.messages + input

        message = {
            "role": "user",
            "content": content,
        }
        self.messages.append(message)

        req_max_tokens = self.agent_max_tokens if max_tokens is None else max_tokens

        result = get_response(
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


class VideoModel:
    """
    A model for video understanding tasks.
    Wraps SimpleChatModel with MultiModalMessage for video input.
    """

    def __init__(
        self,
        model: str = NERIF_DEFAULT_LLM_MODEL,
        default_prompt: str = "You are a helpful assistant that can analyze video content.",
        temperature: float = 0.0,
        counter: Optional[NerifTokenCounter] = None,
        max_tokens: int | None = None,
    ):
        self._chat_model = SimpleChatModel(
            model=model,
            default_prompt=default_prompt,
            temperature=temperature,
            counter=counter,
            max_tokens=max_tokens,
        )

    def analyze_url(self, video_url: str, prompt: str = "Describe this video.", max_tokens: int | None = None) -> str:
        msg = MultiModalMessage().add_video_url(video_url).add_text(prompt)
        return self._chat_model.chat(msg, max_tokens=max_tokens)

    def analyze_path(self, video_path: str, prompt: str = "Describe this video.", max_tokens: int | None = None) -> str:
        msg = MultiModalMessage().add_video_path(video_path).add_text(prompt)
        return self._chat_model.chat(msg, max_tokens=max_tokens)

    def reset(self):
        self._chat_model.reset()
