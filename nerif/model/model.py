import base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..utils import (
    LOGGER,
    NERIF_DEFAULT_LLM_MODEL,
    OPENAI_MODEL,
    MessageType,
    NerifTokenCounter,
    get_litellm_embedding,
    get_litellm_response,
    get_model_response,
)


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
    """

    def __init__(
        self,
        model: str = NERIF_DEFAULT_LLM_MODEL,
        default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
        temperature: float = 0.0,
        counter: NerifTokenCounter = None,
        max_tokens: None | int = None,
    ):
        self.model = model
        self.temperature = temperature
        self.default_prompt = default_prompt
        self.messages: List[Any] = [
            {"role": "system", "content": default_prompt},
        ]
        self.counter = counter
        self.agent_max_tokens = max_tokens

    def reset(self, prompt: Optional[str] = None) -> None:
        if prompt is None:
            prompt = self.default_prompt
        self.messages: List[Any] = [{"role": "system", "content": prompt}]

    def set_max_tokens(self, max_tokens: None | int = None):
        self.agent_max_tokens = max_tokens

    def chat(
        self,
        message: Union[str, MultiModalMessage],
        append: bool = False,
        max_tokens: None | int = None,
        tools: Optional[List[Union[Dict[str, Any], ToolDefinition]]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Any] = None,
    ) -> Union[str, List[ToolCallResult]]:
        """
        Send a message and get a response.

        Args:
            message: Text string or MultiModalMessage for multi-modal input.
            append: If True, keep conversation history; if False, reset after response.
            max_tokens: Override max tokens for this request.
            tools: List of tool definitions for function calling.
            tool_choice: Tool choice parameter (e.g. "auto", "none", or specific tool).
            response_format: Response format (e.g. {"type": "json_object"}).

        Returns:
            Text response string, or list of ToolCallResult if tools were called.
        """
        if isinstance(message, MultiModalMessage):
            new_message = {"role": "user", "content": message.to_content()}
        else:
            new_message = {"role": "user", "content": message}
        self.messages.append(new_message)

        req_max_tokens = self.agent_max_tokens if max_tokens is None else max_tokens

        # Normalize tool definitions
        tool_dicts = None
        if tools is not None:
            tool_dicts = []
            for t in tools:
                if isinstance(t, ToolDefinition):
                    tool_dicts.append(t.to_dict())
                else:
                    tool_dicts.append(t)

        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": req_max_tokens,
        }

        if self.counter is not None:
            kwargs["counter"] = self.counter
        if tool_dicts is not None:
            kwargs["tools"] = tool_dicts
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format

        LOGGER.debug("requested with message: %s", self.messages)
        LOGGER.debug("arguments of request: %s", kwargs)

        result = get_model_response(self.messages, **kwargs)

        choice = result.choices[0]

        # Check if the model returned tool calls
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            tool_results = [
                ToolCallResult(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in choice.message.tool_calls
            ]
            if append:
                self.messages.append({"role": "assistant", "content": None, "tool_calls": choice.message.tool_calls})
            else:
                self.reset()
            return tool_results

        text_result = choice.message.content
        if append:
            self.messages.append({"role": "assistant", "content": text_result})
        else:
            self.reset()
        return text_result


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
        result = get_litellm_embedding(
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
        result = get_litellm_embedding(
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

    def analyze_path(
        self, video_path: str, prompt: str = "Describe this video.", max_tokens: int | None = None
    ) -> str:
        msg = MultiModalMessage().add_video_path(video_path).add_text(prompt)
        return self._chat_model.chat(msg, max_tokens=max_tokens)

    def reset(self):
        self._chat_model.reset()
