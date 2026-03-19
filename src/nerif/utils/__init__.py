from .callbacks import (
    CallbackHandler,
    CallbackManager,
    FallbackEvent,
    LLMEndEvent,
    LLMErrorEvent,
    LLMStartEvent,
    LoggingCallbackHandler,
    MemoryEvent,
    RetryEvent,
    ToolCallEvent,
)
from .fallback import FallbackConfig
from .format import (
    FormatVerifierBase,
    FormatVerifierFloat,
    FormatVerifierHumanReadableList,
    FormatVerifierInt,
    FormatVerifierJson,
    FormatVerifierListInt,
    NerifFormat,
)
from .log import NerifFormatter, set_up_logging, timestamp_filename
from .prompt import PromptTemplate
from .rate_limit import RateLimitConfig, RateLimiter, RateLimiterRegistry, rate_limiters
from .retry import AGGRESSIVE_RETRY, DEFAULT_RETRY, NO_RETRY, RetryConfig, retry_async, retry_sync
from .token_counter import ModelCost, NerifTokenCounter, OllamaResponseParser, OpenAIResponseParser, ResponseParserBase
from .utils import (
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    LOGGER,
    NERIF_DEFAULT_EMBEDDING_MODEL,
    NERIF_DEFAULT_LLM_MODEL,
    OLLAMA_API_KEY,
    OLLAMA_URL,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL,
    SLLM_API_KEY,
    SLLM_URL,
    VLLM_API_KEY,
    VLLM_URL,
    ChatCompletionResponse,
    EmbeddingResponse,
    MessageType,
    StreamChunk,
    get_embedding,
    get_embedding_async,
    get_litellm_embedding,
    get_litellm_response,
    get_model_response,
    get_model_response_async,
    get_model_response_stream,
    get_model_response_stream_async,
    get_ollama_response,
    get_response,
    get_sllm_response,
    get_vllm_response,
    similarity_dist,
)

try:
    from .image_compress import ImageCompressor, compress_image_simple
except ImportError:
    ImageCompressor = None
    compress_image_simple = None

from .format import FormatVerifierPydantic

__all__ = [
    # retry
    "RetryConfig",
    "DEFAULT_RETRY",
    "NO_RETRY",
    "AGGRESSIVE_RETRY",
    "retry_sync",
    "retry_async",
    # format
    "FormatVerifierBase",
    "FormatVerifierFloat",
    "FormatVerifierHumanReadableList",
    "FormatVerifierInt",
    "FormatVerifierJson",
    "FormatVerifierListInt",
    "FormatVerifierPydantic",
    "MessageType",
    "NerifFormat",
    # image compression
    "ImageCompressor",
    "compress_image_simple",
    # log
    "NerifFormatter",
    "set_up_logging",
    "timestamp_filename",
    # token counter
    "NerifTokenCounter",
    "ModelCost",
    "ResponseParserBase",
    "OpenAIResponseParser",
    "OllamaResponseParser",
    # response types
    "ChatCompletionResponse",
    "EmbeddingResponse",
    "StreamChunk",
    # utils - new names
    "similarity_dist",
    "get_embedding",
    "get_embedding_async",
    "get_response",
    "get_model_response",
    "get_model_response_async",
    "get_model_response_stream",
    "get_model_response_stream_async",
    "get_ollama_response",
    "get_sllm_response",
    "get_vllm_response",
    # utils - backward-compat aliases
    "get_litellm_embedding",
    "get_litellm_response",
    # environment variables
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_MODEL",
    "NERIF_DEFAULT_EMBEDDING_MODEL",
    "NERIF_DEFAULT_LLM_MODEL",
    "OLLAMA_URL",
    "OLLAMA_API_KEY",
    "VLLM_URL",
    "VLLM_API_KEY",
    "SLLM_URL",
    "SLLM_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    # default models
    "OPENAI_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    # default logger
    "LOGGER",
    # prompt
    "PromptTemplate",
    # callbacks
    "CallbackHandler",
    "CallbackManager",
    "LoggingCallbackHandler",
    "LLMStartEvent",
    "LLMEndEvent",
    "LLMErrorEvent",
    "ToolCallEvent",
    "FallbackEvent",
    "RetryEvent",
    "MemoryEvent",
    # fallback
    "FallbackConfig",
    # rate limiting
    "RateLimitConfig",
    "RateLimiter",
    "RateLimiterRegistry",
    "rate_limiters",
]
