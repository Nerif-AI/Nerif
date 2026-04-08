from . import agent, batch, core, model, utils
from .exceptions import (
    ConfigError,
    ConversationMemoryError,
    FormatError,
    ModelNotFoundError,
    NerifError,
    ProviderError,
    TokenLimitError,
)

__all__ = [
    "agent",
    "batch",
    "core",
    "model",
    "observability",
    "utils",
    "NerifError",
    "ProviderError",
    "FormatError",
    "ConfigError",
    "ConversationMemoryError",
    "ModelNotFoundError",
    "TokenLimitError",
]


def __getattr__(name):
    """Lazy import for optional feature subpackages."""
    if name in ("asr", "img_gen", "tts", "rag", "memory", "observability"):
        import importlib

        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
