"""Custom exception hierarchy for Nerif.

All Nerif exceptions inherit from NerifError, allowing users to catch
any Nerif-related error with a single except clause.
"""


class NerifError(Exception):
    """Base exception for all Nerif errors."""


class ProviderError(NerifError):
    """Error from LLM provider (API errors, auth failures, rate limits)."""

    def __init__(self, message: str, provider: str, status_code: int = None, response=None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.response = response


class FormatError(NerifError, ValueError):
    """Error parsing or verifying LLM output format.

    Inherits from both NerifError and ValueError for backward compatibility
    with code that catches ValueError from format parsing failures.
    """

    def __init__(self, message: str, raw_output: str = None, expected_type: type = None):
        super().__init__(message)
        self.raw_output = raw_output
        self.expected_type = expected_type


class ConversationMemoryError(NerifError):
    """Error in conversation memory operations (save/load, serialization)."""


class ConfigError(NerifError):
    """Missing or invalid configuration (API keys, model names)."""

    def __init__(self, message: str, missing_key: str = None):
        super().__init__(message)
        self.missing_key = missing_key


class ModelNotFoundError(ConfigError):
    """Model name could not be resolved to any known provider."""


class TokenLimitError(NerifError):
    """Request exceeds model's token limit."""


# Re-export from observability for discoverability
try:
    from nerif.observability.budget import BudgetExceededError  # noqa: F401
except ImportError:
    pass
