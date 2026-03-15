"""Model fallback chain configuration."""

from dataclasses import dataclass, field
from typing import Set

from .retry import _is_transient_error


@dataclass
class FallbackConfig:
    """Configure model fallback behavior.

    Args:
        models: Ordered list of model names — primary first, then fallbacks.
        fallback_on: HTTP status codes that trigger fallback.
        fallback_on_timeout: Whether to fallback on timeout errors.
    """

    models: list
    fallback_on: Set[int] = field(default_factory=lambda: {429, 500, 502, 503, 504})
    fallback_on_timeout: bool = True

    def should_fallback(self, exception: Exception) -> bool:
        """Check if the exception should trigger a fallback to the next model."""
        return _is_transient_error(exception, self.fallback_on, self.fallback_on_timeout)
