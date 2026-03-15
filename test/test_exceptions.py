"""Tests for Nerif custom exception hierarchy."""

import pytest

from nerif.exceptions import (
    ConfigError,
    ConversationMemoryError,
    FormatError,
    ModelNotFoundError,
    NerifError,
    ProviderError,
    TokenLimitError,
)


class TestExceptionHierarchy:
    """All custom exceptions should be catchable via NerifError."""

    def test_provider_error_is_nerif_error(self):
        with pytest.raises(NerifError):
            raise ProviderError("API failed", provider="openai", status_code=500)

    def test_format_error_is_nerif_error(self):
        with pytest.raises(NerifError):
            raise FormatError("bad output", raw_output="abc", expected_type=int)

    def test_format_error_is_value_error(self):
        """FormatError must also be catchable as ValueError for backward compatibility."""
        with pytest.raises(ValueError):
            raise FormatError("bad output", raw_output="abc", expected_type=int)

    def test_conversation_memory_error_is_nerif_error(self):
        with pytest.raises(NerifError):
            raise ConversationMemoryError("save failed")

    def test_config_error_is_nerif_error(self):
        with pytest.raises(NerifError):
            raise ConfigError("missing key", missing_key="OPENAI_API_KEY")

    def test_model_not_found_is_config_error(self):
        with pytest.raises(ConfigError):
            raise ModelNotFoundError("unknown model")

    def test_model_not_found_is_nerif_error(self):
        with pytest.raises(NerifError):
            raise ModelNotFoundError("unknown model")

    def test_token_limit_error_is_nerif_error(self):
        with pytest.raises(NerifError):
            raise TokenLimitError("too many tokens")


class TestExceptionAttributes:
    """Verify custom attributes are accessible."""

    def test_provider_error_attrs(self):
        err = ProviderError("rate limit", provider="anthropic", status_code=429, response={"error": "rate_limit"})
        assert err.provider == "anthropic"
        assert err.status_code == 429
        assert err.response == {"error": "rate_limit"}
        assert str(err) == "rate limit"

    def test_format_error_attrs(self):
        err = FormatError("parse failed", raw_output="not a number", expected_type=int)
        assert err.raw_output == "not a number"
        assert err.expected_type is int

    def test_config_error_attrs(self):
        err = ConfigError("API key not set", missing_key="OPENAI_API_KEY")
        assert err.missing_key == "OPENAI_API_KEY"

    def test_provider_error_default_attrs(self):
        err = ProviderError("fail", provider="openai")
        assert err.status_code is None
        assert err.response is None

    def test_config_error_default_attrs(self):
        err = ConfigError("bad config")
        assert err.missing_key is None


class TestNoShadowBuiltin:
    """Ensure we don't shadow Python's built-in MemoryError."""

    def test_conversation_memory_error_is_not_builtin(self):
        assert ConversationMemoryError is not MemoryError
        assert not issubclass(ConversationMemoryError, MemoryError)
