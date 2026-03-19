"""Tests for v1.3.1 fixes."""

import asyncio

import pytest

from nerif.exceptions import FormatError
from nerif.utils.format import FormatVerifierStringList
from nerif.utils.rate_limit import RateLimitConfig, RateLimiter


class TestFormatVerifierStringListSecurity:
    """Test that FormatVerifierStringList uses safe parsing."""

    def test_convert_valid_list(self):
        v = FormatVerifierStringList()
        result = v.convert('["hello", "world"]')
        assert result == ["hello", "world"]

    def test_convert_invalid_input_raises(self):
        v = FormatVerifierStringList()
        with pytest.raises(FormatError):
            v.convert("not a list at all")

    def test_convert_rejects_code_injection(self):
        """eval() would execute this; ast.literal_eval must reject it."""
        v = FormatVerifierStringList()
        with pytest.raises(FormatError):
            v.convert("__import__('os').system('echo pwned')")

    def test_convert_single_element(self):
        v = FormatVerifierStringList()
        result = v.convert('["single"]')
        assert result == ["single"]


class TestRateLimiterAsyncSafety:
    """Test that async lock initialization is safe under concurrency."""

    def test_concurrent_aacquire_no_race(self):
        # No max_concurrent so aacquire() returns immediately after the
        # interval check; no semaphore deadlock between gather and arelease.
        config = RateLimitConfig(requests_per_second=100, max_concurrent=0)
        limiter = RateLimiter(config)

        async def run():
            tasks = [limiter.aacquire() for _ in range(10)]
            await asyncio.gather(*tasks)

        asyncio.run(run())
        # If no exception, the async primitives initialised without a race
