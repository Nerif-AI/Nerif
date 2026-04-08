"""Tests for rate limiting / concurrency control."""

import asyncio
import time

from nerif.utils.rate_limit import RateLimitConfig, RateLimiter, RateLimiterRegistry, rate_limiters


class TestRateLimitConfig:
    def test_default_no_limit(self):
        cfg = RateLimitConfig()
        assert cfg.min_interval == 0.0

    def test_requests_per_second(self):
        cfg = RateLimitConfig(requests_per_second=10.0)
        assert abs(cfg.min_interval - 0.1) < 1e-9

    def test_requests_per_minute(self):
        cfg = RateLimitConfig(requests_per_minute=60)
        assert abs(cfg.min_interval - 1.0) < 1e-9

    def test_rps_takes_precedence(self):
        cfg = RateLimitConfig(requests_per_second=10.0, requests_per_minute=600)
        assert abs(cfg.min_interval - 0.1) < 1e-9


class TestRateLimiterSync:
    def test_no_limit_is_instant(self):
        limiter = RateLimiter(RateLimitConfig())
        start = time.monotonic()
        limiter.acquire()
        limiter.release()
        assert time.monotonic() - start < 0.05

    def test_interval_enforcement(self):
        limiter = RateLimiter(RateLimitConfig(requests_per_second=20.0))
        limiter.acquire()
        limiter.release()
        start = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - start
        limiter.release()
        assert elapsed >= 0.03


class TestRateLimiterAsync:
    def test_async_acquire_release(self):
        limiter = RateLimiter(RateLimitConfig(max_concurrent=2))

        async def run():
            await limiter.aacquire()
            limiter.arelease()

        asyncio.run(run())

    def test_async_interval(self):
        limiter = RateLimiter(RateLimitConfig(requests_per_second=20.0))

        async def run():
            await limiter.aacquire()
            limiter.arelease()
            start = time.monotonic()
            await limiter.aacquire()
            elapsed = time.monotonic() - start
            limiter.arelease()
            assert elapsed >= 0.03

        asyncio.run(run())


class TestRateLimiterRegistry:
    def test_exact_match(self):
        reg = RateLimiterRegistry()
        cfg = RateLimitConfig(requests_per_minute=60)
        reg.configure("gpt-4o", cfg)
        limiter = reg.get("gpt-4o")
        assert limiter is not None

    def test_prefix_match(self):
        reg = RateLimiterRegistry()
        reg.configure("anthropic/", RateLimitConfig(requests_per_minute=30))
        limiter = reg.get("anthropic/claude-3-5-sonnet-20241022")
        assert limiter is not None

    def test_no_match(self):
        reg = RateLimiterRegistry()
        assert reg.get("gpt-4o") is None

    def test_global_registry_exists(self):
        assert rate_limiters is not None
        assert isinstance(rate_limiters, RateLimiterRegistry)
