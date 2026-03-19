"""Rate limiting and concurrency control for LLM API requests."""

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a model or provider.

    Args:
        requests_per_minute: Max requests per minute (0 = unlimited).
        requests_per_second: Max requests per second (0 = unlimited). Takes precedence.
        max_concurrent: Max concurrent requests (0 = unlimited).
    """

    requests_per_minute: int = 0
    requests_per_second: float = 0.0
    max_concurrent: int = 0

    @property
    def min_interval(self) -> float:
        """Minimum seconds between requests."""
        if self.requests_per_second > 0:
            return 1.0 / self.requests_per_second
        if self.requests_per_minute > 0:
            return 60.0 / self.requests_per_minute
        return 0.0


class RateLimiter:
    """Thread-safe rate limiter using interval enforcement + semaphore.

    Uses separate locks for sync (threading.Lock) and async (asyncio.Lock)
    paths to avoid holding a threading lock across await points.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._sync_lock = threading.Lock()
        self._async_lock: Optional[asyncio.Lock] = None
        self._last_request = 0.0
        self._semaphore: Optional[threading.Semaphore] = None
        self._async_semaphore: Optional[asyncio.Semaphore] = None
        if config.max_concurrent > 0:
            self._semaphore = threading.Semaphore(config.max_concurrent)
            # async semaphore created lazily in aacquire() to avoid
            # binding to a specific event loop at init time

    def acquire(self) -> None:
        """Block until we can make a request (sync)."""
        if self._semaphore is not None:
            self._semaphore.acquire()

        interval = self.config.min_interval
        if interval > 0:
            with self._sync_lock:
                now = time.monotonic()
                elapsed = now - self._last_request
                if elapsed < interval:
                    time.sleep(interval - elapsed)
                self._last_request = time.monotonic()

    def release(self) -> None:
        """Release a concurrent request slot (sync)."""
        if self._semaphore is not None:
            self._semaphore.release()

    def _ensure_async_primitives(self) -> None:
        """Lazily create async primitives. Must be called from async context.

        Safe because asyncio is single-threaded within an event loop —
        no two coroutines execute this simultaneously.
        """
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        if self.config.max_concurrent > 0 and self._async_semaphore is None:
            self._async_semaphore = asyncio.Semaphore(self.config.max_concurrent)

    async def aacquire(self) -> None:
        """Block until we can make a request (async).

        Uses asyncio.Lock to avoid holding a threading lock across await.
        """
        self._ensure_async_primitives()

        if self._async_semaphore is not None:
            await self._async_semaphore.acquire()

        interval = self.config.min_interval
        if interval > 0:
            async with self._async_lock:
                now = time.monotonic()
                elapsed = now - self._last_request
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)
                self._last_request = time.monotonic()

    def arelease(self) -> None:
        """Release a concurrent request slot (async).

        Sync method — asyncio.Semaphore.release() is not a coroutine.
        Named arelease for API symmetry with aacquire.
        """
        if self._async_semaphore is not None:
            self._async_semaphore.release()


class RateLimiterRegistry:
    """Global registry of per-model/provider rate limiters."""

    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

    def configure(self, key: str, config: RateLimitConfig) -> None:
        """Set rate limit for a model name or provider prefix."""
        with self._lock:
            self._limiters[key] = RateLimiter(config)

    def get(self, model: str) -> Optional[RateLimiter]:
        """Get limiter for a model. Checks exact match, then provider prefix."""
        with self._lock:
            if model in self._limiters:
                return self._limiters[model]
            for prefix, limiter in self._limiters.items():
                if model.startswith(prefix):
                    return limiter
            return None


# Global registry instance
rate_limiters = RateLimiterRegistry()
