import random
import time
from dataclasses import dataclass, field
from typing import Optional, Set

import httpx

from .callbacks import RetryEvent


def _is_transient_error(exception: Exception, status_codes, on_timeout: bool) -> bool:
    """Check if an exception represents a transient error worth retrying/falling back from."""
    if isinstance(exception, httpx.TimeoutException):
        return on_timeout
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code in status_codes
    if isinstance(exception, (httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    return False


@dataclass
class RetryConfig:
    """Configurable retry strategy with exponential backoff."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_status_codes: Set[int] = field(default_factory=lambda: {429, 500, 502, 503, 504})
    retry_on_timeout: bool = True

    def get_delay(self, attempt: int) -> float:
        delay = min(
            self.base_delay * (self.exponential_base**attempt),
            self.max_delay,
        )
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        return delay

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        if attempt >= self.max_retries:
            return False
        return _is_transient_error(exception, self.retryable_status_codes, self.retry_on_timeout)


NO_RETRY = RetryConfig(max_retries=0)
DEFAULT_RETRY = RetryConfig()
AGGRESSIVE_RETRY = RetryConfig(max_retries=5, base_delay=0.5)


def retry_sync(func, *args, retry_config: Optional[RetryConfig] = None, counter=None, model="", callbacks=None, **kwargs):
    """Execute a function with retry logic."""
    config = retry_config or DEFAULT_RETRY
    last_exception = None
    for attempt in range(config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if not config.should_retry(attempt, e):
                raise
            delay = config.get_delay(attempt)
            # Respect Retry-After header for 429
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
            if callbacks is not None:
                callbacks.fire(
                    "on_retry",
                    RetryEvent(
                        model=model,
                        attempt=attempt + 1,
                        max_retries=config.max_retries,
                        delay=delay,
                        error=e,
                    ),
                )
            time.sleep(delay)
            if counter is not None:
                counter.record_retry(model)
    raise last_exception


async def retry_async(func, *args, retry_config: Optional[RetryConfig] = None, counter=None, model="", callbacks=None, **kwargs):
    """Execute an async function with retry logic."""
    import asyncio

    config = retry_config or DEFAULT_RETRY
    last_exception = None
    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if not config.should_retry(attempt, e):
                raise
            delay = config.get_delay(attempt)
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
            if callbacks is not None:
                callbacks.fire(
                    "on_retry",
                    RetryEvent(
                        model=model,
                        attempt=attempt + 1,
                        max_retries=config.max_retries,
                        delay=delay,
                        error=e,
                    ),
                )
            await asyncio.sleep(delay)
            if counter is not None:
                counter.record_retry(model)
    raise last_exception
