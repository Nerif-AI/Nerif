"""Tests for the retry mechanism in nerif.utils.retry."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from nerif.utils.callbacks import CallbackHandler, CallbackManager
from nerif.utils.retry import (
    AGGRESSIVE_RETRY,
    DEFAULT_RETRY,
    NO_RETRY,
    RetryConfig,
    retry_async,
    retry_sync,
)


class _RetryRecordingHandler(CallbackHandler):
    def __init__(self):
        self.events = []

    def on_retry(self, event):
        self.events.append(event)


def test_retry_config_defaults():
    config = RetryConfig()
    assert config.max_retries == 3
    assert config.base_delay == 1.0
    assert config.max_delay == 60.0
    assert config.exponential_base == 2.0
    assert config.jitter is True
    assert config.retry_on_timeout is True
    assert 429 in config.retryable_status_codes
    assert 500 in config.retryable_status_codes
    assert 502 in config.retryable_status_codes
    assert 503 in config.retryable_status_codes
    assert 504 in config.retryable_status_codes


def test_retry_config_delay_no_jitter():
    config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=60.0, jitter=False)
    assert config.get_delay(0) == 1.0
    assert config.get_delay(1) == 2.0
    assert config.get_delay(2) == 4.0
    assert config.get_delay(3) == 8.0


def test_retry_config_delay_capped_at_max():
    config = RetryConfig(base_delay=10.0, exponential_base=2.0, max_delay=15.0, jitter=False)
    assert config.get_delay(0) == 10.0
    assert config.get_delay(1) == 15.0  # capped
    assert config.get_delay(2) == 15.0  # still capped


def test_retry_config_delay_with_jitter():
    config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=60.0, jitter=True)
    # With jitter, delay is in [0.5 * base, 1.0 * base]
    for _ in range(20):
        delay = config.get_delay(0)
        assert 0.5 <= delay <= 1.0


def test_preset_no_retry():
    assert NO_RETRY.max_retries == 0


def test_preset_default_retry():
    assert DEFAULT_RETRY.max_retries == 3


def test_preset_aggressive_retry():
    assert AGGRESSIVE_RETRY.max_retries == 5
    assert AGGRESSIVE_RETRY.base_delay == 0.5


# ---------------------------------------------------------------------------
# should_retry logic
# ---------------------------------------------------------------------------


def _make_status_error(status_code: int) -> httpx.HTTPStatusError:
    """Helper to create an httpx.HTTPStatusError with a given status code."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.headers = {}
    return httpx.HTTPStatusError("error", request=MagicMock(), response=mock_response)


def test_should_retry_429():
    config = RetryConfig()
    exc = _make_status_error(429)
    assert config.should_retry(0, exc) is True


def test_should_retry_500():
    config = RetryConfig()
    exc = _make_status_error(500)
    assert config.should_retry(0, exc) is True


def test_should_retry_502():
    config = RetryConfig()
    exc = _make_status_error(502)
    assert config.should_retry(0, exc) is True


def test_should_retry_503():
    config = RetryConfig()
    exc = _make_status_error(503)
    assert config.should_retry(0, exc) is True


def test_should_retry_504():
    config = RetryConfig()
    exc = _make_status_error(504)
    assert config.should_retry(0, exc) is True


def test_should_not_retry_400():
    config = RetryConfig()
    exc = _make_status_error(400)
    assert config.should_retry(0, exc) is False


def test_should_not_retry_401():
    config = RetryConfig()
    exc = _make_status_error(401)
    assert config.should_retry(0, exc) is False


def test_should_not_retry_403():
    config = RetryConfig()
    exc = _make_status_error(403)
    assert config.should_retry(0, exc) is False


def test_should_not_retry_404():
    config = RetryConfig()
    exc = _make_status_error(404)
    assert config.should_retry(0, exc) is False


def test_should_retry_timeout():
    config = RetryConfig()
    exc = httpx.ReadTimeout("timed out", request=MagicMock())
    assert config.should_retry(0, exc) is True


def test_should_not_retry_timeout_when_disabled():
    config = RetryConfig(retry_on_timeout=False)
    exc = httpx.ReadTimeout("timed out", request=MagicMock())
    assert config.should_retry(0, exc) is False


def test_should_retry_connect_error():
    config = RetryConfig()
    exc = httpx.ConnectError("connection refused")
    assert config.should_retry(0, exc) is True


def test_should_not_retry_max_retries_exceeded():
    config = RetryConfig(max_retries=3)
    exc = _make_status_error(503)
    # attempt == max_retries means we've exhausted retries
    assert config.should_retry(3, exc) is False
    # but attempt < max_retries should still retry
    assert config.should_retry(2, exc) is True


# ---------------------------------------------------------------------------
# retry_sync: basic behavior
# ---------------------------------------------------------------------------


def test_retry_sync_succeeds_without_retry():
    call_count = 0

    def func():
        nonlocal call_count
        call_count += 1
        return "success"

    result = retry_sync(func, retry_config=NO_RETRY)
    assert result == "success"
    assert call_count == 1


def test_retry_sync_no_retry_raises_immediately():
    call_count = 0

    def func():
        nonlocal call_count
        call_count += 1
        raise _make_status_error(503)

    with pytest.raises(httpx.HTTPStatusError):
        retry_sync(func, retry_config=NO_RETRY)
    assert call_count == 1


@patch("nerif.utils.retry.time.sleep")
def test_retry_sync_retries_on_429_and_succeeds(mock_sleep):
    call_count = 0

    def func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _make_status_error(429)
        return "ok"

    config = RetryConfig(max_retries=3, jitter=False)
    result = retry_sync(func, retry_config=config)
    assert result == "ok"
    assert call_count == 3
    assert mock_sleep.call_count == 2


@patch("nerif.utils.retry.time.sleep")
def test_retry_sync_raises_after_max_retries(mock_sleep):
    call_count = 0

    def func():
        nonlocal call_count
        call_count += 1
        raise _make_status_error(503)

    config = RetryConfig(max_retries=2, jitter=False)
    with pytest.raises(httpx.HTTPStatusError):
        retry_sync(func, retry_config=config)
    # initial attempt + 2 retries = 3 total calls
    assert call_count == 3
    assert mock_sleep.call_count == 2


@patch("nerif.utils.retry.time.sleep")
def test_retry_sync_non_retryable_raises_immediately(mock_sleep):
    call_count = 0

    def func():
        nonlocal call_count
        call_count += 1
        raise _make_status_error(404)

    config = RetryConfig(max_retries=3, jitter=False)
    with pytest.raises(httpx.HTTPStatusError):
        retry_sync(func, retry_config=config)
    assert call_count == 1
    mock_sleep.assert_not_called()


@patch("nerif.utils.retry.time.sleep")
def test_retry_sync_respects_retry_after_header(mock_sleep):
    """When server returns Retry-After header, delay should be at least that value."""
    call_count = 0

    def func():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 429
            mock_response.headers = {"retry-after": "5"}
            raise httpx.HTTPStatusError("rate limited", request=MagicMock(), response=mock_response)
        return "done"

    # Use config with no jitter and small base delay so Retry-After dominates
    config = RetryConfig(max_retries=2, base_delay=0.1, jitter=False)
    result = retry_sync(func, retry_config=config)
    assert result == "done"
    # sleep should have been called with at least 5.0 seconds
    sleep_arg = mock_sleep.call_args[0][0]
    assert sleep_arg >= 5.0


@patch("nerif.utils.retry.time.sleep")
def test_retry_sync_uses_default_retry_when_none(mock_sleep):
    """When retry_config is None, DEFAULT_RETRY should be used."""
    call_count = 0

    def func():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_status_error(503)
        return "ok"

    result = retry_sync(func, retry_config=None)
    assert result == "ok"
    assert call_count == 2


@patch("nerif.utils.retry.time.sleep")
def test_retry_sync_emits_retry_event(mock_sleep):
    handler = _RetryRecordingHandler()
    callbacks = CallbackManager()
    callbacks.add_handler(handler)
    call_count = 0

    def func():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_status_error(503)
        return "ok"

    result = retry_sync(func, retry_config=RetryConfig(max_retries=1, jitter=False), model="gpt-4o", callbacks=callbacks)
    assert result == "ok"
    assert len(handler.events) == 1
    assert handler.events[0].model == "gpt-4o"
    assert handler.events[0].attempt == 1


@pytest.mark.asyncio
@patch("nerif.utils.retry.time.sleep")
async def test_retry_async_succeeds_without_retry(mock_sleep):
    call_count = 0

    async def func():
        nonlocal call_count
        call_count += 1
        return "async ok"

    result = await retry_async(func, retry_config=NO_RETRY)
    assert result == "async ok"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_async_retries_on_error():
    call_count = 0

    async def func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise _make_status_error(500)
        return "async done"

    config = RetryConfig(max_retries=3, base_delay=0.0, jitter=False)
    result = await retry_async(func, retry_config=config)
    assert result == "async done"
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_async_raises_after_max_retries():
    call_count = 0

    async def func():
        nonlocal call_count
        call_count += 1
        raise _make_status_error(502)

    config = RetryConfig(max_retries=1, base_delay=0.0, jitter=False)
    with pytest.raises(httpx.HTTPStatusError):
        await retry_async(func, retry_config=config)
    assert call_count == 2  # initial + 1 retry


# ---------------------------------------------------------------------------
# Integration: get_model_response with retry_config
# ---------------------------------------------------------------------------


@patch("nerif.utils.retry.time.sleep")
@patch("nerif.utils.utils._openai_compatible_completion")
def test_get_model_response_passes_retry_config(mock_completion, mock_sleep):
    """get_model_response should retry on transient errors when retry_config is set."""
    from nerif.utils.utils import get_model_response

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _make_status_error(503)
        # Return a minimal valid response
        from nerif.utils.utils import ChatCompletionResponse, _Choice, _Message

        return ChatCompletionResponse(choices=[_Choice(message=_Message(role="assistant", content="hello"))])

    mock_completion.side_effect = side_effect

    config = RetryConfig(max_retries=3, jitter=False)
    result = get_model_response(
        [{"role": "user", "content": "hi"}],
        model="gpt-4o",
        retry_config=config,
    )
    assert result.choices[0].message.content == "hello"
    assert call_count == 3


@patch("nerif.utils.retry.time.sleep")
@patch("nerif.utils.utils._openai_compatible_completion")
def test_get_model_response_no_retry_config_uses_default(mock_completion, mock_sleep):
    """When retry_config=None, DEFAULT_RETRY (3 retries) is used."""
    from nerif.utils.utils import get_model_response

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise _make_status_error(503)
        from nerif.utils.utils import ChatCompletionResponse, _Choice, _Message

        return ChatCompletionResponse(choices=[_Choice(message=_Message(role="assistant", content="hi"))])

    mock_completion.side_effect = side_effect

    result = get_model_response(
        [{"role": "user", "content": "hello"}],
        model="gpt-4o",
        retry_config=None,
    )
    assert result.choices[0].message.content == "hi"
    assert call_count == 2


# ---------------------------------------------------------------------------
# Integration: SimpleChatModel with retry_config
# ---------------------------------------------------------------------------


@patch("nerif.utils.retry.time.sleep")
@patch("nerif.utils.utils._openai_compatible_completion")
def test_simple_chat_model_with_retry_config(mock_completion, mock_sleep):
    """SimpleChatModel should pass retry_config to get_model_response."""
    from nerif.model.model import SimpleChatModel

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise _make_status_error(429)
        from nerif.utils.utils import ChatCompletionResponse, _Choice, _Message

        return ChatCompletionResponse(choices=[_Choice(message=_Message(role="assistant", content="replied"))])

    mock_completion.side_effect = side_effect

    config = RetryConfig(max_retries=3, jitter=False)
    model = SimpleChatModel(model="gpt-4o", retry_config=config)
    response = model.chat("hello")
    assert response == "replied"
    assert call_count == 2
    assert mock_sleep.call_count == 1


@patch("nerif.utils.utils._openai_compatible_completion")
def test_simple_chat_model_no_retry_config(mock_completion):
    """SimpleChatModel with no retry_config still works (uses DEFAULT_RETRY internally)."""
    from nerif.model.model import SimpleChatModel
    from nerif.utils.utils import ChatCompletionResponse, _Choice, _Message

    mock_completion.return_value = ChatCompletionResponse(
        choices=[_Choice(message=_Message(role="assistant", content="ok"))]
    )

    model = SimpleChatModel(model="gpt-4o")
    assert model.retry_config is None
    response = model.chat("hello")
    assert response == "ok"
