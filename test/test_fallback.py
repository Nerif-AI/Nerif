"""Tests for model fallback chain."""

import httpx

from nerif.utils.fallback import FallbackConfig


class TestFallbackConfig:
    def test_should_fallback_on_429(self):
        cfg = FallbackConfig(models=["a", "b"])
        resp = httpx.Response(429, request=httpx.Request("POST", "http://x"))
        exc = httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
        assert cfg.should_fallback(exc) is True

    def test_should_fallback_on_500(self):
        cfg = FallbackConfig(models=["a", "b"])
        resp = httpx.Response(500, request=httpx.Request("POST", "http://x"))
        exc = httpx.HTTPStatusError("server error", request=resp.request, response=resp)
        assert cfg.should_fallback(exc) is True

    def test_should_not_fallback_on_400(self):
        cfg = FallbackConfig(models=["a", "b"])
        resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
        exc = httpx.HTTPStatusError("bad request", request=resp.request, response=resp)
        assert cfg.should_fallback(exc) is False

    def test_should_fallback_on_timeout(self):
        cfg = FallbackConfig(models=["a", "b"])
        exc = httpx.ReadTimeout("timeout")
        assert cfg.should_fallback(exc) is True

    def test_should_not_fallback_on_timeout_when_disabled(self):
        cfg = FallbackConfig(models=["a", "b"], fallback_on_timeout=False)
        exc = httpx.ReadTimeout("timeout")
        assert cfg.should_fallback(exc) is False

    def test_should_fallback_on_connect_error(self):
        cfg = FallbackConfig(models=["a", "b"])
        exc = httpx.ConnectError("connection refused")
        assert cfg.should_fallback(exc) is True

    def test_custom_status_codes(self):
        cfg = FallbackConfig(models=["a", "b"], fallback_on={502})
        resp = httpx.Response(429, request=httpx.Request("POST", "http://x"))
        exc = httpx.HTTPStatusError("", request=resp.request, response=resp)
        assert cfg.should_fallback(exc) is False

    def test_unknown_exception_no_fallback(self):
        cfg = FallbackConfig(models=["a", "b"])
        assert cfg.should_fallback(ValueError("bad")) is False
