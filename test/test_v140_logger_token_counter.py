"""Tests for v1.4.0 logger and token counter improvements."""

import json
import logging
import os

import pytest

from nerif.utils.log import JsonFormatter, enable_debug_logging, set_up_logging
from nerif.utils.token_counter import (
    ModelCost,
    NerifTokenCounter,
)

# ---------------------------------------------------------------------------
# Logger tests
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    def test_basic_output(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="nerif", level=logging.INFO, pathname="", lineno=0,
            msg="hello world", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "nerif"
        assert parsed["message"] == "hello world"
        assert "timestamp" in parsed

    def test_exception_included(self):
        formatter = JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="nerif", level=logging.ERROR, pathname="", lineno=0,
            msg="failed", args=(), exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


class TestSetUpLogging:
    def test_json_format_to_file(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        logger = logging.getLogger("nerif.test_json")
        logger.handlers.clear()

        set_up_logging(out_file=log_file, time_stamp=False, json_format=True, level=logging.DEBUG)

        # The "nerif" logger should have handlers now
        nerif_logger = logging.getLogger("nerif")
        initial_handler_count = len(nerif_logger.handlers)
        assert initial_handler_count > 0

        # Clean up handlers to avoid side effects
        for h in nerif_logger.handlers[:]:
            if isinstance(h, logging.FileHandler) and log_file in getattr(h, 'baseFilename', ''):
                nerif_logger.removeHandler(h)
                h.close()

    def test_rotating_file_handler(self, tmp_path):
        log_file = str(tmp_path / "rotating.log")
        nerif_logger = logging.getLogger("nerif")
        before = len(nerif_logger.handlers)

        set_up_logging(out_file=log_file, time_stamp=False, max_bytes=1024, backup_count=3)

        # Should have added a RotatingFileHandler
        new_handlers = nerif_logger.handlers[before:]
        rotating = [h for h in new_handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(rotating) == 1
        assert rotating[0].maxBytes == 1024
        assert rotating[0].backupCount == 3

        # Clean up
        for h in new_handlers:
            nerif_logger.removeHandler(h)
            h.close()


class TestEnableDebugLogging:
    def test_adds_stdout_handler(self):
        nerif_logger = logging.getLogger("nerif")
        before = len(nerif_logger.handlers)

        enable_debug_logging()

        assert len(nerif_logger.handlers) > before
        assert nerif_logger.level <= logging.DEBUG

        # Clean up
        for h in nerif_logger.handlers[before:]:
            nerif_logger.removeHandler(h)


class TestEnvVarConfig:
    def test_nerif_log_level_env(self):
        # This tests that _auto_configure reads env vars.
        # We can't easily test the import-time behavior, but we can test the function.
        from nerif.utils.log import _auto_configure

        nerif_logger = logging.getLogger("nerif")
        before = len(nerif_logger.handlers)

        old_level = os.environ.get("NERIF_LOG_LEVEL")
        old_file = os.environ.get("NERIF_LOG_FILE")
        try:
            os.environ["NERIF_LOG_LEVEL"] = "WARNING"
            if "NERIF_LOG_FILE" in os.environ:
                del os.environ["NERIF_LOG_FILE"]
            _auto_configure()
        finally:
            if old_level is not None:
                os.environ["NERIF_LOG_LEVEL"] = old_level
            elif "NERIF_LOG_LEVEL" in os.environ:
                del os.environ["NERIF_LOG_LEVEL"]
            if old_file is not None:
                os.environ["NERIF_LOG_FILE"] = old_file

        # Clean up handlers
        for h in nerif_logger.handlers[before:]:
            nerif_logger.removeHandler(h)


# ---------------------------------------------------------------------------
# Token Counter tests
# ---------------------------------------------------------------------------


class TestPerModelSuccessRate:
    def test_per_model_tracking(self):
        counter = NerifTokenCounter()
        counter.record_request("gpt-4o", latency_ms=100, success=True)
        counter.record_request("gpt-4o", latency_ms=100, success=True)
        counter.record_request("gpt-4o", latency_ms=100, success=False)
        counter.record_request("gpt-4o-mini", latency_ms=50, success=True)

        assert counter.success_rate("gpt-4o") == pytest.approx(66.666, abs=0.1)
        assert counter.success_rate("gpt-4o-mini") == 100.0
        assert counter.success_rate("nonexistent") == 100.0

    def test_per_model_counts(self):
        counter = NerifTokenCounter()
        counter.record_request("gpt-4o", latency_ms=100, success=True)
        counter.record_request("gpt-4o", latency_ms=100, success=False)

        assert counter.successful_by_model["gpt-4o"] == 1
        assert counter.failed_by_model["gpt-4o"] == 1

    def test_reset_clears_per_model(self):
        counter = NerifTokenCounter()
        counter.record_request("gpt-4o", latency_ms=100, success=True)
        counter.record_request("gpt-4o", latency_ms=100, success=False)

        counter.reset_stats()

        assert counter.successful_by_model == {}
        assert counter.failed_by_model == {}


class TestToDict:
    def test_basic_export(self):
        counter = NerifTokenCounter()
        counter.model_token.append(ModelCost("gpt-4o", request=1000, response=500))
        counter.record_request("gpt-4o", latency_ms=150, success=True, prompt_tokens=1000, completion_tokens=500)

        d = counter.to_dict()
        assert "models" in d
        assert "gpt-4o" in d["models"]
        assert d["models"]["gpt-4o"]["input_tokens"] == 1000
        assert d["models"]["gpt-4o"]["output_tokens"] == 500
        assert d["models"]["gpt-4o"]["successful_requests"] == 1
        assert d["models"]["gpt-4o"]["failed_requests"] == 0
        assert d["total_requests"] == 1
        assert d["successful_requests"] == 1
        assert d["total_cost_usd"] > 0
        assert d["success_rate"] == 100.0

    def test_empty_export(self):
        counter = NerifTokenCounter()
        d = counter.to_dict()
        assert d["models"] == {}
        assert d["total_requests"] == 0


class TestToJson:
    def test_valid_json(self):
        counter = NerifTokenCounter()
        counter.model_token.append(ModelCost("gpt-4o", request=100, response=50))
        counter.record_request("gpt-4o", latency_ms=100, success=True, prompt_tokens=100, completion_tokens=50)

        output = counter.to_json()
        parsed = json.loads(output)
        assert parsed["models"]["gpt-4o"]["input_tokens"] == 100

    def test_indent_parameter(self):
        counter = NerifTokenCounter()
        compact = counter.to_json(indent=None)
        assert "\n" not in compact


class TestContextManager:
    def test_enter_exit(self):
        counter = NerifTokenCounter()
        with counter as c:
            assert c is counter
            c.record_request("gpt-4o", latency_ms=100, success=True)

        assert counter.total_requests == 1

    def test_exception_propagates(self):
        counter = NerifTokenCounter()
        with pytest.raises(ValueError, match="test"):
            with counter:
                raise ValueError("test")


class TestRecordRetry:
    def test_increments_counter(self):
        counter = NerifTokenCounter()
        assert counter.retried_requests == 0

        counter.record_retry("gpt-4o")
        counter.record_retry("gpt-4o")
        counter.record_retry("gpt-4o-mini")

        assert counter.retried_requests == 3


class TestRetryIntegration:
    def test_retry_sync_records_retries(self):
        from nerif.utils.retry import RetryConfig, retry_sync

        counter = NerifTokenCounter()
        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                import httpx
                resp = httpx.Response(500, request=httpx.Request("POST", "http://x"))
                raise httpx.HTTPStatusError("fail", request=resp.request, response=resp)
            return "success"

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        result = retry_sync(flaky_func, retry_config=config, counter=counter, model="gpt-4o")

        assert result == "success"
        assert counter.retried_requests == 2  # retried twice before succeeding
