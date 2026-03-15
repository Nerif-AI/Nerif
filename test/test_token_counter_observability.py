"""Tests for enhanced TokenCounter observability features (no API calls needed)."""

import pytest

from nerif.utils.token_counter import (
    DEFAULT_PRICING,
    ModelCost,
    ModelPricing,
    NerifTokenCounter,
    RequestEndEvent,
    RequestErrorEvent,
    RequestStartEvent,
)


class TestRecordRequest:
    def test_record_success(self):
        counter = NerifTokenCounter()
        counter.record_request("gpt-4o", latency_ms=150.0, success=True, prompt_tokens=100, completion_tokens=50)

        assert counter.total_requests == 1
        assert counter.successful_requests == 1
        assert counter.failed_requests == 0
        assert counter.total_latency_ms == 150.0
        assert "gpt-4o" in counter.latency_by_model
        assert counter.latency_by_model["gpt-4o"] == [150.0]

    def test_record_failure(self):
        counter = NerifTokenCounter()
        counter.record_request("gpt-4o", latency_ms=50.0, success=False, error=Exception("timeout"))

        assert counter.total_requests == 1
        assert counter.successful_requests == 0
        assert counter.failed_requests == 1

    def test_multiple_records(self):
        counter = NerifTokenCounter()
        counter.record_request("gpt-4o", latency_ms=100.0, success=True, prompt_tokens=50, completion_tokens=20)
        counter.record_request("gpt-4o", latency_ms=200.0, success=True, prompt_tokens=60, completion_tokens=30)
        counter.record_request("gpt-4o-mini", latency_ms=50.0, success=True, prompt_tokens=40, completion_tokens=10)

        assert counter.total_requests == 3
        assert counter.successful_requests == 3
        assert len(counter.latency_by_model["gpt-4o"]) == 2
        assert len(counter.latency_by_model["gpt-4o-mini"]) == 1


class TestLatency:
    def test_avg_latency_overall(self):
        counter = NerifTokenCounter()
        counter.record_request("gpt-4o", latency_ms=100.0, success=True)
        counter.record_request("gpt-4o", latency_ms=200.0, success=True)

        assert counter.avg_latency() == 150.0

    def test_avg_latency_by_model(self):
        counter = NerifTokenCounter()
        counter.record_request("gpt-4o", latency_ms=100.0, success=True)
        counter.record_request("gpt-4o", latency_ms=200.0, success=True)
        counter.record_request("gpt-4o-mini", latency_ms=50.0, success=True)

        assert counter.avg_latency("gpt-4o") == 150.0
        assert counter.avg_latency("gpt-4o-mini") == 50.0

    def test_avg_latency_empty(self):
        counter = NerifTokenCounter()
        assert counter.avg_latency() == 0.0
        assert counter.avg_latency("nonexistent") == 0.0


class TestCost:
    def test_calculate_cost_known_model(self):
        counter = NerifTokenCounter()
        # gpt-4o: $0.0025/1K input, $0.01/1K output
        cost = counter._calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=1000)
        assert cost == pytest.approx(0.0025 + 0.01)

    def test_calculate_cost_unknown_model(self):
        counter = NerifTokenCounter()
        cost = counter._calculate_cost("unknown-model", prompt_tokens=1000, completion_tokens=1000)
        assert cost == 0.0

    def test_calculate_cost_prefix_match(self):
        counter = NerifTokenCounter()
        # "gpt-4o-2024-08-06" should match "gpt-4o" pricing
        cost = counter._calculate_cost("gpt-4o-2024-08-06", prompt_tokens=1000, completion_tokens=1000)
        assert cost == pytest.approx(0.0025 + 0.01)

    def test_total_cost(self):
        counter = NerifTokenCounter()
        # Simulate token tracking
        counter.model_token.append(ModelCost("gpt-4o", request=1000, response=500))
        counter.model_token.append(ModelCost("gpt-4o-mini", request=2000, response=1000))

        total = counter.total_cost()
        expected_4o = (1000 / 1000.0) * 0.0025 + (500 / 1000.0) * 0.01
        expected_mini = (2000 / 1000.0) * 0.00015 + (1000 / 1000.0) * 0.0006
        assert total == pytest.approx(expected_4o + expected_mini)

    def test_custom_pricing(self):
        counter = NerifTokenCounter()
        counter.model_pricing["my-custom-model"] = ModelPricing(0.001, 0.002)

        cost = counter._calculate_cost("my-custom-model", prompt_tokens=1000, completion_tokens=500)
        assert cost == pytest.approx(0.001 + 0.001)


class TestSuccessRate:
    def test_all_success(self):
        counter = NerifTokenCounter()
        counter.record_request("gpt-4o", latency_ms=100.0, success=True)
        counter.record_request("gpt-4o", latency_ms=100.0, success=True)

        assert counter.success_rate() == 100.0

    def test_mixed(self):
        counter = NerifTokenCounter()
        counter.record_request("gpt-4o", latency_ms=100.0, success=True)
        counter.record_request("gpt-4o", latency_ms=100.0, success=False)

        assert counter.success_rate() == 50.0

    def test_empty(self):
        counter = NerifTokenCounter()
        assert counter.success_rate() == 100.0


class TestCallbacks:
    def test_on_request_end_callback(self):
        events = []
        counter = NerifTokenCounter()
        counter.on_request_end = lambda e: events.append(e)

        counter.record_request("gpt-4o", latency_ms=100.0, success=True, prompt_tokens=50, completion_tokens=20)

        assert len(events) == 1
        assert isinstance(events[0], RequestEndEvent)
        assert events[0].model == "gpt-4o"
        assert events[0].latency_ms == 100.0
        assert events[0].prompt_tokens == 50
        assert events[0].completion_tokens == 20
        assert events[0].success is True
        assert events[0].cost_usd > 0

    def test_on_error_callback(self):
        errors = []
        counter = NerifTokenCounter()
        counter.on_error = lambda e: errors.append(e)

        counter.record_request("gpt-4o", latency_ms=50.0, success=False, error=TimeoutError("timed out"))

        assert len(errors) == 1
        assert isinstance(errors[0], RequestErrorEvent)
        assert errors[0].model == "gpt-4o"
        assert isinstance(errors[0].error, TimeoutError)

    def test_no_callback_no_error(self):
        counter = NerifTokenCounter()
        # Should not raise even without callbacks set
        counter.record_request("gpt-4o", latency_ms=100.0, success=True)
        counter.record_request("gpt-4o", latency_ms=50.0, success=False)


class TestResetStats:
    def test_reset_clears_observability(self):
        counter = NerifTokenCounter()
        counter.record_request("gpt-4o", latency_ms=100.0, success=True)
        counter.record_request("gpt-4o", latency_ms=200.0, success=False)

        counter.reset_stats()

        assert counter.total_requests == 0
        assert counter.successful_requests == 0
        assert counter.failed_requests == 0
        assert counter.total_latency_ms == 0.0
        assert counter.latency_by_model == {}

    def test_reset_preserves_token_counts(self):
        counter = NerifTokenCounter()
        counter.model_token.append(ModelCost("gpt-4o", request=100, response=50))
        counter.record_request("gpt-4o", latency_ms=100.0, success=True)

        counter.reset_stats()

        # Token counts should be preserved
        assert counter.model_token["gpt-4o"].request == 100


class TestSummary:
    def test_summary_format(self):
        counter = NerifTokenCounter()
        counter.model_token.append(ModelCost("gpt-4o", request=1000, response=500))
        counter.record_request("gpt-4o", latency_ms=150.0, success=True, prompt_tokens=1000, completion_tokens=500)

        output = counter.summary()
        assert "gpt-4o" in output
        assert "1000" in output
        assert "500" in output
        assert "$" in output
        assert "success" in output.lower()

    def test_summary_empty(self):
        counter = NerifTokenCounter()
        output = counter.summary()
        assert "Total estimated cost" in output


class TestBackwardCompatibility:
    def test_existing_count_from_response_works(self):
        """Ensure existing count_from_response API is unchanged."""
        counter = NerifTokenCounter()

        # Create a mock response object
        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50

        class MockResponse:
            model = "gpt-4o"
            usage = MockUsage()

        counter.count_from_response(MockResponse())
        assert counter.model_token["gpt-4o"].request == 100
        assert counter.model_token["gpt-4o"].response == 50

    def test_repr_unchanged(self):
        """__repr__ should still delegate to model_token."""
        counter = NerifTokenCounter()
        counter.model_token.append(ModelCost("gpt-4o", request=100, response=50))
        output = repr(counter)
        assert "gpt-4o" in output
        assert "100" in output

    def test_set_parser_still_works(self):
        counter = NerifTokenCounter()
        counter.set_parser_based_on_model("ollama/llama3.1")
        assert counter.response_parser.__class__.__name__ == "OllamaResponseParser"


class TestDefaultPricing:
    def test_has_common_models(self):
        assert "gpt-4o" in DEFAULT_PRICING
        assert "gpt-4o-mini" in DEFAULT_PRICING
        assert "text-embedding-3-small" in DEFAULT_PRICING

    def test_pricing_values_positive(self):
        for model, pricing in DEFAULT_PRICING.items():
            assert pricing.input_cost_per_1k >= 0, f"{model} has negative input cost"
            assert pricing.output_cost_per_1k >= 0, f"{model} has negative output cost"


class TestRequestStartEvent:
    def test_dataclass_fields(self):
        event = RequestStartEvent(model="gpt-4o", timestamp=1234.0, message_count=3)
        assert event.model == "gpt-4o"
        assert event.timestamp == 1234.0
        assert event.message_count == 3
