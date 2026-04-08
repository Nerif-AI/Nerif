import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


def _simple_table(headers: List[str], rows: List[List[str]]) -> str:
    """Format a simple ASCII table without external dependencies."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    lines = [sep, header_line, sep]
    for row in rows:
        line = "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


class ModelCost:
    def __init__(self, model_name, request=0, response=0):
        self.model_name = model_name
        self.request = request
        self.response = response

    def add_cost(self, request, response=0):
        self.request += request
        self.response += response

    def __repr__(self) -> str:
        return f"{self.request} tokens requested, {self.response} tokens returned"


class NerifTokenConsume:
    def __init__(self):
        self.model_cost = {}

    def __getitem__(self, key):
        return self.model_cost[key]

    def append(self, consume: ModelCost):
        if consume is not None:
            model_name = consume.model_name
            if self.model_cost.get(model_name) is None:
                self.model_cost[model_name] = ModelCost(model_name)
            self.model_cost[model_name].add_cost(consume.request, consume.response)

        return self

    def __repr__(self) -> str:
        headers = ["model name", "requested tokens", "response tokens"]
        rows = [[key, str(value.request), str(value.response)] for key, value in self.model_cost.items()]
        return _simple_table(headers, rows)


class ResponseParserBase:
    def __call__(self, response) -> NerifTokenConsume:
        raise NotImplementedError("ResponseParserBase __call__ is not implemented")


class OpenAIResponseParser(ResponseParserBase):
    def __call__(self, response) -> ModelCost:
        model_name = response.model
        response_type = response.__class__.__name__
        if response_type == "EmbeddingResponse":
            requested_tokens = len(response.data[0]["embedding"])
            completation_tokens = 0
        else:
            usage = response.usage
            requested_tokens = usage.prompt_tokens
            completation_tokens = usage.completion_tokens

        consume = ModelCost(model_name, requested_tokens, completation_tokens)
        return consume


class OllamaResponseParser(ResponseParserBase):
    def __call__(self, response) -> ModelCost:
        model_name = response.model
        requested_tokens = response.prompt_eval_count
        completation_tokens = response.eval_count

        consume = ModelCost(model_name, requested_tokens, completation_tokens)
        return consume


# ---------------------------------------------------------------------------
# Observability: pricing, events, enhanced counter
# ---------------------------------------------------------------------------


@dataclass
class ModelPricing:
    """Cost per 1K tokens for a model."""

    input_cost_per_1k: float
    output_cost_per_1k: float


# Built-in pricing table (USD per 1K tokens, as of early 2026)
DEFAULT_PRICING: Dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing(0.0025, 0.01),
    "gpt-4o-mini": ModelPricing(0.00015, 0.0006),
    "gpt-4-turbo": ModelPricing(0.01, 0.03),
    "gpt-4": ModelPricing(0.03, 0.06),
    "gpt-3.5-turbo": ModelPricing(0.0005, 0.0015),
    "gpt-o1": ModelPricing(0.015, 0.06),
    "gpt-o1-mini": ModelPricing(0.003, 0.012),
    # Anthropic
    "claude-3-5-sonnet-20241022": ModelPricing(0.003, 0.015),
    "claude-3-haiku-20240307": ModelPricing(0.00025, 0.00125),
    "claude-3-opus-20240229": ModelPricing(0.015, 0.075),
    # Google
    "gemini-1.5-pro": ModelPricing(0.00125, 0.005),
    "gemini-1.5-flash": ModelPricing(0.000075, 0.0003),
    "gemini-2.0-flash": ModelPricing(0.0001, 0.0004),
    # Embeddings
    "text-embedding-3-small": ModelPricing(0.00002, 0.0),
    "text-embedding-3-large": ModelPricing(0.00013, 0.0),
}


@dataclass
class RequestStartEvent:
    """Fired before an API request."""

    model: str
    timestamp: float
    message_count: int


@dataclass
class RequestEndEvent:
    """Fired after a successful API request."""

    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    success: bool


@dataclass
class RequestErrorEvent:
    """Fired when an API request fails."""

    model: str
    error: Exception
    latency_ms: float
    will_retry: bool


class NerifTokenCounter:
    """Token counter with observability: latency, cost, success rate, and callbacks.

    Attributes:
        model_token: Accumulated token consumption by model.
    """

    def __init__(self, response_parser: ResponseParserBase = None):
        if response_parser is None:
            response_parser = OpenAIResponseParser()
        self.model_token = NerifTokenConsume()
        self.response_parser = response_parser

        # Observability stats
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
        self.retried_requests: int = 0
        self.total_latency_ms: float = 0.0
        self.latency_by_model: Dict[str, List[float]] = {}
        self.successful_by_model: Dict[str, int] = {}
        self.failed_by_model: Dict[str, int] = {}

        # Cost
        self.model_pricing: Dict[str, ModelPricing] = dict(DEFAULT_PRICING)

        # Callbacks
        self.on_request_start: Optional[Callable[[RequestStartEvent], None]] = None
        self.on_request_end: Optional[Callable[[RequestEndEvent], None]] = None
        self.on_error: Optional[Callable[[RequestErrorEvent], None]] = None
        self.callbacks = None

    def set_parser(self, parser: ResponseParserBase):
        self.response_parser = parser

    def set_parser_based_on_model(self, model_name: str):
        if model_name.startswith("ollama"):
            self.set_parser(OllamaResponseParser())
        else:
            self.set_parser(OpenAIResponseParser())

    def count_from_response(self, response):
        """Counting tokens consumed by the model from response.

        Parameters:
            response: Response object from the model.
        """
        consume = self.response_parser(response)
        self.model_token.append(consume)

    def record_request(
        self,
        model: str,
        latency_ms: float,
        success: bool,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error: Optional[Exception] = None,
    ) -> None:
        """Record a completed API request with timing and outcome.

        This is the main instrumentation entry point called from utils.py
        after each API call completes.
        """
        self.total_requests += 1
        self.total_latency_ms += latency_ms

        if model not in self.latency_by_model:
            self.latency_by_model[model] = []
        self.latency_by_model[model].append(latency_ms)

        if success:
            self.successful_requests += 1
            self.successful_by_model[model] = self.successful_by_model.get(model, 0) + 1
            cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

            if self.on_request_end is not None:
                event = RequestEndEvent(
                    model=model,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost,
                    success=True,
                )
                self.on_request_end(event)
        else:
            self.failed_requests += 1
            self.failed_by_model[model] = self.failed_by_model.get(model, 0) + 1
            if self.on_error is not None:
                event = RequestErrorEvent(
                    model=model,
                    error=error or Exception("Unknown error"),
                    latency_ms=latency_ms,
                    will_retry=False,
                )
                self.on_error(event)

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate USD cost for a request."""
        pricing = self.model_pricing.get(model)
        if pricing is None:
            # Try matching by prefix (e.g. "gpt-4o-2024-08-06" matches "gpt-4o")
            for name, p in self.model_pricing.items():
                if model.startswith(name):
                    pricing = p
                    break
        if pricing is None:
            return 0.0

        return (prompt_tokens / 1000.0) * pricing.input_cost_per_1k + (
            completion_tokens / 1000.0
        ) * pricing.output_cost_per_1k

    def total_cost(self) -> float:
        """Total estimated USD cost across all models."""
        total = 0.0
        for model_name, mc in self.model_token.model_cost.items():
            total += self._calculate_cost(model_name, mc.request, mc.response)
        return total

    def avg_latency(self, model: Optional[str] = None) -> float:
        """Average latency in milliseconds. Optionally filter by model."""
        if model is not None:
            latencies = self.latency_by_model.get(model, [])
            return sum(latencies) / len(latencies) if latencies else 0.0

        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    def success_rate(self, model: Optional[str] = None) -> float:
        """Success rate as a percentage (0-100)."""
        if model is not None:
            s = self.successful_by_model.get(model, 0)
            f = self.failed_by_model.get(model, 0)
            total = s + f
            if total == 0:
                return 100.0
            return (s / total) * 100.0
        total = self.successful_requests + self.failed_requests
        if total == 0:
            return 100.0
        return (self.successful_requests / total) * 100.0

    def reset_stats(self) -> None:
        """Clear all observability stats (not token counts)."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.retried_requests = 0
        self.total_latency_ms = 0.0
        self.latency_by_model.clear()
        self.successful_by_model.clear()
        self.failed_by_model.clear()

    def summary(self) -> str:
        """Pretty table summarizing tokens, latency, cost, and success rate."""
        headers = ["Model", "Input Tokens", "Output Tokens", "Avg Latency (ms)", "Est. Cost (USD)"]
        rows = []

        for model_name, mc in self.model_token.model_cost.items():
            latencies = self.latency_by_model.get(model_name, [])
            avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
            cost = self._calculate_cost(model_name, mc.request, mc.response)
            rows.append(
                [
                    model_name,
                    str(mc.request),
                    str(mc.response),
                    f"{avg_lat:.1f}",
                    f"${cost:.6f}",
                ]
            )

        lines = [_simple_table(headers, rows)]
        total = self.successful_requests + self.failed_requests
        if total > 0:
            lines.append(
                f"\nRequests: {total} total, {self.successful_requests} succeeded, "
                f"{self.failed_requests} failed ({self.success_rate():.1f}% success)"
            )
        lines.append(f"Total estimated cost: ${self.total_cost():.6f}")
        return "\n".join(lines)

    def record_retry(self, model: str) -> None:
        self.retried_requests += 1

    def to_dict(self) -> dict:
        """Export all counter state as a plain dictionary."""
        models = {}
        for model_name, mc in self.model_token.model_cost.items():
            latencies = self.latency_by_model.get(model_name, [])
            models[model_name] = {
                "input_tokens": mc.request,
                "output_tokens": mc.response,
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
                "estimated_cost_usd": self._calculate_cost(model_name, mc.request, mc.response),
                "successful_requests": self.successful_by_model.get(model_name, 0),
                "failed_requests": self.failed_by_model.get(model_name, 0),
            }
        return {
            "models": models,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "retried_requests": self.retried_requests,
            "total_latency_ms": self.total_latency_ms,
            "total_cost_usd": self.total_cost(),
            "success_rate": self.success_rate(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Export counter state as a JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __repr__(self) -> str:
        return repr(self.model_token)

    @staticmethod
    def _now_ms() -> float:
        """Current time in milliseconds (monotonic)."""
        return time.monotonic() * 1000.0
