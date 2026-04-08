---
sidebar_position: 3
---

# Nerif Token Counter

`NerifTokenCounter` tracks token usage and also exposes lightweight observability metrics such as latency, estimated cost, success rate, retry counts, and export helpers.

## Basic usage

```python
from nerif.model import SimpleChatModel
from nerif.utils import NerifTokenCounter

counter = NerifTokenCounter()
model = SimpleChatModel(model="gpt-4o", counter=counter)

response = model.chat("Explain tracing in one sentence.")
print(response)
print(counter.summary())
print(counter.to_dict())
```

## What it tracks

- input/output tokens per model
- average latency per model
- successful and failed request counts
- retry counts
- estimated USD cost using built-in pricing

## Useful methods

- `count_from_response(response)`
- `record_request(...)`
- `record_retry(model)`
- `avg_latency(model=None)`
- `success_rate(model=None)`
- `total_cost()`
- `summary()`
- `to_dict()`
- `to_json()`
- `reset_stats()`

## Request callbacks

`NerifTokenCounter` can also emit request lifecycle callbacks:

- `on_request_start`
- `on_request_end`
- `on_error`

These are separate from the higher-level `CallbackManager` event system used by models and agents.

## Core classes

### `NerifTokenCounter`
Main counter class.

### `ResponseParserBase`
Base parser for extracting token usage from provider responses.

Derived classes:
- `OpenAIResponseParser`
- `OllamaResponseParser`

### `ModelCost`
Stores input/output token totals for a model.

### `NerifTokenConsume`
Internal container for aggregated model costs.
