"""Example 24: Observability with Enhanced TokenCounter.

Demonstrates latency tracking, cost calculation, success/failure rates,
and callback hooks for monitoring LLM API usage.
"""

from nerif.model import SimpleChatModel
from nerif.utils import NerifTokenCounter

# Create a counter with a callback
counter = NerifTokenCounter()

# Set up a callback to log each request
counter.on_request_end = lambda e: print(f"  -> {e.model}: {e.latency_ms:.0f}ms, ${e.cost_usd:.6f}")

# Use the counter with a chat model
model = SimpleChatModel(counter=counter)

print("Sending requests...")
response = model.chat("What is 2+2?")
print(f"Response: {response[:50]}...")

response = model.chat("What is the capital of France?")
print(f"Response: {response[:50]}...")

# Print summary
print("\n" + counter.summary())

# Access individual metrics
print(f"\nAverage latency: {counter.avg_latency():.1f}ms")
print(f"Success rate: {counter.success_rate():.1f}%")
print(f"Total cost: ${counter.total_cost():.6f}")
