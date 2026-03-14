"""Example: Configurable retry with exponential backoff.

Demonstrates RetryConfig for automatic retries on API failures (429, 500, etc.)
with exponential backoff, jitter, and Retry-After header support.
"""

from nerif.model import SimpleChatModel
from nerif.utils import AGGRESSIVE_RETRY, NO_RETRY, RetryConfig

# Default: 3 retries with exponential backoff
model = SimpleChatModel()
print(model.chat("Hello"))

# No retry - fail immediately on error
model_no_retry = SimpleChatModel(retry_config=NO_RETRY)

# Aggressive retry - 5 retries, faster base delay
model_aggressive = SimpleChatModel(retry_config=AGGRESSIVE_RETRY)

# Custom retry config
custom_config = RetryConfig(
    max_retries=5,
    base_delay=0.5,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True,
    retryable_status_codes={429, 500, 502, 503},
    retry_on_timeout=True,
)
model_custom = SimpleChatModel(retry_config=custom_config)
print(model_custom.chat("Describe retry patterns in distributed systems."))
