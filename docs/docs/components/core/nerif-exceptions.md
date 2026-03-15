---
sidebar_position: 6
---

# Error Handling

Nerif provides a structured exception hierarchy for consistent error handling.

## Exception Hierarchy

```
NerifError (base)
├── ProviderError      - API errors (rate limits, auth failures)
├── FormatError        - Output parsing failures (also ValueError)
├── ConversationMemoryError - Memory save/load errors
├── ConfigError        - Missing API keys, invalid config
│   └── ModelNotFoundError - Unknown model name
└── TokenLimitError    - Token limit exceeded
```

## Usage

```python
from nerif.exceptions import NerifError, ProviderError, ConfigError

try:
    model.chat("Hello")
except ProviderError as e:
    print(f"Provider {e.provider} error: {e.status_code}")
except NerifError as e:
    print(f"Nerif error: {e}")
```

`FormatError` is also a `ValueError` for backward compatibility.
