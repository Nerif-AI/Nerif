---
sidebar_position: 2
---

# Nerif Log

`nerif` provides logging helpers for stdout/file logging and JSON-formatted logs.

## Setup

Use `set_up_logging` from `nerif.utils` or `nerif.utils.log`.

```python
from nerif.model import SimpleChatModel
from nerif.utils import set_up_logging

set_up_logging(std=True)

model = SimpleChatModel()
print(model.chat("What is the capital of the moon?"))
```

You can also enable logging via environment variables:

- `NERIF_LOG_LEVEL`
- `NERIF_LOG_FILE`

## Options

`set_up_logging(...)` supports:

- `out_file`: output log file path
- `time_stamp`: whether to append a timestamp to the filename
- `mode`: file open mode, such as `"a"` or `"w"`
- `fmt`: log line format
- `std`: whether to also print logs to stdout
- `level`: Python logging level
- `json_format`: emit structured JSON logs
- `max_bytes` / `backup_count`: rotating file handler settings

## Example JSON logging

```python
from nerif.utils import set_up_logging

set_up_logging(std=True, json_format=True)
```

## Logging in your own code

```python
import logging

LOGGER = logging.getLogger("Nerif")
LOGGER.debug("formatted value: %s", 42)
```
