---
sidebar_position: 2
---

# Nerif Log

`nerif log` is the utility tool to output certain debug/info message to a file, or just printing it out.

# setting logs up

to turn on the default logger, you could use the `set_up_logging` function in `nerif.core.log`

```python
import nerif.model as model
import nerif.core.log as log
log.set_up_logging(std=True)

model = model.SimpleChatModel()
print(model.chat("What is the capital of the moon?"))
```

once you run the code, you should be able to see following message in your terminal:

```
INFO    Nerif   2024-11-15 03:00:25,630 --------------------
INFO    Nerif   2024-11-15 03:00:25,630 logging enabled
DEBUG   Nerif   2024-11-15 03:00:25,631 requested with message: [{'role': 'system', 'content': 'You are a helpful assistant. You can help me by answering my questions.'}, {'role': 'user', 'content': 'What is the capital of the moon?'}]
DEBUG   Nerif   2024-11-15 03:00:25,631 arguments of request: {'model': 'gpt-4o', 'temperature': 0.0, 'max_tokens': None}
The Moon does not have a capital. It is a natural satellite of Earth and does not have any political or administrative divisions like a country does. There are no permanent human settlements on the Moon, so it does not have a capital city.
```

There are several optional values for setup function to play with:
 - `out_file`: the name of the output log file. If you leave this value empty, the logger won't create a log file as we have done above.
 - `time_stamp`: defaultly `False`, if setting it to true, filename of the log will have a time stamp at the end.
 - `mode`: defaultly `a`, setting the writing mode to the log file as `open(filename, 'a')` does. `a` means adding new log at the end of a file, `w` means overwrite the previous content in the file with same filename from the start.
 - `fmt`: the output format of every line of log. defaultly `%(levelname)s\t%(name)s\t%(asctime)s\t%(message)s`. please check python standard lib [documentation](https://docs.python.org/3/library/logging.html#logrecord-attributes) for logging format for more details
 - `std`: defaultly `False`. If set to true the logger would output lines of log into standard output, which in most case is terminal
 - `level`: the level of debug would be loaded into stdout or logfile. could be number representing log level or enum value like `logging.DEBUG`,  `logging.INFO`, and etc.

# logging thing in your code

before logging anything in your own logger, you need to create a `logger` object to send message to correct place.

```python
LOGGER = logging.getLogger("Nerif")
```

if you want to log anything somewhere in your code, write:

```
things_you_want_to_log = 114514
LOGGER.debug("I am a formattable string %s", things_you_want_to_log)
```

you could also use `LOGGER.info()`, `LOGGER.error()` etc. as the `logger` in `logging` library, check the [documentation](https://docs.python.org/3/library/logging.html#logging.Logger.debug) for further information