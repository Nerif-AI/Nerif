---
sidebar_position: 2
---

# Nerif Log

`nerif log` 是一个实用工具，用于将调试/信息日志输出到文件或直接打印到终端。

# 设置日志

要启用默认日志记录器，可以使用 `nerif.core.log` 中的 `set_up_logging` 函数：

```python
import nerif.model as model
import nerif.core.log as log
log.set_up_logging(std=True)

model = model.SimpleChatModel()
print(model.chat("What is the capital of the moon?"))
```

运行代码后，你应该能在终端中看到以下信息：

```
INFO    Nerif   2024-11-15 03:00:25,630 --------------------
INFO    Nerif   2024-11-15 03:00:25,630 logging enabled
DEBUG   Nerif   2024-11-15 03:00:25,631 requested with message: [{'role': 'system', 'content': 'You are a helpful assistant. You can help me by answering my questions.'}, {'role': 'user', 'content': 'What is the capital of the moon?'}]
DEBUG   Nerif   2024-11-15 03:00:25,631 arguments of request: {'model': 'gpt-4o', 'temperature': 0.0, 'max_tokens': None}
The Moon does not have a capital. It is a natural satellite of Earth and does not have any political or administrative divisions like a country does. There are no permanent human settlements on the Moon, so it does not have a capital city.
```

设置函数有以下可选参数：
 - `out_file`：输出日志文件的名称。如果留空，日志记录器将不会创建日志文件，如上面的示例所示。
 - `time_stamp`：默认为 `False`，设置为 `True` 时，日志文件名末尾会附加时间戳。
 - `mode`：默认为 `a`，设置日志文件的写入模式，与 `open(filename, 'a')` 相同。`a` 表示在文件末尾追加新日志，`w` 表示从头覆盖同名文件中的先前内容。
 - `fmt`：每行日志的输出格式。默认为 `%(levelname)s\t%(name)s\t%(asctime)s\t%(message)s`。更多日志格式详情请参阅 Python 标准库的[文档](https://docs.python.org/3/library/logging.html#logrecord-attributes)。
 - `std`：默认为 `False`。设置为 `True` 时，日志记录器会将日志输出到标准输出（通常是终端）。
 - `level`：加载到标准输出或日志文件的调试级别。可以是表示日志级别的数字或枚举值，如 `logging.DEBUG`、`logging.INFO` 等。

# 在代码中记录日志

在记录任何内容之前，你需要创建一个 `logger` 对象以将消息发送到正确的位置。

```python
LOGGER = logging.getLogger("Nerif")
```

如果你想在代码中的某处记录日志，可以这样写：

```
things_you_want_to_log = 114514
LOGGER.debug("I am a formattable string %s", things_you_want_to_log)
```

你也可以使用 `LOGGER.info()`、`LOGGER.error()` 等方法，与 `logging` 库中的 `logger` 用法一致。更多信息请参阅[文档](https://docs.python.org/3/library/logging.html#logging.Logger.debug)。
