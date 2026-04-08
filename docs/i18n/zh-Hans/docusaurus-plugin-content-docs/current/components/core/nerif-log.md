---
sidebar_position: 2
---

# Nerif Log

`nerif` 提供了标准输出、文件输出和 JSON 结构化日志的辅助工具。

## 设置

使用 `nerif.utils` 或 `nerif.utils.log` 中的 `set_up_logging`。

```python
from nerif.model import SimpleChatModel
from nerif.utils import set_up_logging

set_up_logging(std=True)

model = SimpleChatModel()
print(model.chat("What is the capital of the moon?"))
```

也可以通过环境变量启用日志：

- `NERIF_LOG_LEVEL`
- `NERIF_LOG_FILE`

## 可选参数

`set_up_logging(...)` 支持：

- `out_file`：日志文件路径
- `time_stamp`：是否在文件名后追加时间戳
- `mode`：文件写入模式，例如 `"a"` 或 `"w"`
- `fmt`：日志格式
- `std`：是否同时输出到标准输出
- `level`：Python 日志级别
- `json_format`：是否输出 JSON 结构化日志
- `max_bytes` / `backup_count`：滚动日志文件参数

## JSON 日志示例

```python
from nerif.utils import set_up_logging

set_up_logging(std=True, json_format=True)
```

## 在代码中记录日志

```python
import logging

LOGGER = logging.getLogger("Nerif")
LOGGER.debug("formatted value: %s", 42)
```
