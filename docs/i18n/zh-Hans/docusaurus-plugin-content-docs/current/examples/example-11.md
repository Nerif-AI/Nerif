# 示例 11：结构化输出 / JSON 模式

本示例展示如何使用**结构化输出**（JSON 模式）从大语言模型获取格式规范的 JSON 响应，以及如何使用 `NerifFormat.json_parse()` 进行稳健的解析。

## 核心概念

- **`response_format`**：向 `model.chat()` 传递 `{"type": "json_object"}` 以指示模型返回有效的 JSON。
- **`NerifFormat.json_parse()`**：一个稳健的静态方法，用于从大语言模型的响应中提取 JSON，能够处理常见模式，如 Markdown 代码块（`` ```json ... ``` ``）、纯 JSON 以及嵌入在其他文本中的 JSON。

## 代码

```python
from nerif.model import SimpleChatModel
from nerif.utils import NerifFormat

# Use JSON mode
model = SimpleChatModel(model="gpt-4o")
result = model.chat(
    "List three programming languages with their year of creation. "
    "Respond in JSON format as an array of objects with 'name' and 'year' fields.",
    response_format={"type": "json_object"},
)

print("Raw response:", result)

# Parse the JSON robustly
parsed = NerifFormat.json_parse(result)
print("Parsed:", parsed)
```

## 工作原理

1. 在 `chat()` 调用中设置 `response_format={"type": "json_object"}`，这会告诉模型输出有效的 JSON。
2. 原始响应是一个 JSON 字符串，使用 `NerifFormat.json_parse()` 安全地解析它。
3. `json_parse()` 能处理各种边界情况：Markdown 代码块、多余的空白字符以及嵌入在文本中的 JSON。

## 适用场景

- 当你需要从大语言模型的响应中获取结构化数据（列表、对象等）时
- 当构建以编程方式消费大语言模型输出的流水线时
- 与 `FormatVerifierJson` 结合使用，可从非结构化响应中进行更灵活的提取
