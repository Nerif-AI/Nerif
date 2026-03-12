---
sidebar_position: 3
---

# Nerif Token 计数器

统计特定模型或请求方法（如 `nerif()`）消耗的 token 数量。

## 基本用法

计数器需要单独创建，然后传入模型类的构造函数或特定方法中。

```python
from nerif.model import NerifTokenCounter, SimpleChatModel
from nerif.core import nerif

counter = NerifTokenCounter()

if nerif("the sky is blue", counter=counter):
    print("True")

model = SimpleChatModel(counter=counter)

print(counter.model_token)
```

## 类

### `NerifTokenCounter`

用于统计特定模型或方法消耗的 token 数量的类。

属性：

- `response_parser (ResponseParserBase)`：LLM 后端响应的解析器，默认值：`OpenAIResponseParser()`

方法：

- `set_parser(parser=ResponseParserBase)`：设置特定的响应解析器。
- `set_parser_based_on_model(self, model_name=str)`：根据模型名称设置响应解析器。
- `count_from_response(response=any)`：从响应中统计模型消耗的 token 数量。

示例：

```python
from nerif.model import NerifTokenCounter, SimpleChatModel
from nerif.core import nerif

counter = NerifTokenCounter()

if nerif("the sky is blue", counter=counter):
    print("True")

model = SimpleChatModel(counter=counter)

print(counter.model_token)

```

### `ResponseParserBase`

响应解析器的基类。

方法：

- `__call__(response=any) -> ModelCost`：从响应中解析 token 使用量。

派生类：

- `OpenAIResponseParser`：用于 OpenAI 兼容 API 的解析器。
- `OllamaResponseParser`：用于 Ollama API 的解析器。

### `NerifTokenConsume`

:::warning

请勿直接使用此类，请使用 `NerifTokenCounter`

:::

属性：

- `model_cost (dict{str: ModelCost})`：存储模型名称和 `ModelCost` 的字典。

方法：

- `__getitem__(key=str) -> ModelCost`：从内部字典获取 `ModelCost`。
- `append(consume=ModelCost)`：追加费用信息。
- `__repr__() -> str`：格式化输出费用信息。


### `ModelCost`

:::warning

请勿直接使用此类，请使用 `NerifTokenCounter`

:::

用于存储特定模型消耗的 token 的类。

属性：

- `model_name (str)`：模型名称。
- `request (int)`：请求中的 token 数量。
- `response (int)`：响应中的 token 数量。

方法：

- `add_cost(request=int, response=None|int)`：追加 token 使用量。
