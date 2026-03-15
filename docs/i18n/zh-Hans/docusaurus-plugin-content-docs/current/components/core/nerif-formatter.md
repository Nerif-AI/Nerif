---
sidebar_position: 2
---
# Nerif Format

`NerifFormat` 是一个实用工具，用于将 LLM 的输出转换为 Python 可以直接使用的数据类型。


## 基本用法

`NerifFormat` 需要**验证器**来检查和提取 LLM 输出字符串中所需的值。Nerif 已实现以下验证器：

- `FormatVerifierInt`：从输出中提取 `int` 值。
- `FormatVerifierFloat`：从输出中提取 `float` 值。
- `FormatVerifierListInt`：从输出中提取 `list[int]` 值。
- `FormatVerifierHumanReadableList`：将人类可读的列表（编号项目）转换为 Python 的 `list[str]`。
- `FormatVerifierStringList`：从多种格式中提取 `list[str]`，包括 JSON 数组、Markdown 无序列表和编号列表。
- `FormatVerifierJson`：从 LLM 输出中提取 JSON 对象或数组，即使它们被额外的文本包围也能处理。

要从 LLM 的字符串输出中提取值，`NerifFormat` 提供了一个名为 `try_convert` 的函数，它接受输出内容和一个验证器。

```python title="Example: Basic type extraction"
from nerif.utils import NerifFormat, FormatVerifierInt, FormatVerifierFloat, FormatVerifierListInt

formatter = NerifFormat()
llm_output = "The result is 19"

value = formatter.try_convert(llm_output, FormatVerifierInt)
assert value == 19
assert isinstance(value, int)

llm_output = "The result is 19.1"
value_float = formatter.try_convert(llm_output, FormatVerifierFloat)
assert value_float == 19.1
assert isinstance(value_float, float)

llm_output = "The result is [1, 2, 3, 4]"
value_list = formatter.try_convert(llm_output, FormatVerifierListInt)
assert len(value_list) == 4
for i in range(4):
    assert value_list[i] == i + 1
assert isinstance(value_list, list)
```

### 字符串列表提取

`FormatVerifierStringList` 可以处理 LLM 常见的多种输出格式：

```python title="Example: String list extraction"
from nerif.utils import NerifFormat, FormatVerifierStringList

formatter = NerifFormat()

# JSON array format
llm_output = 'Here are the items: ["apple", "banana", "cherry"]'
items = formatter.try_convert(llm_output, FormatVerifierStringList)
# items = ["apple", "banana", "cherry"]

# Markdown bullet list format
llm_output = """Here are some fruits:
- Apple
- Banana
- Cherry
"""
items = formatter.try_convert(llm_output, FormatVerifierStringList)
# items = ["Apple", "Banana", "Cherry"]

# Numbered list format
llm_output = """Top languages:
1. Python
2. JavaScript
3. Rust
"""
items = formatter.try_convert(llm_output, FormatVerifierStringList)
# items = ["Python", "JavaScript", "Rust"]
```

### JSON 提取

`FormatVerifierJson` 从 LLM 响应中提取 JSON 对象或数组，能够处理 JSON 嵌入在其他文本中的情况：

```python title="Example: JSON extraction"
from nerif.utils import NerifFormat, FormatVerifierJson

formatter = NerifFormat()

llm_output = 'Here is the data: {"name": "Alice", "age": 30}'
data = formatter.try_convert(llm_output, FormatVerifierJson)
# data = {"name": "Alice", "age": 30}

# Also handles JSON arrays
llm_output = 'The results are: [1, 2, 3]'
data = formatter.try_convert(llm_output, FormatVerifierJson)
# data = [1, 2, 3]
```

### 使用 `json_parse` 进行健壮的 JSON 解析

如需更健壮的 JSON 提取，可以使用 `NerifFormat.json_parse()`。这个静态方法可以处理常见的 LLM 输出模式，包括 Markdown 代码块：

````python title="Example: Robust JSON parsing"
from nerif.utils import NerifFormat

# Handles markdown code blocks
llm_output = """Here is the result:
```json
{"name": "Alice", "age": 30}
```
"""
data = NerifFormat.json_parse(llm_output)
# data = {"name": "Alice", "age": 30}

# Handles plain JSON
data = NerifFormat.json_parse('{"key": "value"}')
# data = {"key": "value"}

# Falls back to FormatVerifierJson for embedded JSON
data = NerifFormat.json_parse('The answer is {"result": 42} as expected.')
# data = {"result": 42}
````

### Pydantic 模型提取

`FormatVerifierPydantic` 将 LLM 输出验证并解析为 Pydantic 模型实例。Pydantic 自 v1.3 起已作为核心依赖内置。

```python title="示例：Pydantic 模型提取"
from pydantic import BaseModel
from nerif.utils import NerifFormat, FormatVerifierPydantic

class Person(BaseModel):
    name: str
    age: int

# 直接使用
verifier = FormatVerifierPydantic(Person)
person = verifier('{"name": "Alice", "age": 30}')
print(person.name)  # Alice

# 使用 NerifFormat 的静态方法
person = NerifFormat.pydantic_parse('{"name": "Bob", "age": 25}', Person)

# 支持 markdown 代码块和嵌入式 JSON
messy = 'Here is the data: ```json\n{"name": "Charlie", "age": 35}\n```'
person = NerifFormat.pydantic_parse(messy, Person)
```

## 实现自定义验证器

所有验证器都继承自 `FormatVerifierBase`。一个有效的验证器需要实现三个方法：`verify`、`match` 和 `convert`，并将 `cls` 设置为目标类型。这些方法将按以下逻辑被调用：

```python
if verify(val):
  return convert(val)
else:
  res = self.match(val)
  if res is not None:
    return res
  else:
    raise ValueError("Cannot convert to {}".format(self.cls.__name__))
```

例如，要实现一个 `int` 类型的验证器，我们需要将 `cls` 设置为 `int`，并实现以下方法：

```python
class FormatVerifierInt(FormatVerifierBase):
    cls = int
    pattern = re.compile(r"\b\d+\b")

    # check if the string is a number
    def verify(self, val: str) -> bool:
        return val.isdigit()

    # extract the number from the string
    def match(self, val: str) -> int:
        candidate = self.pattern.findall(val)
        if len(candidate) > 0:
            return int(candidate[0])
        return None

    # type converter
    def convert(self, val: str) -> int:
        return int(val)
```

如上面的代码所示，`verify` 会尝试检查输入是否可以直接转换为目标类型，而 `convert` 则直接进行转换。如果 `verify` 返回 `False`，`match` 将尝试从输入中查找目标值。如果找不到目标值，它将返回 `None` 并抛出异常。

对于更复杂的场景，比如验证和转换列表，可以让 `verify` 直接返回 `False`，这样你只需要实现 `match` 方法即可。

## 类参考

### `NerifFormat`

方法：

- `try_convert(val=str, verifier_cls=FormatVerifierBase)`：使用验证器尝试转换字符串 `val`。如果转换失败则抛出异常。
- `json_parse(val=str)` *(静态方法)*：从 LLM 响应中健壮地提取 JSON。支持 Markdown 代码块、纯 JSON 字符串，以及嵌入在其他文本中的 JSON。
- `pydantic_parse(val=str, pydantic_model)` *(静态方法)*：从 LLM 响应中提取并验证 Pydantic 模型。支持 markdown 代码块和嵌入式 JSON。

### `FormatVerifierBase`

检查和提取 LLM 输出字符串中所需的值。

方法：

- `verify(val=str)`：验证 `val` 是否恰好是期望的值。
- `convert(val=str)`：将 `val` 直接转换为期望的类型。
- `match(val=str)`：从 `val` 中查找并提取期望的值。

派生类：

- `FormatVerifierInt`：从字符串中提取 `int` 值。
- `FormatVerifierFloat`：从字符串中提取 `float` 值。
- `FormatVerifierListInt`：从字符串中提取 `list[int]` 值。
- `FormatVerifierHumanReadableList`：将人类可读的编号列表转换为 Python 的 `list[str]`。
- `FormatVerifierStringList`：从 JSON 数组、Markdown 无序列表或编号列表中提取 `list[str]`。
- `FormatVerifierJson`：从字符串中提取 JSON 对象或数组。
- `FormatVerifierPydantic`：从字符串中提取并验证 Pydantic 模型实例。需要 `pydantic>=2.0`。
