---
sidebar_position: 2
---
# Nerif Format

`NerifFormat` is a useful tool to transform output of LLMs to data types Python can use directly.


## Basic Usage

`NerifFormat` requires **verifier** to check and extract required value from output string from LLMs. Nerif has implemented following verifiers:

- `FormatVerifierInt`: Extract `int` value from output.
- `FormatVerifierFloat`: Extract `float` value from output.
- `FormatVerifierListInt`: Extract `list[int]` value from output.
- `FormatVerifierHumanReadableList`: Convert a human readable list (numbered items) to a Python `list[str]`.
- `FormatVerifierStringList`: Extract `list[str]` from various formats including JSON arrays, markdown bullet lists, and numbered lists.
- `FormatVerifierJson`: Extract JSON objects or arrays from LLM output, even when surrounded by extra text.

To extract value from string output of LLMs, `NerifFormat` provides a function called `try_convert`, which accepts the output and a verifier.

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

### String List Extraction

`FormatVerifierStringList` handles multiple formats that LLMs commonly produce:

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

### JSON Extraction

`FormatVerifierJson` extracts JSON objects or arrays from LLM responses, handling cases where the JSON is embedded in surrounding text:

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

### Robust JSON Parsing with `json_parse`

For even more robust JSON extraction, use `NerifFormat.json_parse()`. This static method handles common LLM output patterns including markdown code blocks:

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

### Pydantic Model Extraction

`FormatVerifierPydantic` validates and parses LLM output into Pydantic model instances. Pydantic is included as a core dependency since v1.3.

```python title="Example: Pydantic model extraction"
from pydantic import BaseModel
from nerif.utils import NerifFormat, FormatVerifierPydantic

class Person(BaseModel):
    name: str
    age: int

# Direct usage
verifier = FormatVerifierPydantic(Person)
person = verifier('{"name": "Alice", "age": 30}')
print(person.name)  # Alice

# Static method on NerifFormat
person = NerifFormat.pydantic_parse('{"name": "Bob", "age": 25}', Person)

# Handles markdown code blocks and embedded JSON
messy = 'Here is the data: ```json\n{"name": "Charlie", "age": 35}\n```'
person = NerifFormat.pydantic_parse(messy, Person)
```

## Implement Your Own Verifier

All verifiers are derived from `FormatVerifierBase`. A valid verifier should implement three methods: `verify`, `match`, and `convert`, and set the `cls` to the target type. These methods will be called with the following logic:

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

For example, to implement a verifier for `int`, we should set `cls` to `int`, and implement the methods:

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

As we can see in the previous code, `verify` will try to check if the input can be directly converted to the target type, while `convert` will directly convert the input. If `verify` returns `False`, `match` will attempt to find the target value from the input. If it cannot find the value, it will return `None` and raise an exception.

For more complex scenarios, like verifying and converting a list, we can let `verify` just return `False`, so the only method you need to implement is `match`.

## Classes

### `NerifFormat`

Methods:

- `try_convert(val=str, verifier_cls=FormatVerifierBase)`: Try convert string `val` by using the verifier. Raises an exception if conversion fails.
- `json_parse(val=str)` *(static)*: Robust JSON extraction from LLM responses. Handles markdown code blocks, plain JSON strings, and embedded JSON within surrounding text.
- `pydantic_parse(val=str, pydantic_model)` *(static)*: Extract and validate a Pydantic model from LLM response. Handles markdown code blocks and embedded JSON.

### `FormatVerifierBase`

Check and extract required value from output string from LLMs.

Methods:

- `verify(val=str)`: Verify if `val` is exactly the expected value.
- `convert(val=str)`: Convert `val` to the expected type directly.
- `match(val=str)`: Find and extract the expected value from `val`.

Derived Classes:

- `FormatVerifierInt`: Extract `int` value from string.
- `FormatVerifierFloat`: Extract `float` value from string.
- `FormatVerifierListInt`: Extract `list[int]` value from string.
- `FormatVerifierHumanReadableList`: Convert a human readable numbered list to a Python `list[str]`.
- `FormatVerifierStringList`: Extract `list[str]` from JSON arrays, markdown bullet lists, or numbered lists.
- `FormatVerifierJson`: Extract JSON objects or arrays from string.
- `FormatVerifierPydantic`: Extract and validate Pydantic model instances from string. Requires `pydantic>=2.0`.
