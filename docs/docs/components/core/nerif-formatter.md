---
sidebar_position: 2
---
# Nerif Format

`NerifFormat` is a useful tool to transform output of LLMs to data types Python can use directly.


## Basic Usage

`NerifFormat` requires **verifier** to check and extract required value from output string from LLMs. Nerif has implemented following verifier:

- `FormatVerifierInt`: Extract `int` value from output.
- `FormatVerifierFloat`: Extract `float` value from output.
- `FormatVerifierListInt`: Extract `list[int]` value from output.
- `FormatVerifierHumanReadableList`: Convert a human readable list to a Python `list`

To extract value from string output of LLMs, `NerifFormat` provides a function called `try_convert`, which accept the output and a verifier.

An example code shown as follow

```python title=Example code of NerifToken
formatter = NerifFormat()
llm_output = "The result is 19"

value = formatter.try_convert(llm_output, FormatVerifierInt)
assert value == 19
assert isinstance(value, int)

llm_output = "The result is 19.1"
value_float = formatter.try_convert(llm_output, FormatVerifierFloat)
assert value_float = 19.1
assert isinstance(value_float, float)

llm_output = "The result is [1, 2, 3, 4]"
value_list = formatter.try_convert(llm_output, FormatVerifierListInt)
assert len(value_list) == 4
for i in range(4):
  assert value_list[i] == i + 1
assert isinstance(value_list, list)

llm_output = """Here are some fluits:
1. Apple
2. Banana
3. Orange
"""
value_float = formatter.try_convert(llm_output, FormatVerifierHumanReadableList)
assert len(value_list_hr) == 3
assert isinstance(value_list_hr[2]) == "Orange"
assert isinstance(value_float, list[str])
```

## Implement Your Own Verifier

All verifiers are derived from `FormatVerifierBase`. A valid verifier should implement three methods: `verify`, `match`, and `convert`, and set the `cls` to the target type. These methods will be called with the following logic:

```python
if verify(val):
  return convert(val):
else:
  res = self.match(val)
  if res is not None:
    return res
  else:
    raise ValueError("Cannot convert to {}".format(self.cls.__name__))
```

For example, to implement a verifier for `int`, we should set `cls` to `int`, and implement there methods

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

As we can see in the previous code, `verify` will try to convert the input into the target type directly, while `convert` will directly convert the input into the target type. If `verify` returns `False`, `match` will attempt to find the target value from the input. If it cannot find the value, it will return `None` and raise an exception.

For more complex scenario, like verify and convert a list, we can let verify just return `False`, so the only method you should implement is the `match`.

## Classes

### `NerifFormat`

Methods:

`try_convert(val=str, verifier_cls=FormatVerifierBase)`: Try convert string `val` by using the verifier, if failed, raise an exception

### `FormatVerifierBase`

Check and extract required value from output string from LLMs. 

Methods:

- `verify(val=str)`: Verify if the `val` is just the expect value.
- `convert(val=str)`: Convert `val` to expect type directly.
- `match(val=str)`: Find expect value from `val`, and try to extract it.

Derived Classes:

- `FormatVerifierInt`: Extract `int` value from string.
- `FormatVerifierFloat`: Extract `float` value from string.
- `FormatVerifierListInt`: Extract `list[int]` value from string.
- `FormatVerifierHumanReadableList`: Convert a human readable list to a Python `list`
