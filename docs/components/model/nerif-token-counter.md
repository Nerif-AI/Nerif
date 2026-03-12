---
sidebar_position: 3
---

# Nerif Token Counter

Counting token consumed by specific model or request method like `nerif()`

## Basic Usage

A counter should be create seperately, and pass it into constructor of model class or special methods.

```python
from nerif.model import NerifTokenCounter
from nerif.core import nerif

counter = NerifTokenCounter()

if nerif("the sky is blue", counter=counter):
    print("True")

model = nerif.model.SimpleChatModel(counter=counter)

print(counter.model_token)
```

## Classes

### `NerifTokenCounter`

A class to count the token consumed by specific Model or method.

Attributes:

- `response_parser (ResponseParserBase)`: Parser for response from llm backend, default: `OpenAIResponseParser()`

Methods:

- `set_parser(parser=ResponseParserBase)`: Set response parser to specific parser.
- `set_parser_based_on_model(self, model_name=str)`: Set response parser from name of model.
- `count_from_response(response=any)`: Counting tokens consumed by the model from response.

Example:

```python
from nerif.model import NerifTokenCounter
from nerif.core import nerif

counter = NerifTokenCounter()

if nerif("the sky is blue", counter=counter):
    print("True")

model = nerif.model.SimpleChatModel(counter=counter)

print(counter.model_token)

```

### `ResponseParserBase`

Base class of response parser.

Methods:

- `__call__(response=any) -> ModelCost`: Parse token usage from response.

Derived Classes:

- `OpenAIResponseParser`: Parser for OpenAI compatible API.
- `OllamaResponseParser`: Parser for Ollama API.

### `NerifTokenConsume`

:::warning

Do not use this class directly, plase use `NerifTokenCounter`

:::

Attributes: 

- `model_cost (dict{str: ModelCost})`: A dict to store model name and `ModelCost`.

Methods:

- `__getitem__(key=str) -> ModelCost`: Get `ModelCost` from internal dict.
- `append(consume=ModelCost)`: Append cost information.
- `__repr__() -> str`: Prettyprint cost information.


### `ModelCost`

:::warning

Do not use this class directly, plase use `NerifTokenCounter`

:::

A class to store token consumed by specific model.

Attributes:

- `model_name (str)`: The name of model.
- `request (int)`: The count of token in request.
- `response (int)`: The count of token in response.

Methods:

- `add_cost(request=int, response=None|int)`: Append token usage.


