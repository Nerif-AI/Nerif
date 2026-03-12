# Example 4: Other Utilities

## Token Counter

```python
from nerif.core import nerif
from nerif.utils import NerifTokenCounter

counter = NerifTokenCounter()

if nerif("the sky is blue", counter=counter):
    print("True")

print(counter.model_token)
```

## Logger

```python
import nerif.model as model
import nerif.utils.log as log

log.set_up_logging(out_file="sample.log", mode="w", std=True)

model = model.SimpleChatModel()

print(model.chat("What is the capital of the moon?"))
print(model.chat("What is the capital of the moon?", max_tokens=10))
```

## Max Tokens

```python
import nerif

# case for ever specify

agent1 = nerif.model.SimpleChatModel()

print(agent1.chat("What is the capital of the moon?"))
agent1.counter

# case for specified in agent

agent2 = nerif.model.SimpleChatModel(max_tokens=5)

print(agent2.chat("What is the capital of the moon?"))
print(agent2.chat("What is the capital of the moon?", max_tokens=20))
```

