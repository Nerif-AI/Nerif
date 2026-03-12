# Example 2: Nerif Cores

:::tip[TL;DR]

`nerif` can be used as a judger, always returning `True` or `False`.

`nerif_match` can help you find the best match from a list of possible values.

`nerification` is a basic matcher that matches the embedding of the given value with the possible values. It is the core for both `nerif` and `nerif_match`.

:::

## nerif

```python
from nerif.core import Nerif, nerif
from nerif.model import SimpleChatModel

agent = SimpleChatModel()

if nerif("the sky is blue"):
    print("True")
else:
    print("No", end=", ")
    print(agent.chat("what is the color of the sky?"))


judger = Nerif(model="gpt-4o-mini")
print(judger.judge("the sky is blue"))
```

`nerif` can be used as a decorator to judge the result of a function.

## nerif_match

```python
from nerif.core import nerif_match

selections = ["iPhone 5", "iPhone 6", "iPhone 12"]

best_choice = nerif_match(selections=selections, text="Which iPhone is the most powerful one?")

print(best_choice)

match best_choice:
    case 0:
        print("iPhone 5")
    case 1:
        print("iPhone 6")
    case 2:
        print("iPhone 12")
    case _:
        print("No match")
```

`nerif_match` can be used to match the result of a function.

## nerification

```python
from nerif.core import Nerification, NerificationInt, NerificationString

nerification = Nerification(model="text-embedding-3-large")

print(nerification.simple_fit("yes, it is"))
# result: None
print(nerification.force_fit("yes, it is"))
# result: True
print(nerification.simple_fit("true"))
# result: True
print(nerification.force_fit("true"))
# result: True

nerification_int = NerificationInt(model="text-embedding-3-large", possible_values=[1, 233, 343])


print(nerification_int.simple_fit(1))
# result: 1
print(nerification_int.force_fit(1))
# result: 1
print(nerification_int.simple_fit(233))
# result: 233
print(nerification_int.force_fit("The value is 233"))
# result: 233
print(nerification_int.simple_fit(343))
# result: 343
print(nerification_int.force_fit("The value is 343"))
# result: 343

nerification_string = NerificationString(model="text-embedding-3-large", possible_values=["YES", "NO"])

print(nerification_string.simple_fit("yes"))
# result: YES
print(nerification_string.force_fit("Well, I guess you are right"))
# result: YES
print(nerification_string.simple_fit("no"))
# result: NO
print(nerification_string.force_fit("Oh, I don't think so"))
# result: NO
```

`nerification` can be used to fit the result of a function to a list of possible values. However, this is not recommended in most cases. The matching is based on the similarity of the embedding of the result and the possible values.

If you need more precise matching, you can use `nerif_format` and other verifiers instead.
