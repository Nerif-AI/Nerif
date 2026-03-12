# 示例 2：Nerif 核心功能

:::tip[简要说明]

`nerif` 可以用作判断器，始终返回 `True` 或 `False`。

`nerif_match` 可以帮助你从一组候选值中找到最佳匹配。

`nerification` 是一个基础匹配器，通过将给定值的嵌入向量与候选值进行匹配来工作。它是 `nerif` 和 `nerif_match` 的核心。

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

`nerif` 可以用作装饰器来判断函数的返回结果。

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

`nerif_match` 可以用于匹配函数的返回结果。

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

`nerification` 可以将函数的返回结果拟合到一组候选值中。但在大多数情况下不建议使用此方法。匹配基于结果与候选值的嵌入向量相似度。

如果你需要更精确的匹配，可以使用 `nerif_format` 和其他验证器。
