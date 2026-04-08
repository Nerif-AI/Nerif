---
sidebar_position: 3
---

# Nerif Token 计数器

`NerifTokenCounter` 不仅统计 token，还提供轻量级可观测性指标，例如延迟、估算成本、成功率、重试次数和导出接口。

## 基本用法

```python
from nerif.model import SimpleChatModel
from nerif.utils import NerifTokenCounter

counter = NerifTokenCounter()
model = SimpleChatModel(model="gpt-4o", counter=counter)

response = model.chat("Explain tracing in one sentence.")
print(response)
print(counter.summary())
print(counter.to_dict())
```

## 可追踪内容

- 每个模型的输入/输出 token
- 每个模型的平均延迟
- 成功/失败请求数量
- 重试次数
- 基于内置价格表计算的估算美元成本

## 常用方法

- `count_from_response(response)`
- `record_request(...)`
- `record_retry(model)`
- `avg_latency(model=None)`
- `success_rate(model=None)`
- `total_cost()`
- `summary()`
- `to_dict()`
- `to_json()`
- `reset_stats()`

## 请求回调

`NerifTokenCounter` 还支持请求生命周期回调：

- `on_request_start`
- `on_request_end`
- `on_error`

这些回调与模型/agent 使用的 `CallbackManager` 事件系统是分开的。

## 核心类

### `NerifTokenCounter`
主计数器类。

### `ResponseParserBase`
用于从 provider 响应中提取 token 使用量的基类。

派生类：
- `OpenAIResponseParser`
- `OllamaResponseParser`

### `ModelCost`
存储单个模型的输入/输出 token 总量。

### `NerifTokenConsume`
内部使用的聚合容器。
