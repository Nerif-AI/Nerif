---
sidebar_position: 6
---

# 错误处理

Nerif 提供结构化的异常层次体系，以实现一致的错误处理。

## 异常层次结构

```
NerifError (基类)
├── ProviderError      - API 错误（速率限制、认证失败）
├── FormatError        - 输出解析失败（同时也是 ValueError）
├── ConversationMemoryError - 记忆保存/加载错误
├── ConfigError        - 缺少 API 密钥、配置无效
│   └── ModelNotFoundError - 未知模型名称
└── TokenLimitError    - 超出 token 限制
```

## 使用方法

```python
from nerif.exceptions import NerifError, ProviderError, ConfigError

try:
    model.chat("Hello")
except ProviderError as e:
    print(f"Provider {e.provider} error: {e.status_code}")
except NerifError as e:
    print(f"Nerif error: {e}")
```

`FormatError` 同时也是 `ValueError` 的子类，以保持向后兼容性。
