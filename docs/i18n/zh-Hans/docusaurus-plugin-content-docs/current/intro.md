---
sidebar_position: 1
---

# 快速入门

:::tip[太长不看]

使用 Nerif，你可以同时用"**人类语言**"和"**Python**"进行编程。

示例：
```python
from nerif.core import nerif, nerif_match_string
from nerif.model import SimpleChatModel

model = SimpleChatModel()

# Use nerif to judge "natural language statement"
if nerif("the sky is blue"):
    print("True")
else:
    print("No", end=", ")
    print(model.chat("what is the color of the sky?"))

# Use nerif_match to select from options
choices = ["sunny", "rainy", "cloudy"]
idx = nerif_match_string(selections=choices, text="The weather is warm and bright")
print(f"Best match: {choices[idx]}")
```

你可以通过[示例](./category/start-with-examples)快速上手。
:::

## 简介

使用大语言模型（LLM）往往会产生不可预测的结果。它们可能生成过于礼貌的回复，虽然技术上正确，但并不总是符合上下文需求。面对不确定性时，LLM 可能会表现出类似人类的困惑，提供详细的思考过程而非清晰的答案——这种方式对最终用户并不总是有用的。

为了应对这些挑战，我们推出了 **Nerif**。这个工具将 LLM 与 Python 编程无缝集成，在自然语言和结构化代码之间建立可靠的桥梁。

### 核心特性

- **自然语言判断** — `nerif("statement")` 返回 `True` 或 `False`
- **选项匹配** — `nerif_match_string(selections, text)` 返回最佳匹配的索引
- **格式验证** — 从 LLM 输出中提取 `int`、`float`、`list`、JSON 等类型
- **多模态输入** — 使用 `MultiModalMessage` 将图像、音频和视频与文本一同发送
- **工具调用** — 使用 `ToolDefinition` 定义工具并让模型调用它们
- **结构化输出** — 通过 `response_format={"type": "json_object"}` 获取 JSON 格式的响应
- **Agent 框架** — `NerifAgent` 自动运行 ReAct 风格的工具调用循环
- **Token 追踪** — 使用 `NerifTokenCounter` 监控 token 使用量
- **批量处理** — 使用兼容 OpenAI 的 Batch API 处理大量请求

## 安装

```bash
pip install nerif
```

## 项目结构

Nerif 使用标准的 Python `src/` 布局：

```
Nerif/
├── src/nerif/          # Python source package
│   ├── core/           # nerif(), nerif_match(), Nerification
│   ├── model/          # SimpleChatModel, LogitsChatModel, VisionModel, VideoModel
│   ├── agent/          # NerifAgent, Tool - ReAct-style tool calling agent
│   ├── batch/          # Batch API processing
│   ├── utils/          # Format verification, token counting, logging
│   └── cli/            # Command-line utilities
├── docs/               # Docusaurus documentation site
├── test/               # Test suite
├── examples/           # Usage examples
└── pyproject.toml      # Build configuration
```
