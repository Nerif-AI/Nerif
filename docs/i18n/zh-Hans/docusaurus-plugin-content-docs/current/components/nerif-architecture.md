---
sidebar_position: 1
---

# Nerif 架构

本节逐一介绍 Nerif 的核心设计。以下是整体架构图：

![Nerif Architecture](img/arch.png)

### Nerif Model

Nerif Model 提供了灵活使用各种 AI 模型的能力，包括多模态功能。我们的框架支持与外部 API 和工具交互的模型，以增强功能。

当前支持的功能：
- **LLM 对话模型** (`SimpleChatModel`) - 支持对话历史的文本生成
- **多模态输入** (`MultiModalMessage`) - 支持图像、音频和视频与文本的组合输入
- **工具调用** (`ToolDefinition`, `ToolCallResult`) - 兼容 OpenAI 的函数调用
- **结构化输出** - 通过 `response_format` 参数启用 JSON 模式
- **视觉模型** (`VisionModel`, `VideoModel`) - 专用的图像和视频分析
- **音频模型** (`AudioModel`) - 语音转文字
- **嵌入模型** (`SimpleEmbeddingModel`, `OllamaEmbeddingModel`) - 文本嵌入
- **流式输出** (`stream_chat()`、`astream_chat()`) - 实时逐 token 输出
- **异步 API** (`achat()`、`aembed()`、`agenerate()`) - 原生 async/await 支持
- **重试机制** (`RetryConfig`) - 指数退避的自动重试

### Nerif Agent

Agent 框架（`NerifAgent`）提供了 ReAct 风格（推理 -> 行动 -> 观察）的多步工具调用循环。将 `Tool` 对象注册到 Agent 后，它将自动执行以下步骤：
1. 推理应该使用哪个工具
2. 执行工具并观察结果
3. 重复上述步骤直到产生最终答案

详见 [Agent 框架](./model/nerif-agent-framework.md)。

### Nerif Core

:::note
从 v1.1 开始，embedding 模型是可选的。未配置 embedding 模型时，Nerif 会回退到基于文本的匹配方式，使 `nerif()` 和 `nerif_match()` 无需 embedding API key 即可工作。
:::

`model` 和 `core` 之间的关键区别在于它们的类型系统实现。虽然 LLM/VLM 模型通常生成需要复杂后处理的自然语言输出，但 Nerif Core 确保输出具有正确的类型并可在应用程序中直接使用。

Nerif Core 使用**三层匹配方法**：

1. **Logits 模式** - 使用 LLM 的 logprobs API 进行快速、直接的 token 概率分析
2. **结构化输出模式** - 回退到 JSON 格式的结构化响应
3. **嵌入模式** - 使用嵌入相似度比较作为最终的兜底方案

核心功能由以下模块组成：

1. **Nerif**：评估语句并返回布尔值（`True`/`False`）
2. **Nerification**：验证语句并返回布尔值（`True`/`False`）
3. **Nerif Match**：接受一个语句和一个列表作为输入，返回最佳匹配项的索引
4. **Nerif Format**：处理不同格式之间的类型转换，包括 JSON 解析
5. **Nerif Log**：提供完善的日志功能

### Nerif Batch

Batch API 模块提供兼容 OpenAI 的批量处理功能，用于异步处理大量请求，同时降低成本。

### Nerif Flow

此功能将在 v1.0 版本发布后提供。
