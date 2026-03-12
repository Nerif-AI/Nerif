---
sidebar_position: 2
---

# 环境变量

本节将介绍如何设置 Nerif 环境变量以支持不同的 LLM 服务。

为了保持 LLM API 和代理服务器的简洁性，我们提供了几种方式来配置应用。

## 默认模型

```bash
# default llm model is gpt-4o
export NERIF_DEFAULT_LLM_MODEL="gpt-4o"
# default embedding model is text-embedding-3-small
export NERIF_DEFAULT_EMBEDDING_MODEL="text-embedding-3-small"
```

## OpenAI

```bash
export OPENAI_API_KEY="..."
```

模型名称：

- `gpt-3.5-turbo`,
- `gpt-4o`,
- `gpt-4o-mini`,
- `gpt-4o-2024-05-13`,
- `gpt-4o-2024-05-13-preview`,
- `gpt-4o-2024-08-06`,
- `gpt-4o-2024-08-06-preview`,
- `gpt-4o-2024-09-13`,
- `gpt-4o-2024-09-13-preview`,
- `gpt-4-turbo`,
- `gpt-4-turbo-preview`,
- `gpt-4`,
- `gpt-4-preview`,
- `gpt-4-turbo-2024-04-09`,
- `gpt-4-turbo-2024-04-09-preview`,
- `gpt-o1`,
- `gpt-o1-preview`,
- `gpt-o1-mini`,
- `text-embedding-3-small`,
- `text-embedding-3-large`,
- `text-embedding-ada-002`,


## 第三方 OpenAI 兼容服务

```bash
export OPENAI_API_KEY="..."
export OPENAI_API_BASE="https://api.xxxx.com/v1/"
```

模型名称：与 OpenAI 相同

## Openrouter

```bash
export OPENROUTER_API_KEY="..."
```

模型名称：`openrouter/xxx`
示例：`openrouter/openai/gpt-4o-2024-08-06`

## Ollama

```bash
# Default url: http://localhost:11434/v1/
export OLLAMA_URL_BASE="http://localhost:11434/v1/"
```

模型名称：`ollama/xxx`
示例：`ollama/llama3.1`


## VLLM

```bash
# Default url: http://localhost:8000/v1/
export VLLM_URL_BASE="http://localhost:8000/v1/"
export VLLM_API_KEY="..."
```

模型名称：`vllm/xxx`
示例：`vllm/llama3.1`

## SLLM

```bash
# Default url: http://localhost:8343/v1/
export SLLM_URL_BASE="http://localhost:8343/v1/"
export SLLM_API_KEY="..."
```

模型名称：`sllm/xxx`
示例：`sllm/llama3.1`
