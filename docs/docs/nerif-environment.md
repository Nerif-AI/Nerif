---
sidebar_position: 2
---

# Environment Variable

This Part we will discuss about how to set Nerif environment variables to support different LLM services.

To keep the LLM API and Proxy Server simple, basically we provided several method to serve the application.

## Default Models

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

Model name:

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


## 3rd-party OpenAI Service

```bash
export OPENAI_API_KEY="..."
export OPENAI_API_BASE="https://api.xxxx.com/v1/"
```

Model name: Same with OpenAI

## Openrouter

```bash
export OPENROUTER_API_KEY="..."
```

Model name: `openrouter/xxx`
Example: `openrouter/openai/gpt-4o-2024-08-06`

## Ollama

```bash
# Default url: http://localhost:11434/v1/
export OLLAMA_URL_BASE="http://localhost:11434/v1/"
```

Model name: `ollama/xxx`
Example: `ollama/llama3.1`


## VLLM

```bash
# Default url: http://localhost:8000/v1/
export VLLM_URL_BASE="http://localhost:8000/v1/"
export VLLM_API_KEY="..."
```

Model name: `vllm/xxx`
Example: `vllm/llama3.1`

## SLLM

```bash
# Default url: http://localhost:8343/v1/
export SLLM_URL_BASE="http://localhost:8343/v1/"
export SLLM_API_KEY="..."
```

Model name: `sllm/xxx`
Example: `sllm/llama3.1`