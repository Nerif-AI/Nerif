# Development Setup

This project is a Python package with optional live checks against an OpenAI-compatible provider or a local Ollama server.

## 1. Create and activate a virtual environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install the project

For normal local development:

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

If you only need the runtime package:

```bash
python -m pip install -e .
```

## 3. Configure provider credentials

Nerif reads provider settings from environment variables. Export them in your shell before running Python.

Typical setups:

- OpenAI-compatible:
  - `OPENAI_API_KEY`
  - optional: `OPENAI_API_BASE`
- OpenRouter:
  - `OPENAI_API_KEY`
  - `OPENAI_API_BASE=https://openrouter.ai/api/v1`
- Ollama:
  - no cloud key required, but you need a running Ollama server and a pulled model

Example:

```bash
export OPENAI_API_KEY="sk-..."
export NERIF_DEFAULT_LLM_MODEL=gpt-4o-mini
```

## 4. Verify the environment

Offline verification:

```bash
python -m pytest test/nerif_format_test.py -q
```

Minimal OpenAI-compatible smoke test:

```bash
python -c "from nerif.model import SimpleChatModel; model = SimpleChatModel(model='gpt-4o-mini'); print(model.chat('Reply with OK only.', max_tokens=3))"
```

Minimal OpenRouter smoke test:

```bash
OPENAI_API_BASE=https://openrouter.ai/api/v1 python -c "from nerif.model import SimpleChatModel; model = SimpleChatModel(model='openrouter/openai/gpt-4o-mini'); print(model.chat('Reply with OK only.', max_tokens=3))"
```

Minimal Ollama smoke test:

```bash
ollama pull llama3.1
python -c "from nerif.model import SimpleChatModel; model = SimpleChatModel(model='ollama/llama3.1'); print(model.chat('Reply with OK only.', max_tokens=3))"
```

## 5. Common failure modes

- `ModuleNotFoundError`: dependencies were not installed into the active interpreter.
- `No module named pytest`: install with `python -m pip install -e .[dev]`.
- OpenAI `429 insufficient_quota`: the key is present, but the account does not have usable quota.
- Ollama connection errors: make sure the Ollama server is running locally and the requested model has been pulled.
