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

Nerif reads environment variables on import. You can either export them in your shell or create a local `.env`.

Option A, create a local `.env` file in the repository root and fill in one of these provider setups:

- OpenAI-compatible setup:
  `OPENAI_API_KEY`
  Optional: `OPENAI_API_BASE`, `OPENAI_PROXY_URL`
- OpenRouter setup:
  `OPENROUTER_API_KEY`
  Optional: `OR_SITE_URL`, `OR_APP_NAME`
- Ollama setup:
  No cloud key is required, but you need a running Ollama server and a local model.

Option B, export directly in your shell:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

or

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
```

Persisting the variables is shell-specific. For Windows, `setx` works if you want them available in future terminals.

## 4. Verify the environment

Offline verification:

```bash
python -m pytest test/nerif_format_test.py -q
```

Minimal OpenAI-compatible smoke test:

```bash
python -c "from nerif.agent import SimpleChatAgent; agent = SimpleChatAgent(model='gpt-4o-mini'); print(agent.chat('Reply with OK only.', max_tokens=3))"
```

Minimal OpenRouter smoke test:

```bash
python -c "from nerif.agent import SimpleChatAgent; agent = SimpleChatAgent(model='openrouter/openai/gpt-4o-mini'); print(agent.chat('Reply with OK only.', max_tokens=3))"
```

Minimal Ollama smoke test:

```bash
ollama pull llama3.1
python -c "from nerif.agent import SimpleChatAgent; agent = SimpleChatAgent(model='ollama/llama3.1'); print(agent.chat('Reply with OK only.', max_tokens=3))"
```

## 5. Common failure modes

- `ModuleNotFoundError`: dependencies were not installed into the active interpreter.
- `No module named pytest`: install with `python -m pip install -e .[dev]`.
- OpenAI `429 insufficient_quota`: the key is present, but the account does not have usable quota.
- Ollama connection errors: make sure the Ollama server is running locally and the requested model has been pulled.
