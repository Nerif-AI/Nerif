# Nerif - LLM Powered Python

Nerif integrates Large Language Models with Python programming, providing natural language judgment (`nerif()`), option matching (`nerif_match()`), format verification, multi-modal support, tool calling, and agent capabilities.

## Project Structure

```
Nerif/
├── src/nerif/          # Python source package (src layout)
│   ├── core/           # nerif(), nerif_match(), Nerification classes
│   ├── model/          # SimpleChatModel, LogitsChatModel, VisionModel, AudioModel
│   ├── agent/          # NerifAgent, Tool - ReAct-style tool calling agent
│   ├── batch/          # Batch API processing (OpenAI-compatible)
│   ├── utils/          # NerifFormat, NerifTokenCounter, logging
│   └── cli/            # CLI utilities (image compression)
├── docs/               # Docusaurus 3.5.2 documentation site
│   ├── docs/           # Markdown documentation content
│   ├── blog/           # Blog posts
│   ├── src/            # React components
│   └── static/         # Static assets
├── test/               # Unit tests (pytest)
├── examples/           # Usage examples (01-14)
└── pyproject.toml      # Hatch build system
```

## Development Commands

```bash
pip install -e ".[dev]"          # Editable install with dev deps
pytest test/                     # Run all tests
pytest test/nerif_format_test.py # Run specific test file
ruff check src/ test/            # Lint
ruff format src/ test/           # Format
```

## Documentation

```bash
cd docs && npm install && npm run build   # Build docs
cd docs && npm start                      # Dev server
```

## Key Modules

- **core**: `nerif(text)` returns bool judgment; `nerif_match_string(selections, text)` returns best match index. Uses three-tier approach: logits mode → embedding mode fallback.
- **model**: `SimpleChatModel` for chat, `LogitsChatModel` for logprobs, `MultiModalMessage` for vision, `ToolDefinition` for tool calling.
- **agent**: `NerifAgent` with `Tool` registration for multi-step ReAct loops.
- **utils**: `NerifFormat` for parsing LLM outputs (int, float, list, JSON, JSON schema). `NerifTokenCounter` for tracking token usage.

## Version Management

Uses `bumpver` for version bumps. Version is tracked in `pyproject.toml`.

```bash
bumpver update --patch   # 0.11.0 -> 0.11.1
bumpver update --minor   # 0.11.0 -> 0.12.0
```

## CI/CD

- `.github/workflows/lint.yml` - Ruff linting on PRs
- `.github/workflows/release.yml` - PyPI publish on version tag push

## Environment Variables

- `NERIF_DEFAULT_LLM_MODEL` - Default LLM (default: `gpt-4o`)
- `NERIF_DEFAULT_EMBEDDING_MODEL` - Default embedding model (default: `text-embedding-3-small`)
- `OPENAI_API_KEY` - OpenAI API key
- See `docs/docs/nerif-environment.md` for Ollama, VLLM, OpenRouter setup
