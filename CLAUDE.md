# Nerif - LLM Powered Python

Nerif integrates Large Language Models with Python programming, providing natural language judgment (`nerif()`), option matching (`nerif_match()`), format verification, multi-modal support, tool calling, agent capabilities, streaming, async support, retry, and Pydantic structured output.

## Project Structure

```
Nerif/
├── src/nerif/          # Python source package (src layout)
│   ├── core/           # nerif(), nerif_match(), Nerification classes
│   ├── model/          # SimpleChatModel, LogitsChatModel, VisionModel, AudioModel
│   ├── agent/          # NerifAgent, Tool - ReAct-style tool calling agent
│   ├── batch/          # Batch API processing (OpenAI-compatible)
│   ├── memory/         # ConversationMemory - sliding window, summarization, persistence
│   ├── rag/            # SimpleRAG, NumpyVectorStore, VectorStoreBase
│   ├── utils/          # NerifFormat, NerifTokenCounter, RetryConfig, logging
│   ├── exceptions.py   # Structured exception hierarchy (NerifError, ProviderError, etc.)
│   ├── asr/            # Audio speech recognition (optional: nerif[asr])
│   ├── tts/            # Text-to-speech (optional: nerif[tts])
│   ├── img_gen/        # Image generation (optional: nerif[img-gen])
│   └── cli/            # CLI utilities
│       ├── main.py     # CLI entry point
│       ├── check.py    # Environment/config checker
│       └── test_model.py  # Model connectivity tester
├── docs/               # Docusaurus 3.5.2 documentation site
│   ├── docs/           # Markdown documentation content (English)
│   ├── i18n/zh-Hans/   # Chinese (Simplified) translations
│   ├── blog/           # Blog posts
│   ├── src/            # React components
│   └── static/         # Static assets
├── test/               # Unit tests (pytest)
├── examples/           # Usage examples (01-22)
└── pyproject.toml      # Hatch build system
```

## Development Commands

```bash
pip install -e ".[dev]"          # Editable install with dev deps (pydantic included)
pytest test/                     # Run all tests
pytest test/nerif_format_test.py # Run specific test file
ruff check src/ test/ examples/  # Lint (must pass before committing)
ruff format src/ test/ examples/ # Format

# CLI commands (after install)
nerif check                      # Check environment and API key config
nerif test-model                 # Test model connectivity
```

## Documentation

```bash
cd docs && npm install && npm run build   # Build docs
cd docs && npm start                      # Dev server
```

Documentation is bilingual (English + Chinese). When updating docs:
1. Update English docs in `docs/docs/`
2. Mirror changes to `docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/`

## Key Modules

- **core**: `nerif(text)` returns bool judgment; `nerif_match_string(selections, text)` returns best match index. Uses three-tier approach: logits mode → embedding mode (or text fallback) → force_fit. Embedding is optional since v1.1.
- **model**: `SimpleChatModel` for chat (sync/async/streaming), `LogitsChatModel` for logprobs, `MultiModalMessage` for vision, `ToolDefinition` for tool calling. Supports `retry_config` and `response_model` (Pydantic).
- **agent**: `NerifAgent` with `Tool` registration for multi-step ReAct loops.
- **memory**: `ConversationMemory` for sliding-window conversation history with optional LLM-based summarization and JSON persistence.
- **rag**: `SimpleRAG` with `NumpyVectorStore` for lightweight retrieval-augmented generation. Extend `VectorStoreBase` for custom backends (ChromaDB, FAISS, etc.).
- **utils**: `NerifFormat` for parsing LLM outputs (int, float, list, JSON, JSON schema, Pydantic). `NerifTokenCounter` for tracking token usage. `RetryConfig` for retry with exponential backoff.
- **exceptions**: Structured hierarchy — `NerifError`, `ProviderError`, `FormatError` (also `ValueError`), `ConversationMemoryError`, `ConfigError`, `ModelNotFoundError`, `TokenLimitError`.

## Version Management

Uses `bumpver` for patch/minor bumps. For major version jumps, edit `pyproject.toml` manually (both `version` and `current_version` fields).

```bash
bumpver update --patch   # 1.1.0 -> 1.1.1
bumpver update --minor   # 1.1.0 -> 1.2.0
```

## Release Process

1. Bump version in `pyproject.toml` (both `version` and `current_version`)
2. Commit with message: `release: Bump version X.Y.Z -> A.B.C`
3. Push to `main` branch (or create PR to `Nerif-AI/nerif` upstream)
4. GitHub Actions (`.github/workflows/release.yml`) automatically:
   - Detects version change in `pyproject.toml`
   - Builds the package
   - Creates GitHub Release with `v{version}` tag
   - Publishes to PyPI via trusted publisher

No manual `twine upload` or `gh release create` needed — the CI handles everything.

## Pre-commit Checklist

Before committing code, always verify:

```bash
ruff check src/ test/ examples/  # Lint must pass (CI runs this)
pytest test/                     # Tests must pass
```

The CI lint job (`lint.yml`) checks `src/`, `test/`, and `examples/` — all three directories. Forgetting `examples/` will cause CI failures.

## CI/CD

- `.github/workflows/lint.yml` - Ruff linting on PRs (`ruff check src/ test/ examples/`)
- `.github/workflows/release.yml` - Auto PyPI publish + GitHub release on version bump to `main`

## Environment Variables

- `NERIF_DEFAULT_LLM_MODEL` - Default LLM (default: `gpt-4o`)
- `NERIF_DEFAULT_EMBEDDING_MODEL` - Default embedding model (default: `text-embedding-3-small`; set to `""` to disable)
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key (for Claude models)
- `GOOGLE_API_KEY` - Google API key (for Gemini models)
- See `docs/docs/nerif-environment.md` for Ollama, VLLM, OpenRouter setup
