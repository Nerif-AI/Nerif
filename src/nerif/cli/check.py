#!/usr/bin/env python3
"""CLI command to check Nerif configuration and provider availability."""

import os
import sys


def main():
    """Check API key configuration and list available providers."""
    providers = [
        ("OpenAI", "OPENAI_API_KEY", "gpt-4o, gpt-4o-mini, gpt-3.5-turbo"),
        ("Anthropic", "ANTHROPIC_API_KEY", "claude-3.5-sonnet, claude-3-haiku"),
        ("Google Gemini", "GOOGLE_API_KEY", "gemini-1.5-pro, gemini-1.5-flash"),
        ("OpenRouter", "OPENROUTER_API_KEY", "openrouter/* models"),
        ("Ollama", "OLLAMA_URL", "ollama/* (local, no key needed)"),
        ("vLLM", "VLLM_URL", "vllm/* (local)"),
        ("SLLM", "SLLM_URL", "sllm/* (local)"),
    ]

    print("Nerif Configuration Check")
    print("=" * 50)

    available = 0
    for name, env_var, models in providers:
        value = os.environ.get(env_var)
        if env_var in ("OLLAMA_URL", "VLLM_URL", "SLLM_URL"):
            # Local providers: check if URL is set (optional)
            if value:
                status = f"configured ({value})"
                available += 1
            else:
                status = "not configured (using default)"
                available += 1  # Local providers work with defaults
        elif value:
            masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
            status = f"set ({masked})"
            available += 1
        else:
            status = "NOT SET"

        icon = "+" if value or env_var in ("OLLAMA_URL", "VLLM_URL", "SLLM_URL") else "-"
        print(f"  [{icon}] {name:<15} {env_var:<25} {status}")
        print(f"      Models: {models}")

    print(f"\n{available}/{len(providers)} providers available")

    # Check default model
    default_llm = os.environ.get("NERIF_DEFAULT_LLM_MODEL", "gpt-4o")
    default_embed = os.environ.get("NERIF_DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")
    print(f"\nDefault LLM model: {default_llm}")
    print(f"Default embedding model: {default_embed or '(disabled)'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
