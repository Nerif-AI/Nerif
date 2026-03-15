#!/usr/bin/env python3
"""Main CLI entry point for Nerif commands."""

import sys


def main():
    """Route to subcommands."""
    if len(sys.argv) < 2:
        print("Usage: nerif <command> [args]")
        print()
        print("Commands:")
        print("  check             Check API key configuration and available providers")
        print("  test-model MODEL  Test a model's connectivity and measure latency")
        print("  models            List supported model prefixes and providers")
        print("  compress          Compress images (nerif-compress)")
        return 1

    command = sys.argv[1]
    # Remove the subcommand from argv so sub-parsers work correctly
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "check":
        from .check import main as check_main

        return check_main()
    elif command == "test-model":
        from .test_model import main as test_model_main

        return test_model_main()
    elif command == "models":
        return _list_models()
    elif command == "compress":
        from .compress_image import main as compress_main

        return compress_main()
    else:
        print(f"Unknown command: {command}")
        print("Run 'nerif' without arguments to see available commands.")
        return 1


def _list_models():
    """List supported model prefixes and their providers."""
    print("Supported Model Prefixes")
    print("=" * 50)
    prefixes = [
        ("(no prefix)", "OpenAI", "gpt-4o, gpt-4o-mini, gpt-3.5-turbo, gpt-o1"),
        ("anthropic/", "Anthropic", "claude-3-5-sonnet, claude-3-haiku, claude-3-opus"),
        ("gemini/", "Google Gemini", "gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash"),
        ("openrouter/", "OpenRouter", "Any model via OpenRouter API"),
        ("ollama/", "Ollama (local)", "llama3.1, mistral, codellama, etc."),
        ("vllm/", "vLLM (local)", "Any model served by vLLM"),
        ("sllm/", "SLLM (local)", "Any model served by SLLM"),
        ("custom_openai/", "Custom OpenAI", "Any OpenAI-compatible endpoint"),
    ]

    for prefix, provider, examples in prefixes:
        print(f"\n  {prefix:<20} -> {provider}")
        print(f"  {'':20}    Examples: {examples}")

    print("\n\nUsage: model = SimpleChatModel(model='anthropic/claude-3-5-sonnet-20241022')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
