#!/usr/bin/env python3
"""CLI command to test model connectivity and measure latency."""

import argparse
import sys
import time


def main():
    """Send a test prompt to a model and report results."""
    parser = argparse.ArgumentParser(
        description="Test a model's connectivity, latency, and token usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nerif test-model gpt-4o
  nerif test-model anthropic/claude-3-5-sonnet-20241022
  nerif test-model ollama/llama3.1
  nerif test-model gemini/gemini-1.5-flash
""",
    )
    parser.add_argument("model", help="Model name (e.g. gpt-4o, anthropic/claude-3-5-sonnet)")
    parser.add_argument("--prompt", default="Say 'hello' in one word.", help="Test prompt to send")
    args = parser.parse_args()

    from nerif.utils import NerifTokenCounter, get_model_response

    counter = NerifTokenCounter()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]

    print(f"Testing model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print("-" * 40)

    start = time.monotonic()
    try:
        response = get_model_response(
            messages=messages,
            model=args.model,
            counter=counter,
            max_tokens=50,
        )
        latency = (time.monotonic() - start) * 1000

        text = response.choices[0].message.content if response.choices else "(no response)"
        print(f"Response: {text}")
        print(f"Latency: {latency:.0f}ms")
        print(f"Tokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
        if counter.total_cost() > 0:
            print(f"Est. cost: ${counter.total_cost():.6f}")
        print("\nModel is working!")
        return 0

    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        print(f"FAILED after {latency:.0f}ms")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
