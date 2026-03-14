"""Example: Multi-provider usage - same task across OpenAI, Anthropic, and Gemini.

Demonstrates Nerif's provider-agnostic API. Set the relevant API keys.
"""
from nerif.model import SimpleChatModel

providers = {
    "OpenAI": "gpt-4o-mini",
    "Anthropic": "anthropic/claude-3-haiku-20240307",
    "Gemini": "gemini/gemini-2.0-flash",
}

prompt = "In exactly one sentence, what is machine learning?"

for name, model_name in providers.items():
    try:
        model = SimpleChatModel(model=model_name)
        response = model.chat(prompt)
        print(f"[{name}] {response}")
    except Exception as e:
        print(f"[{name}] Skipped: {e}")
