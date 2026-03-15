"""Example 23: Conversation Memory with SimpleChatModel.

This example demonstrates how to use ConversationMemory to manage
conversation history with sliding windows and persistence.

NOTE: This example requires a valid OPENAI_API_KEY to run the live
      chat calls. The save/load section works without an API key.
"""

import os
import tempfile

from nerif.memory import ConversationMemory
from nerif.model import SimpleChatModel

# ---------------------------------------------------------------------------
# 1. Basic usage with a sliding message window
# ---------------------------------------------------------------------------

memory = ConversationMemory(max_messages=10)
model = SimpleChatModel(memory=memory)

# These calls require a real API key; they are shown for illustration.
if os.getenv("OPENAI_API_KEY"):
    model.chat("Hello!", append=True)
    model.chat("What's 2+2?", append=True)
    print(f"Messages in memory: {len(memory.get_messages())}")
else:
    # Manually seed messages to show the API without a real key.
    memory.add_message("user", "Hello!")
    memory.add_message("assistant", "Hi there! How can I help you?")
    memory.add_message("user", "What's 2+2?")
    memory.add_message("assistant", "2+2 equals 4.")
    print(f"Messages in memory: {len(memory.get_messages())}")

# ---------------------------------------------------------------------------
# 2. Save and load conversation state
# ---------------------------------------------------------------------------

with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    path = f.name

memory.save(path)
loaded = ConversationMemory.load(path)
print(f"Loaded {len(loaded.get_messages())} messages from {path}")

os.unlink(path)

# ---------------------------------------------------------------------------
# 3. Summarization mode (requires API key)
# ---------------------------------------------------------------------------

print("\n--- Summarization mode ---")
mem_summarize = ConversationMemory(
    max_messages=6,
    summarize=True,
    summarize_model="gpt-4o-mini",
    summary_prompt="Summarize the following conversation concisely, preserving key facts and context:",
)

# Add synthetic messages to illustrate the window without API calls
for i in range(8):
    mem_summarize.add_message("user", f"Question number {i}")
    mem_summarize.add_message("assistant", f"Answer number {i}")

print(f"Non-system messages retained: {len([m for m in mem_summarize._messages if m['role'] != 'system'])}")
if mem_summarize._summary:
    print(f"Summary exists: {mem_summarize._summary[:80]}...")
else:
    print("No API key — summarization was not triggered (uses get_model_response internally).")

# ---------------------------------------------------------------------------
# 4. Token-budget window
# ---------------------------------------------------------------------------

print("\n--- Token budget window ---")
mem_tokens = ConversationMemory(max_tokens=50)
for i in range(10):
    # Each message is roughly 10 tokens (40 chars / 4)
    mem_tokens.add_message("user", f"This is message number {i:03d} with some padding.")

print(f"Approximate token count: {mem_tokens.token_count()} (target <= 50)")
print(f"Messages retained: {len(mem_tokens._messages)}")

# ---------------------------------------------------------------------------
# 5. Using memory with NerifAgent
# ---------------------------------------------------------------------------

print("\n--- NerifAgent with memory ---")
from nerif.agent import NerifAgent  # noqa: E402

agent_memory = ConversationMemory(max_messages=20)
agent = NerifAgent(memory=agent_memory)
print(f"Agent memory attached: {agent.model.memory is agent_memory}")
print(f"System message seeded: {len(agent_memory._messages)} message(s) in memory")
