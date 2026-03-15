"""Example 29: Combined Features - Memory + RAG + Observability.

Demonstrates using Nerif's v1.2.0 features together:
- ConversationMemory for context management
- NumpyVectorStore + SimpleRAG for retrieval
- NerifTokenCounter with callbacks for cost tracking
- Custom exceptions for error handling

Requires: OPENAI_API_KEY environment variable
"""

from nerif.exceptions import NerifError, ProviderError
from nerif.memory import ConversationMemory
from nerif.model import SimpleChatModel
from nerif.rag import NumpyVectorStore, SimpleRAG  # noqa: F401
from nerif.utils import NerifTokenCounter

# --- 1. Set up observability ---
counter = NerifTokenCounter()
counter.on_request_end = lambda e: print(f"  [cost] {e.model}: {e.latency_ms:.0f}ms, ${e.cost_usd:.6f}")

# --- 2. Set up conversation memory ---
memory = ConversationMemory(
    max_messages=20,
    summarize=True,
    summarize_model="gpt-4o-mini",
    counter=counter,  # Track summarization costs too
)

# --- 3. Create model with memory + counter ---
model = SimpleChatModel(counter=counter, memory=memory)

# --- 4. Set up RAG knowledge base ---
# In production, use SimpleEmbeddingModel for real embeddings
store = NumpyVectorStore()
store.add(
    texts=[
        "Nerif supports streaming responses via stream_chat() and astream_chat().",
        "ConversationMemory provides sliding window and auto-summarization.",
        "NerifTokenCounter tracks latency, cost, and success rate.",
        "Use RetryConfig for exponential backoff on API failures.",
        "FormatVerifierPydantic validates LLM output against Pydantic models.",
    ],
    embeddings=[
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ],
)

print("=== Nerif v1.2.0 Combined Features Demo ===\n")

# --- 5. Use RAG + Memory + Counter together ---
try:
    # Direct chat with memory tracking
    response = model.chat("What can Nerif do?", append=True)
    print(f"Q: What can Nerif do?\nA: {response[:100]}...\n")

    # RAG-enhanced query
    # mock_embed = SimpleEmbeddingModel()  # Use real embeddings in production
    # rag = SimpleRAG(embed_model=mock_embed, store=store)
    # result = rag.query_with_context("How does streaming work?", model=model)

    # Follow-up with memory context
    response = model.chat("Can you summarize what we discussed?", append=True)
    print(f"Q: Summarize our discussion\nA: {response[:100]}...\n")

except ProviderError as e:
    print(f"Provider error ({e.provider}): {e}")
    print(f"Status code: {e.status_code}")
except NerifError as e:
    print(f"Nerif error: {e}")

# --- 6. Print observability summary ---
print("\n" + counter.summary())

# --- 7. Save conversation for later ---
memory.save("/tmp/nerif_demo_conversation.json")
print(f"\nConversation saved. Memory has {len(memory._messages)} messages.")

# --- 8. Load and continue later ---
# loaded_memory = ConversationMemory.load("/tmp/nerif_demo_conversation.json")
# model2 = SimpleChatModel(memory=loaded_memory, counter=counter)
# model2.chat("Continue from where we left off", append=True)
