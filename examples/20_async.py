"""Example: Async/await support for concurrent LLM calls.

Demonstrates achat(), aembed(), and astream_chat() for high-throughput
applications using asyncio.
"""

import asyncio
from nerif.model import SimpleChatModel, SimpleEmbeddingModel


async def main():
    model = SimpleChatModel()

    # Basic async chat
    result = await model.achat("What is the capital of France?")
    print(f"Answer: {result}")

    # Concurrent calls with asyncio.gather
    models = [SimpleChatModel() for _ in range(3)]
    questions = [
        "What is 2 + 2?",
        "Name a primary color.",
        "What planet is closest to the sun?",
    ]
    results = await asyncio.gather(*(m.achat(q) for m, q in zip(models, questions)))
    for q, r in zip(questions, results):
        print(f"Q: {q}\nA: {r}\n")

    # Async streaming
    print("Async streaming:")
    async for chunk in model.astream_chat("Count from 1 to 5."):
        print(chunk, end="", flush=True)
    print()

    # Async embedding
    embed_model = SimpleEmbeddingModel()
    embedding = await embed_model.aembed("Hello world")
    print(f"Embedding dimensions: {len(embedding)}")


asyncio.run(main())
