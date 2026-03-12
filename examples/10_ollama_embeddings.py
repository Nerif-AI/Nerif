"""
Example of using Ollama embedding models with Nerif.

This example demonstrates how to:
1. Create embeddings using Ollama models
2. Compare text similarity using embeddings
3. Use different Ollama embedding models

Prerequisites:
- Ollama running locally (default: http://localhost:11434)
- At least one embedding model pulled (e.g., ollama pull mxbai-embed-large)
"""

import numpy as np

from nerif.model import OllamaEmbeddingModel


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def main():
    # Initialize Ollama embedding model
    # Available models: mxbai-embed-large, nomic-embed-text, all-minilm
    print("Initializing Ollama embedding model...")
    embedder = OllamaEmbeddingModel(model="ollama/mxbai-embed-large")
    
    # Example texts
    texts = [
        "The weather is nice today.",
        "It's a beautiful day outside.",
        "Python is a programming language.",
        "I love coding in Python.",
    ]
    
    print("\nGenerating embeddings for texts:")
    embeddings = []
    for i, text in enumerate(texts):
        print(f"{i+1}. {text}")
        embedding = embedder.embed(text)
        embeddings.append(embedding)
    
    # Calculate similarities
    print("\nSimilarity matrix:")
    print("   ", end="")
    for i in range(len(texts)):
        print(f"  {i+1}  ", end="")
    print()
    
    for i in range(len(texts)):
        print(f"{i+1}: ", end="")
        for j in range(len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"{sim:.3f} ", end="")
        print()
    
    # Find most similar pairs
    print("\nMost similar text pairs:")
    similarities = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append((sim, i, j))
    
    similarities.sort(reverse=True)
    for sim, i, j in similarities[:3]:
        print(f"- \"{texts[i]}\" <-> \"{texts[j]}\": {sim:.3f}")
    
    # Example with different models
    print("\nComparing different Ollama embedding models:")
    models = ["ollama/mxbai-embed-large", "ollama/nomic-embed-text", "ollama/all-minilm"]
    test_text = "Machine learning is fascinating."
    
    for model_name in models:
        try:
            model = OllamaEmbeddingModel(model=model_name)
            embedding = model.embed(test_text)
            print(f"- {model_name}: {len(embedding)} dimensions")
        except Exception as e:
            print(f"- {model_name}: Not available ({str(e)})")


if __name__ == "__main__":
    main()