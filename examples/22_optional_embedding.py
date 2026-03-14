"""Example: Using nerif without an embedding model.

Demonstrates that nerif() and nerif_match() work without embedding
by falling back to text-based matching when embedding is disabled.
"""

from nerif.core import Nerif, NerifMatchString, nerif

# When NERIF_DEFAULT_EMBEDDING_MODEL="" is set, embedding is disabled.
# nerif() and nerif_match() will use text fallback instead of embedding mode.

# You can also explicitly pass embed_model=None
judge = Nerif(model="gpt-4o", embed_model=None)
result = judge.judge("the sky is blue")
print(f"The sky is blue: {result}")  # True

# nerif_match works without embedding too
matcher = NerifMatchString(
    choices=["sunny", "rainy", "cloudy"],
    model="gpt-4o",
    embed_model=None,
)
idx = matcher.match("The weather is warm and bright")
print(f"Weather match: {['sunny', 'rainy', 'cloudy'][idx]}")

# Regular nerif() still works - uses logits mode first, text fallback if needed
result = nerif("Python is a programming language")
print(f"Python is a language: {result}")  # True
