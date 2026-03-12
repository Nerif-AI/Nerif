# Example 11: Structured Output / JSON Mode

This example shows how to use **structured output** (JSON mode) to get well-formatted JSON responses from the LLM, and how to parse them robustly with `NerifFormat.json_parse()`.

## Key Concepts

- **`response_format`**: Pass `{"type": "json_object"}` to `model.chat()` to instruct the model to return valid JSON.
- **`NerifFormat.json_parse()`**: A robust static method that extracts JSON from LLM responses, handling common patterns like markdown code blocks (`` ```json ... ``` ``), plain JSON, and JSON embedded in surrounding text.

## Code

```python
from nerif.model import SimpleChatModel
from nerif.utils import NerifFormat

# Use JSON mode
model = SimpleChatModel(model="gpt-4o")
result = model.chat(
    "List three programming languages with their year of creation. "
    "Respond in JSON format as an array of objects with 'name' and 'year' fields.",
    response_format={"type": "json_object"},
)

print("Raw response:", result)

# Parse the JSON robustly
parsed = NerifFormat.json_parse(result)
print("Parsed:", parsed)
```

## How It Works

1. Set `response_format={"type": "json_object"}` in the `chat()` call. This tells the model to output valid JSON.
2. The raw response is a JSON string. Use `NerifFormat.json_parse()` to safely parse it.
3. `json_parse()` handles edge cases: markdown code blocks, extra whitespace, and embedded JSON within text.

## When to Use

- When you need structured data from LLM responses (lists, objects, etc.)
- When building pipelines that consume LLM output programmatically
- Combined with `FormatVerifierJson` for even more flexible extraction from unstructured responses
