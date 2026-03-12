# Example 11: Structured Output / JSON Mode

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
