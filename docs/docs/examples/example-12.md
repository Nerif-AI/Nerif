# Example 12: Multi-Modal Input

```python
from nerif.model import MultiModalMessage, SimpleChatModel

model = SimpleChatModel(model="gpt-4o")

# Build a multi-modal message with text and an image URL
msg = MultiModalMessage()
msg.add_text("What do you see in this image?")
msg.add_image_url(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
    "PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
)

result = model.chat(msg)
print("Response:", result)
```
