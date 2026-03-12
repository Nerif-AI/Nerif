# Example 12: Multi-Modal Input

This example demonstrates how to send **multi-modal messages** (text + images) to the LLM using `MultiModalMessage`.

## Key Concepts

- **`MultiModalMessage`**: A helper class for building messages that contain text, images, audio, or video. Uses a fluent (chainable) API where each `add_*` method returns `self`.
- Supported content types:
  - **Text**: `add_text(text)`
  - **Images**: `add_image_url(url)`, `add_image_path(path)`, `add_image_base64(b64)`
  - **Audio**: `add_audio_url(url)`, `add_audio_path(path, format)`, `add_audio_base64(b64, format)`
  - **Video**: `add_video_url(url)`, `add_video_path(path)`

## Code

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

## How It Works

1. Create a `MultiModalMessage` instance.
2. Add content parts using `add_text()`, `add_image_url()`, etc. Methods are chainable:
   ```python
   msg = MultiModalMessage().add_text("Describe this").add_image_url("https://...")
   ```
3. Pass the message directly to `model.chat()` — it accepts both strings and `MultiModalMessage`.
4. The model processes all content parts together and returns a text response.

## More Examples

### Image from local file
```python
msg = MultiModalMessage().add_text("What's in this photo?").add_image_path("photo.jpg")
```

### Multiple images
```python
msg = (
    MultiModalMessage()
    .add_text("Compare these two images")
    .add_image_url("https://example.com/before.jpg")
    .add_image_url("https://example.com/after.jpg")
)
```

### Text + Audio
```python
msg = MultiModalMessage().add_text("Transcribe this audio").add_audio_path("recording.wav")
```

For dedicated vision tasks with compression support, see the [Vision Model](../components/model/vision-model.md) docs.
