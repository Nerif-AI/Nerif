# 示例 12：多模态输入

本示例演示如何使用 `MultiModalMessage` 向大语言模型发送**多模态消息**（文本 + 图片）。

## 核心概念

- **`MultiModalMessage`**：一个用于构建包含文本、图片、音频或视频的消息的辅助类。使用流式（可链式调用）API，每个 `add_*` 方法都返回 `self`。
- 支持的内容类型：
  - **文本**：`add_text(text)`
  - **图片**：`add_image_url(url)`、`add_image_path(path)`、`add_image_base64(b64)`
  - **音频**：`add_audio_url(url)`、`add_audio_path(path, format)`、`add_audio_base64(b64, format)`
  - **视频**：`add_video_url(url)`、`add_video_path(path)`

## 代码

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

## 工作原理

1. 创建一个 `MultiModalMessage` 实例。
2. 使用 `add_text()`、`add_image_url()` 等方法添加内容部分。方法支持链式调用：
   ```python
   msg = MultiModalMessage().add_text("Describe this").add_image_url("https://...")
   ```
3. 将消息直接传递给 `model.chat()` —— 它同时接受字符串和 `MultiModalMessage`。
4. 模型会将所有内容部分一起处理并返回文本响应。

## 更多示例

### 从本地文件加载图片
```python
msg = MultiModalMessage().add_text("What's in this photo?").add_image_path("photo.jpg")
```

### 多张图片
```python
msg = (
    MultiModalMessage()
    .add_text("Compare these two images")
    .add_image_url("https://example.com/before.jpg")
    .add_image_url("https://example.com/after.jpg")
)
```

### 文本 + 音频
```python
msg = MultiModalMessage().add_text("Transcribe this audio").add_audio_path("recording.wav")
```

如需具有压缩支持的专用视觉任务，请参阅[视觉模型](../components/model/vision-model.md)文档。
