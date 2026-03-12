# Nerif 视觉模型

Nerif 提供了强大的视觉模型，用于通过自然语言交互分析和理解图像。它包含标准的 VisionModel 和增强版的 VisionModelWithCompression（带图像压缩功能）。

## 概述

Nerif 的视觉模型支持以下功能：
- 分析来自本地文件、URL 或 base64 编码的图像
- 对图像内容进行提问
- 生成详细的描述和洞察
- 在对话中处理多张图像
- 自动压缩图像以获得最佳性能（增强版模型）

## 安装

视觉模型包含在 Nerif 基础包中：

```bash
pip install nerif
```

如需使用带压缩功能的增强版模型：
```bash
pip install nerif[image]
```

## 快速开始

### 基础视觉模型

```python
from nerif.model import VisionModel
from nerif.utils import MessageType

# Initialize vision model
vision_model = VisionModel()

# Add an image from URL
vision_model.append_message(
    MessageType.IMAGE_URL,
    "https://example.com/image.jpg"
)

# Ask about the image
vision_model.append_message(
    MessageType.TEXT,
    "What objects are in this image?"
)

# Get response
response = vision_model.chat()
print(response)
```

### 带压缩功能的增强版视觉模型

```python
from nerif.model import VisionModelWithCompression

# Initialize with compression
vision_model = VisionModelWithCompression(
    compress=True,
    compression_quality=85
)

# Add a large image (will be automatically compressed)
vision_model.add_image("large_photo.jpg")
vision_model.add_text("Describe this image in detail")

response = vision_model.chat()
print(response)
```

## API 参考

### VisionModel 类

用于图像分析任务的标准视觉模型。

#### 初始化

```python
vision_model = VisionModel(
    model="gpt-4-vision-preview",    # Model to use
    default_prompt="You are a helpful assistant specializing in image analysis.",
    temperature=0.0                   # Response consistency
)
```

#### 方法

##### append_message(message_type, content)

向对话中添加消息。

**参数：**
- `message_type` (MessageType)：消息类型（IMAGE_URL、IMAGE_FILE、TEXT）
- `content` (str)：内容（URL、文件路径或文本）

**示例：**
```python
# Add image from file
vision_model.append_message(MessageType.IMAGE_FILE, "photo.jpg")

# Add image from URL
vision_model.append_message(MessageType.IMAGE_URL, "https://example.com/image.jpg")

# Add text question
vision_model.append_message(MessageType.TEXT, "What is in this image?")
```

##### chat(max_tokens=None)

根据对话历史生成响应。

**参数：**
- `max_tokens` (int, 可选)：响应的最大 token 数

**返回值：**
- 模型的文本响应

##### reset()

清除对话历史。

```python
vision_model.reset()
```

##### set_max_tokens(max_tokens)

设置响应的最大 token 数。

```python
vision_model.set_max_tokens(500)
```

### VisionModelWithCompression 类

带有自动图像压缩功能的增强版视觉模型。

#### 初始化

```python
vision_model = VisionModelWithCompression(
    model="gpt-4-vision-preview",
    compress=True,                    # Enable compression
    compression_quality=85,           # JPEG quality (1-100)
    max_dimension=2048,              # Max width/height
    convert_to_jpeg=True             # Convert PNG to JPEG
)
```

#### 方法

##### add_image(image, image_type=None)

添加图像并自动压缩。

**参数：**
- `image`：图像数据（文件路径、URL、base64 或字节流）
- `image_type`（可选）：未自动检测时指定图像类型

**示例：**
```python
# Add from file (auto-compressed)
vision_model.add_image("large_photo.png")

# Add from URL
vision_model.add_image("https://example.com/image.jpg")

# Add base64 encoded image
vision_model.add_image(base64_string, image_type="base64")
```

##### add_text(text)

向对话中添加文本消息。

```python
vision_model.add_text("What are the main colors in this image?")
```

##### chat()

使用优化的图像处理生成响应。

```python
response = vision_model.chat()
```

##### reset()

清除对话和缓存的图像。

```python
vision_model.reset()
```

## 消息类型

```python
from nerif.utils import MessageType

# Available message types
MessageType.TEXT        # Text messages
MessageType.IMAGE_URL   # Images from URLs
MessageType.IMAGE_FILE  # Images from local files
```

## 示例

### 基础图像分析

```python
from nerif.model import VisionModel
from nerif.utils import MessageType

vision_model = VisionModel()

# Analyze a local image
vision_model.append_message(MessageType.IMAGE_FILE, "chart.png")
vision_model.append_message(MessageType.TEXT, "What data does this chart show?")

analysis = vision_model.chat()
print(analysis)
```

### 多图比较

```python
from nerif.model import VisionModel
from nerif.utils import MessageType

vision_model = VisionModel()

# Add multiple images
vision_model.append_message(MessageType.IMAGE_FILE, "before.jpg")
vision_model.append_message(MessageType.IMAGE_FILE, "after.jpg")
vision_model.append_message(
    MessageType.TEXT,
    "What are the differences between these two images?"
)

comparison = vision_model.chat()
print(comparison)
```

### 物体检测与计数

```python
from nerif.model import VisionModel
from nerif.utils import MessageType

vision_model = VisionModel()

# Detect objects
vision_model.append_message(MessageType.IMAGE_URL, "https://example.com/parking_lot.jpg")
vision_model.append_message(MessageType.TEXT, "Count all the vehicles in this image and categorize them by type")

result = vision_model.chat()
print(result)
```

### 用于无障碍的图像描述

```python
from nerif.model import VisionModel
from nerif.utils import MessageType

def generate_alt_text(image_path):
    """Generate descriptive alt text for accessibility"""
    vision_model = VisionModel()

    vision_model.append_message(MessageType.IMAGE_FILE, image_path)
    vision_model.append_message(
        MessageType.TEXT,
        "Generate a concise alt text description for this image suitable for screen readers"
    )

    return vision_model.chat()

# Generate alt text
alt_text = generate_alt_text("product_photo.jpg")
print(f"Alt text: {alt_text}")
```

### 带压缩的批量图像处理

```python
from nerif.model import VisionModelWithCompression
from pathlib import Path

def batch_analyze_images(image_folder, question):
    """Analyze multiple images with the same question"""
    vision_model = VisionModelWithCompression(compress=True)
    results = {}

    for image_path in Path(image_folder).glob("*.jpg"):
        vision_model.reset()  # Clear previous conversation

        # Add compressed image
        vision_model.add_image(str(image_path))
        vision_model.add_text(question)

        # Get analysis
        results[image_path.name] = vision_model.chat()

    return results

# Analyze all images
analyses = batch_analyze_images(
    "product_images/",
    "What product is shown and what are its key features?"
)

for filename, analysis in analyses.items():
    print(f"\n{filename}:\n{analysis}")
```

### OCR 与文本提取

```python
from nerif.model import VisionModel
from nerif.utils import MessageType

vision_model = VisionModel()

# Extract text from image
vision_model.append_message(MessageType.IMAGE_FILE, "document.png")
vision_model.append_message(
    MessageType.TEXT,
    "Extract all text from this image and format it nicely"
)

extracted_text = vision_model.chat()
print(extracted_text)
```

### 视觉问答

```python
from nerif.model import VisionModel
from nerif.utils import MessageType

def visual_qa(image_path, questions):
    """Answer multiple questions about an image"""
    vision_model = VisionModel()
    vision_model.append_message(MessageType.IMAGE_FILE, image_path)

    answers = {}
    for question in questions:
        vision_model.append_message(MessageType.TEXT, question)
        answer = vision_model.chat()
        answers[question] = answer

    return answers

# Ask multiple questions
questions = [
    "What is the weather like in this image?",
    "What time of day was this taken?",
    "Are there any people visible?"
]

answers = visual_qa("landscape.jpg", questions)
for q, a in answers.items():
    print(f"Q: {q}\nA: {a}\n")
```

## 最佳实践

### 1. 图像优化
```python
# Use compression for large images
vision_model = VisionModelWithCompression(
    compress=True,
    compression_quality=85,  # Balance quality vs size
    max_dimension=1920      # Resize very large images
)
```

### 2. 清晰的提示词
```python
# Be specific in your questions
vision_model.add_text("List all visible text in the image, including signs, labels, and captions")

# Rather than
vision_model.add_text("What text is there?")
```

### 3. 错误处理
```python
def safe_analyze_image(image_path):
    try:
        vision_model = VisionModel()
        vision_model.append_message(MessageType.IMAGE_FILE, image_path)
        vision_model.append_message(MessageType.TEXT, "Describe this image")
        return vision_model.chat()
    except FileNotFoundError:
        return "Error: Image file not found"
    except Exception as e:
        return f"Error analyzing image: {str(e)}"
```

### 4. 对话管理
```python
# Reset between unrelated analyses
vision_model.reset()

# Keep conversation for follow-up questions
vision_model.append_message(MessageType.IMAGE_FILE, "diagram.png")
vision_model.append_message(MessageType.TEXT, "What does this diagram show?")
first_response = vision_model.chat()

# Follow-up without reset
vision_model.append_message(MessageType.TEXT, "Explain the third step in more detail")
detailed_response = vision_model.chat()
```

## 与其他 Nerif 组件的集成

### 与音频模型结合
```python
from nerif.model import VisionModel, AudioModel
from nerif.utils import MessageType

# Analyze video frame and audio
vision_model = VisionModel()
audio_model = AudioModel()

# Analyze video frame
vision_model.append_message(MessageType.IMAGE_FILE, "video_frame.jpg")
vision_model.append_message(MessageType.TEXT, "What's happening in this scene?")
visual_context = vision_model.chat()

# Transcribe audio
audio_transcript = audio_model.transcribe(Path("video_audio.mp3"))

# Combine insights
print(f"Visual: {visual_context}")
print(f"Audio: {audio_transcript.text}")
```

### 与 Batch API 配合使用
```python
from nerif.batch import BatchFile
from nerif.model import VisionModelWithCompression

# Prepare batch requests for multiple images
def create_vision_batch_requests(image_urls):
    requests = []
    for i, url in enumerate(image_urls):
        requests.append({
            "custom_id": f"vision-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": url}},
                            {"type": "text", "text": "Describe this image"}
                        ]
                    }
                ],
                "max_tokens": 300
            }
        })
    return requests
```

## 性能优化

### 图像压缩设置
```python
# For web images (balance quality and size)
web_model = VisionModelWithCompression(
    compress=True,
    compression_quality=80,
    max_dimension=1280
)

# For detailed analysis (higher quality)
detail_model = VisionModelWithCompression(
    compress=True,
    compression_quality=95,
    max_dimension=2048
)

# For quick previews (maximum compression)
preview_model = VisionModelWithCompression(
    compress=True,
    compression_quality=70,
    max_dimension=800
)
```

## 常见问题与解决方案

### 大图像文件
```python
# Automatically handled by VisionModelWithCompression
vision_model = VisionModelWithCompression(compress=True)
vision_model.add_image("20mb_photo.jpg")  # Automatically compressed
```

### 速率限制
```python
import time

# Add delays for batch processing
for image in images:
    vision_model.reset()
    vision_model.add_image(image)
    vision_model.add_text("Analyze this image")
    result = vision_model.chat()
    time.sleep(1)  # Prevent rate limiting
```

### 内存管理
```python
# Clear model after processing
vision_model.reset()

# For batch processing, process one at a time
def process_images_memory_efficient(image_list):
    vision_model = VisionModelWithCompression()
    for image in image_list:
        vision_model.reset()  # Clear previous image
        vision_model.add_image(image)
        vision_model.add_text("Describe this image")
        yield vision_model.chat()
```

视觉模型提供了强大的图像分析能力，具有灵活的输入选项和自动优化功能，非常适合 Nerif 生态系统中各类计算机视觉任务。
