# Nerif Vision Model

Nerif provides powerful vision models for analyzing and understanding images through natural language interactions. It includes both a standard VisionModel and an enhanced VisionModelWithCompression for optimized performance.

## Overview

Nerif's vision models enable you to:
- Analyze images from local files, URLs, or base64 encoded data
- Ask questions about image content
- Generate detailed descriptions and insights
- Process multiple images in a conversation
- Automatically compress images for optimal performance (enhanced model)

## Installation

The Vision Models are included in the base Nerif package:

```bash
pip install nerif
```

For the enhanced model with compression capabilities:
```bash
pip install nerif[image]
```

## Quick Start

### Basic Vision Model

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

### Enhanced Vision Model with Compression

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

## API Reference

### VisionModel Class

The standard vision model for image analysis tasks.

#### Initialization

```python
vision_model = VisionModel(
    model="gpt-4-vision-preview",    # Model to use
    default_prompt="You are a helpful assistant specializing in image analysis.",
    temperature=0.0                   # Response consistency
)
```

#### Methods

##### append_message(message_type, content)

Adds a message to the conversation.

**Parameters:**
- `message_type` (MessageType): Type of message (IMAGE_URL, IMAGE_FILE, TEXT)
- `content` (str): The content (URL, file path, or text)

**Example:**
```python
# Add image from file
vision_model.append_message(MessageType.IMAGE_FILE, "photo.jpg")

# Add image from URL
vision_model.append_message(MessageType.IMAGE_URL, "https://example.com/image.jpg")

# Add text question
vision_model.append_message(MessageType.TEXT, "What is in this image?")
```

##### chat(max_tokens=None)

Generates a response based on the conversation history.

**Parameters:**
- `max_tokens` (int, optional): Maximum tokens in response

**Returns:**
- Response text from the model

##### reset()

Clears the conversation history.

```python
vision_model.reset()
```

##### set_max_tokens(max_tokens)

Sets the maximum tokens for responses.

```python
vision_model.set_max_tokens(500)
```

### VisionModelWithCompression Class

Enhanced vision model with automatic image compression.

#### Initialization

```python
vision_model = VisionModelWithCompression(
    model="gpt-4-vision-preview",
    compress=True,                    # Enable compression
    compression_quality=85,           # JPEG quality (1-100)
    max_dimension=2048,              # Max width/height
    convert_to_jpeg=True             # Convert PNG to JPEG
)
```

#### Methods

##### add_image(image, image_type=None)

Adds an image with automatic compression.

**Parameters:**
- `image`: Image data (file path, URL, base64, or bytes)
- `image_type` (optional): Type of image if not auto-detected

**Example:**
```python
# Add from file (auto-compressed)
vision_model.add_image("large_photo.png")

# Add from URL
vision_model.add_image("https://example.com/image.jpg")

# Add base64 encoded image
vision_model.add_image(base64_string, image_type="base64")
```

##### add_text(text)

Adds a text message to the conversation.

```python
vision_model.add_text("What are the main colors in this image?")
```

##### chat()

Generates a response with optimized image handling.

```python
response = vision_model.chat()
```

##### reset()

Clears conversation and cached images.

```python
vision_model.reset()
```

## Message Types

```python
from nerif.utils import MessageType

# Available message types
MessageType.TEXT        # Text messages
MessageType.IMAGE_URL   # Images from URLs
MessageType.IMAGE_FILE  # Images from local files
```

## Examples

### Basic Image Analysis

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

### Multiple Image Comparison

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

### Object Detection and Counting

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

### Image Description for Accessibility

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

### Batch Image Processing with Compression

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

### OCR and Text Extraction

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

### Visual Question Answering

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

## Best Practices

### 1. Image Optimization
```python
# Use compression for large images
vision_model = VisionModelWithCompression(
    compress=True,
    compression_quality=85,  # Balance quality vs size
    max_dimension=1920      # Resize very large images
)
```

### 2. Clear Prompts
```python
# Be specific in your questions
vision_model.add_text("List all visible text in the image, including signs, labels, and captions")

# Rather than
vision_model.add_text("What text is there?")
```

### 3. Error Handling
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

### 4. Conversation Management
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

## Integration with Other Nerif Components

### Combine with Audio Model
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

### Use with Batch API
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

## Performance Optimization

### Image Compression Settings
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

## Common Issues and Solutions

### Large Image Files
```python
# Automatically handled by VisionModelWithCompression
vision_model = VisionModelWithCompression(compress=True)
vision_model.add_image("20mb_photo.jpg")  # Automatically compressed
```

### Rate Limiting
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

### Memory Management
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

The Vision Models provide powerful image analysis capabilities with flexible input options and automatic optimization, making them ideal for a wide range of computer vision tasks within the Nerif ecosystem.