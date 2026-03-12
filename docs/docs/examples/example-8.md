# Example 8: Vision Model with Auto Compression

```python
from nerif.model import VisionModel
from nerif.utils import ImageCompressor, MessageType
import os

# Setup auto-compression for vision tasks
compressor = ImageCompressor(
    size_threshold_mb=1.0,  # Compress images > 1MB
    jpeg_quality=85
)

def analyze_large_image(image_path, question):
    # Compress if needed
    compressed_path = image_path
    was_compressed, ratio, _ = compressor.compress_image(
        image_path,
        f"temp_{os.path.basename(image_path)}"
    )
    
    if was_compressed:
        compressed_path = f"temp_{os.path.basename(image_path)}"
        print(f"Image compressed by {ratio:.2f}x for faster processing")
    
    # Analyze with vision model
    vision_model = VisionModel()
    vision_model.append_message(MessageType.IMAGE_FILE, compressed_path)
    vision_model.append_message(MessageType.TEXT, question)
    
    response = vision_model.chat()
    
    # Cleanup temporary file if created
    if was_compressed:
        os.remove(compressed_path)
    
    return response

# Example usage
result = analyze_large_image(
    "very_large_photo.jpg",
    "What are the main objects in this image?"
)
print(result)
```