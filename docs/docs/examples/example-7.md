# Example 7: Image Compression

```python
from nerif.utils import ImageCompressor, compress_image_simple

# Simple compression (compress if image > 1MB)
was_compressed = compress_image_simple("large_photo.jpg")
if was_compressed:
    print("Image was compressed")

# Advanced compression with custom settings
compressor = ImageCompressor(
    size_threshold_mb=2.0,           # 2MB threshold
    jpeg_quality=90,                 # JPEG quality
    png_compress_level=9,            # PNG compression level
    convert_to_jpeg_threshold=0.7    # Conversion threshold
)

# Compress single image
was_compressed, ratio, message = compressor.compress_image(
    "input.png", 
    "output.jpg"
)
print(f"Compression result: {message}")
print(f"Compression ratio: {ratio:.2f}x")

# Batch compression
image_files = ["photo1.jpg", "photo2.png", "diagram.png"]
results = compressor.compress_batch(
    image_files,
    output_dir="compressed/",
    preserve_structure=True
)

# View results
for path, was_compressed, ratio, message in results:
    print(f"{path}: {message}")

# Get statistics
stats = compressor.get_compression_stats(results)
print(f"Compressed {stats['compressed_files']} files")
print(f"Average compression ratio: {stats['average_compression_ratio']:.2f}x")
```