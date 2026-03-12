# Nerif Image Compression Service

Nerif provides intelligent image compression functionality that automatically compresses images exceeding a specified size (default 1MB) using lossless compression strategies whenever possible.

## Features

- **Intelligent Compression Algorithms**: Selects optimal compression strategy based on image format and content
- **Lossless Optimization**: Prioritizes lossless compression techniques
- **Format Conversion**: Intelligently converts image formats when beneficial (e.g., PNG → JPEG)
- **Transparency Protection**: Automatically detects and preserves transparency in images
- **Batch Processing**: Supports batch compression of multiple images
- **Metadata Preservation**: Optionally preserves EXIF and other metadata
- **Command Line Tool**: Provides an easy-to-use command line interface

## Installation

Image compression functionality requires additional dependencies:

```bash
pip install nerif[image]
# Or manually install dependencies
pip install Pillow
```

## Quick Start

### Basic Usage

```python
from nerif.utils import compress_image_simple

# Simple compression (if image exceeds 1MB)
was_compressed = compress_image_simple("large_photo.jpg")
if was_compressed:
    print("Image compressed successfully")
```

### Advanced Usage

```python
from nerif.utils import ImageCompressor

# Create custom compressor
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
```

## Compression Strategies

### PNG Images
1. **Lossless Optimization**: Uses maximum compression level for lossless compression
2. **Format Conversion**: Converts to JPEG if no transparency and significant size reduction
3. **Transparency Protection**: Automatically detects transparency; PNG format preserved if present

### JPEG Images
1. **Quality Optimization**: Adjusts JPEG quality to reduce file size
2. **Progressive Encoding**: Enables progressive JPEG for improved loading experience
3. **Metadata Optimization**: Optionally preserves or removes EXIF data

### Other Formats
- Automatically converts to the most suitable format (PNG or JPEG)
- Chooses target format based on transparency presence

## Batch Compression

```python
from nerif.utils import ImageCompressor

compressor = ImageCompressor(size_threshold_mb=1.0)

# Batch compression
image_files = ["photo1.jpg", "photo2.png", "diagram.png"]
results = compressor.compress_batch(
    image_files,
    output_dir="compressed/",     # Output directory
    preserve_structure=True       # Preserve directory structure
)

# View results
for path, was_compressed, ratio, message in results:
    print(f"{path}: {message}")

# Get statistics
stats = compressor.get_compression_stats(results)
print(f"Compressed {stats['compressed_files']} files")
print(f"Average compression ratio: {stats['average_compression_ratio']:.2f}x")
```

## Command Line Tool

```bash
# Basic usage
python -m nerif.cli.compress_image image.jpg

# Custom threshold
python -m nerif.cli.compress_image *.png --threshold 2.0

# Batch compression to output directory
python -m nerif.cli.compress_image photos/*.jpg -o compressed/

# Adjust compression parameters
python -m nerif.cli.compress_image photo.jpg --jpeg-quality 95 --png-level 9

# Preview mode (no actual compression)
python -m nerif.cli.compress_image images/* --dry-run

# Verbose output
python -m nerif.cli.compress_image images/* --verbose
```

### Command Line Options

- `-o, --output`: Output path (file or directory)
- `-t, --threshold`: Compression threshold (MB, default 1.0)
- `--jpeg-quality`: JPEG quality (1-100, default 85)
- `--png-level`: PNG compression level (0-9, default 9)
- `--convert-threshold`: PNG to JPEG threshold (default 0.7)
- `--no-preserve-metadata`: Do not preserve metadata
- `--dry-run`: Preview mode
- `-v, --verbose`: Verbose output

## Configuration Options

### ImageCompressor Parameters

```python
compressor = ImageCompressor(
    size_threshold_mb=1.0,           # Compression threshold (MB)
    jpeg_quality=85,                 # JPEG quality (1-100)
    png_compress_level=9,            # PNG compression level (0-9)
    convert_to_jpeg_threshold=0.7    # PNG→JPEG conversion threshold
)
```

- **size_threshold_mb**: Only compress images exceeding this size
- **jpeg_quality**: JPEG compression quality; lower values mean smaller files but lower quality
- **png_compress_level**: PNG compression level; 9 is maximum compression
- **convert_to_jpeg_threshold**: Convert if PNG→JPEG saves more than this ratio of space

## Example Scenarios

### Website Image Optimization
```python
from nerif.utils import ImageCompressor

# Optimize images for web
web_compressor = ImageCompressor(
    size_threshold_mb=0.5,           # 500KB threshold
    jpeg_quality=80,                 # Medium quality
    convert_to_jpeg_threshold=0.8    # More aggressive conversion
)

# Process uploaded images
results = web_compressor.compress_batch(
    ["upload1.png", "upload2.jpg"],
    output_dir="optimized/"
)
```

### Archive Image Compression
```python
# Optimize for long-term storage
archive_compressor = ImageCompressor(
    size_threshold_mb=2.0,    # 2MB threshold
    jpeg_quality=95,          # High quality preservation
    png_compress_level=9      # Maximum lossless compression
)
```

### Mobile Image Optimization
```python
# Optimize for mobile devices
mobile_compressor = ImageCompressor(
    size_threshold_mb=0.2,           # 200KB threshold
    jpeg_quality=75,                 # Lower quality for smaller files
    convert_to_jpeg_threshold=0.9    # Almost always convert to JPEG
)
```

## Best Practices

1. **Choose Appropriate Thresholds**: 
   - Website images: 0.5-1MB
   - Mobile applications: 0.2-0.5MB
   - Archive purposes: 2-5MB

2. **JPEG Quality Settings**:
   - Network transmission: 70-85
   - General purpose: 85-90
   - High-quality preservation: 90-95

3. **Batch Processing**:
   - Use `preserve_structure=True` to maintain directory structure
   - Set appropriate output directory to avoid overwriting original files
   - Check compression statistics to evaluate effectiveness

4. **Transparency Handling**:
   - PNG images with transparency are automatically protected
   - PNG without transparency may be converted to JPEG
   - Use `convert_to_jpeg_threshold` to control conversion strategy

5. **Metadata Handling**:
   - EXIF data is preserved by default
   - Consider removing for web publishing to reduce file size
   - Use `preserve_metadata=False` to remove metadata

## Technical Details

### Supported Formats
- **Input**: JPEG, PNG, BMP, TIFF, WebP, and other PIL-supported formats
- **Output**: Primarily JPEG and PNG, automatically selected based on content

### Compression Algorithms
- **PNG**: Uses PIL's `optimize=True` and configurable `compress_level`
- **JPEG**: Uses configurable `quality` and `progressive=True`
- **Format Conversion**: Intelligently selects format based on file size comparison

### Performance Considerations
- Large image processing may require significant memory
- Batch processing handles images sequentially to avoid memory peaks
- Compression is CPU-intensive and may take time

### Error Handling
- Throws exceptions for non-existent or corrupted files
- Single file errors don't affect other files in batch processing
- Provides detailed error messages and compression results

## Integration with Nerif Workflow

```python
from nerif.core import nerif
from nerif.utils import compress_image_simple

# Compress image before LLM processing
def process_image_with_llm(image_path):
    # First compress the image
    compress_image_simple(image_path, size_threshold_mb=1.0)
    
    # Then use LLM processing
    if nerif(f"Is the image file {image_path} optimized?"):
        print("Image is optimized and ready to use")
    
    return image_path
```

This image compression service helps you automatically optimize image files, reducing storage space and transmission time while maintaining the best possible image quality.