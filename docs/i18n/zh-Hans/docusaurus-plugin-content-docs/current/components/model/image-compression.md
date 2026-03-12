# Nerif 图像压缩服务

Nerif 提供了智能图像压缩功能，能够自动压缩超过指定大小（默认 1MB）的图像，尽可能采用无损压缩策略。

## 功能特性

- **智能压缩算法**：根据图像格式和内容选择最优压缩策略
- **无损优化**：优先使用无损压缩技术
- **格式转换**：在有益时智能转换图像格式（例如 PNG → JPEG）
- **透明度保护**：自动检测并保留图像中的透明度
- **批量处理**：支持多张图像的批量压缩
- **元数据保留**：可选保留 EXIF 及其他元数据
- **命令行工具**：提供易用的命令行界面

## 安装

图像压缩功能需要安装额外的依赖：

```bash
pip install nerif[image]
# Or manually install dependencies
pip install Pillow
```

## 快速开始

### 基础用法

```python
from nerif.utils import compress_image_simple

# Simple compression (if image exceeds 1MB)
was_compressed = compress_image_simple("large_photo.jpg")
if was_compressed:
    print("Image compressed successfully")
```

### 高级用法

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

## 压缩策略

### PNG 图像
1. **无损优化**：使用最大压缩级别进行无损压缩
2. **格式转换**：如果没有透明度且能显著减小文件大小，则转换为 JPEG
3. **透明度保护**：自动检测透明度；如存在透明度则保留 PNG 格式

### JPEG 图像
1. **质量优化**：调整 JPEG 质量以减小文件大小
2. **渐进式编码**：启用渐进式 JPEG 以改善加载体验
3. **元数据优化**：可选保留或移除 EXIF 数据

### 其他格式
- 自动转换为最合适的格式（PNG 或 JPEG）
- 根据透明度选择目标格式

## 批量压缩

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

## 命令行工具

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

### 命令行选项

- `-o, --output`：输出路径（文件或目录）
- `-t, --threshold`：压缩阈值（MB，默认 1.0）
- `--jpeg-quality`：JPEG 质量（1-100，默认 85）
- `--png-level`：PNG 压缩级别（0-9，默认 9）
- `--convert-threshold`：PNG 到 JPEG 的转换阈值（默认 0.7）
- `--no-preserve-metadata`：不保留元数据
- `--dry-run`：预览模式
- `-v, --verbose`：详细输出

## 配置选项

### ImageCompressor 参数

```python
compressor = ImageCompressor(
    size_threshold_mb=1.0,           # Compression threshold (MB)
    jpeg_quality=85,                 # JPEG quality (1-100)
    png_compress_level=9,            # PNG compression level (0-9)
    convert_to_jpeg_threshold=0.7    # PNG→JPEG conversion threshold
)
```

- **size_threshold_mb**：仅压缩超过此大小的图像
- **jpeg_quality**：JPEG 压缩质量；值越低文件越小但质量越低
- **png_compress_level**：PNG 压缩级别；9 为最大压缩
- **convert_to_jpeg_threshold**：当 PNG→JPEG 节省超过此比例的空间时进行转换

## 应用场景示例

### 网站图像优化
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

### 归档图像压缩
```python
# Optimize for long-term storage
archive_compressor = ImageCompressor(
    size_threshold_mb=2.0,    # 2MB threshold
    jpeg_quality=95,          # High quality preservation
    png_compress_level=9      # Maximum lossless compression
)
```

### 移动端图像优化
```python
# Optimize for mobile devices
mobile_compressor = ImageCompressor(
    size_threshold_mb=0.2,           # 200KB threshold
    jpeg_quality=75,                 # Lower quality for smaller files
    convert_to_jpeg_threshold=0.9    # Almost always convert to JPEG
)
```

## 最佳实践

1. **选择合适的阈值**：
   - 网站图像：0.5-1MB
   - 移动应用：0.2-0.5MB
   - 归档用途：2-5MB

2. **JPEG 质量设置**：
   - 网络传输：70-85
   - 通用场景：85-90
   - 高质量保存：90-95

3. **批量处理**：
   - 使用 `preserve_structure=True` 保持目录结构
   - 设置适当的输出目录以避免覆盖原始文件
   - 检查压缩统计数据以评估效果

4. **透明度处理**：
   - 带透明度的 PNG 图像会自动受到保护
   - 不带透明度的 PNG 可能会被转换为 JPEG
   - 使用 `convert_to_jpeg_threshold` 控制转换策略

5. **元数据处理**：
   - 默认保留 EXIF 数据
   - 考虑在网页发布时移除以减小文件大小
   - 使用 `preserve_metadata=False` 移除元数据

## 技术细节

### 支持的格式
- **输入**：JPEG、PNG、BMP、TIFF、WebP 及其他 PIL 支持的格式
- **输出**：主要为 JPEG 和 PNG，根据内容自动选择

### 压缩算法
- **PNG**：使用 PIL 的 `optimize=True` 和可配置的 `compress_level`
- **JPEG**：使用可配置的 `quality` 和 `progressive=True`
- **格式转换**：根据文件大小比较智能选择格式

### 性能考虑
- 处理大图像可能需要大量内存
- 批量处理按顺序处理图像以避免内存峰值
- 压缩是 CPU 密集型操作，可能需要一定时间

### 错误处理
- 对于不存在或损坏的文件会抛出异常
- 批量处理中单个文件的错误不会影响其他文件
- 提供详细的错误信息和压缩结果

## 与 Nerif 工作流集成

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

图像压缩服务帮助你自动优化图像文件，在保持最佳图像质量的同时减少存储空间和传输时间。
