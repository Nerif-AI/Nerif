# Nerif 图片自动压缩服务

Nerif 提供了智能图片压缩功能，自动压缩超过指定大小（默认 1MB）的图片，采用尽可能无损的压缩策略。

## 功能特性

- **智能压缩算法**: 根据图片格式和内容选择最佳压缩策略
- **无损优化**: 优先使用无损压缩技术
- **格式转换**: 在有益时智能转换图片格式（如 PNG → JPEG）
- **透明度保护**: 自动检测并保护带透明度的图片
- **批量处理**: 支持批量压缩多个图片
- **元数据保护**: 可选择保留 EXIF 等元数据
- **命令行工具**: 提供易用的命令行接口

## 安装

图片压缩功能需要额外的依赖包：

```bash
pip install nerif[image]
# 或者手动安装依赖
pip install Pillow
```

## 快速开始

### 基本使用

```python
from nerif.utils import compress_image_simple

# 简单压缩（如果图片超过 1MB）
was_compressed = compress_image_simple("large_photo.jpg")
if was_compressed:
    print("图片已压缩")
```

### 高级使用

```python
from nerif.utils import ImageCompressor

# 创建自定义压缩器
compressor = ImageCompressor(
    size_threshold_mb=2.0,    # 2MB 阈值
    jpeg_quality=90,          # JPEG 质量
    png_compress_level=9,     # PNG 压缩级别
    convert_to_jpeg_threshold=0.7  # 转换阈值
)

# 压缩单个图片
was_compressed, ratio, message = compressor.compress_image(
    "input.png", 
    "output.jpg"
)
print(f"压缩结果: {message}")
print(f"压缩比: {ratio:.2f}x")
```

## 压缩策略

### PNG 图片
1. **无损优化**: 使用最高压缩级别进行无损压缩
2. **格式转换**: 如果没有透明度且转换为 JPEG 能显著减小文件大小，则转换
3. **透明度保护**: 自动检测透明度，有透明度的 PNG 保持格式不变

### JPEG 图片
1. **质量优化**: 调整 JPEG 质量以减小文件大小
2. **渐进式编码**: 启用渐进式 JPEG 以改善加载体验
3. **元数据优化**: 可选择保留或移除 EXIF 数据

### 其他格式
- 自动转换为最适合的格式（PNG 或 JPEG）
- 根据是否有透明度选择目标格式

## 批量压缩

```python
from nerif.utils import ImageCompressor

compressor = ImageCompressor(size_threshold_mb=1.0)

# 批量压缩
image_files = ["photo1.jpg", "photo2.png", "diagram.png"]
results = compressor.compress_batch(
    image_files,
    output_dir="compressed/",  # 输出目录
    preserve_structure=True    # 保持目录结构
)

# 查看结果
for path, was_compressed, ratio, message in results:
    print(f"{path}: {message}")

# 获取统计信息
stats = compressor.get_compression_stats(results)
print(f"压缩了 {stats['compressed_files']} 个文件")
print(f"平均压缩比: {stats['average_compression_ratio']:.2f}x")
```

## 命令行工具

```bash
# 基本使用
python -m nerif.cli.compress_image image.jpg

# 自定义阈值
python -m nerif.cli.compress_image *.png --threshold 2.0

# 批量压缩到输出目录
python -m nerif.cli.compress_image photos/*.jpg -o compressed/

# 调整压缩参数
python -m nerif.cli.compress_image photo.jpg --jpeg-quality 95 --png-level 9

# 预览模式（不实际压缩）
python -m nerif.cli.compress_image images/* --dry-run

# 详细输出
python -m nerif.cli.compress_image images/* --verbose
```

### 命令行选项

- `-o, --output`: 输出路径（文件或目录）
- `-t, --threshold`: 压缩阈值（MB，默认 1.0）
- `--jpeg-quality`: JPEG 质量（1-100，默认 85）
- `--png-level`: PNG 压缩级别（0-9，默认 9）
- `--convert-threshold`: PNG 转 JPEG 阈值（默认 0.7）
- `--no-preserve-metadata`: 不保留元数据
- `--dry-run`: 预览模式
- `-v, --verbose`: 详细输出

## 配置选项

### ImageCompressor 参数

```python
compressor = ImageCompressor(
    size_threshold_mb=1.0,           # 压缩阈值（MB）
    jpeg_quality=85,                 # JPEG 质量 (1-100)
    png_compress_level=9,            # PNG 压缩级别 (0-9)
    convert_to_jpeg_threshold=0.7    # PNG→JPEG 转换阈值
)
```

- **size_threshold_mb**: 只有超过此大小的图片才会被压缩
- **jpeg_quality**: JPEG 压缩质量，越低文件越小但质量越差
- **png_compress_level**: PNG 压缩级别，9 为最高压缩
- **convert_to_jpeg_threshold**: 如果 PNG→JPEG 能节省超过此比例的空间就转换

## 示例场景

### 网站图片优化
```python
from nerif.utils import ImageCompressor

# 为网站优化图片
web_compressor = ImageCompressor(
    size_threshold_mb=0.5,    # 500KB 阈值
    jpeg_quality=80,          # 适中质量
    convert_to_jpeg_threshold=0.8  # 更积极的转换
)

# 处理上传的图片
results = web_compressor.compress_batch(
    ["upload1.png", "upload2.jpg"],
    output_dir="optimized/"
)
```

### 存档图片压缩
```python
# 为长期存储优化
archive_compressor = ImageCompressor(
    size_threshold_mb=2.0,    # 2MB 阈值
    jpeg_quality=95,          # 高质量保存
    png_compress_level=9      # 最大无损压缩
)
```

### 移动端图片优化
```python
# 为移动设备优化
mobile_compressor = ImageCompressor(
    size_threshold_mb=0.2,    # 200KB 阈值
    jpeg_quality=75,          # 较低质量以减小文件
    convert_to_jpeg_threshold=0.9  # 几乎总是转换为 JPEG
)
```

## 最佳实践

1. **选择合适的阈值**: 
   - 网站图片: 0.5-1MB
   - 移动应用: 0.2-0.5MB
   - 存档用途: 2-5MB

2. **JPEG 质量设置**:
   - 网络传输: 70-85
   - 一般用途: 85-90
   - 高质量保存: 90-95

3. **批量处理**:
   - 使用 `preserve_structure=True` 保持目录结构
   - 设置合适的输出目录避免覆盖原文件
   - 检查压缩统计信息评估效果

4. **透明度处理**:
   - PNG 图片如有透明度会自动保护
   - 无透明度的 PNG 可能被转换为 JPEG
   - 使用 `convert_to_jpeg_threshold` 控制转换策略

5. **元数据处理**:
   - 默认保留 EXIF 数据
   - 网络发布时可考虑移除以减小文件大小
   - 使用 `preserve_metadata=False` 移除元数据

## 技术细节

### 支持的格式
- **输入**: JPEG, PNG, BMP, TIFF, WebP 等 PIL 支持的格式
- **输出**: 主要为 JPEG 和 PNG，根据内容自动选择

### 压缩算法
- **PNG**: 使用 PIL 的 `optimize=True` 和可配置的 `compress_level`
- **JPEG**: 使用可配置的 `quality` 和 `progressive=True`
- **格式转换**: 基于文件大小比较智能选择格式

### 性能考虑
- 大图片处理可能需要较多内存
- 批量处理时按顺序处理避免内存峰值
- 压缩是 CPU 密集型操作，可能需要时间

### 错误处理
- 文件不存在或损坏时抛出异常
- 批量处理时单个文件错误不影响其他文件
- 提供详细的错误信息和压缩结果

## 集成到 Nerif 工作流

```python
from nerif.core import nerif
from nerif.utils import compress_image_simple

# 在 LLM 处理前压缩图片
def process_image_with_llm(image_path):
    # 先压缩图片
    compress_image_simple(image_path, size_threshold_mb=1.0)
    
    # 然后使用 LLM 处理
    if nerif(f"这个图片文件 {image_path} 是否已经优化"):
        print("图片已优化，可以使用")
    
    return image_path
```

这个图片压缩服务可以帮助你自动优化图片文件，减少存储空间和传输时间，同时保持尽可能好的图片质量。