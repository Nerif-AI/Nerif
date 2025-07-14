#!/usr/bin/env python3
"""
Example 11: Automatic Image Compression Service

Demonstrates how to use Nerif's image compression feature to automatically compress
images larger than 1MB, using lossless compression strategies whenever possible.
"""

import tempfile
from pathlib import Path
from PIL import Image
import shutil

from nerif.utils.image_compress import ImageCompressor, compress_image_simple


def create_sample_images(output_dir: Path) -> None:
    """Create sample images for testing."""
    print("Creating sample images...")
    
    # Create a large JPEG image (high quality)
    large_jpg = Image.new('RGB', (2000, 1500), color='red')
    # Add gradient effect to make image more realistic
    pixels = large_jpg.load()
    for x in range(large_jpg.width):
        for y in range(large_jpg.height):
            r = int(255 * (1 - x / large_jpg.width))
            g = int(255 * (y / large_jpg.height))
            b = 100
            pixels[x, y] = (r, g, b)
    
    large_jpg_path = output_dir / "large_photo.jpg"
    large_jpg.save(large_jpg_path, format='JPEG', quality=95)
    
    # Create a large PNG image
    large_png = Image.new('RGB', (1800, 1200), color='blue')
    # Add some patterns
    pixels = large_png.load()
    for x in range(0, large_png.width, 50):
        for y in range(large_png.height):
            pixels[x, y] = (255, 255, 255)
    
    large_png_path = output_dir / "large_diagram.png"
    large_png.save(large_png_path, format='PNG')
    
    # Create a PNG with transparency
    transparent_png = Image.new('RGBA', (1500, 1000), (255, 0, 0, 128))
    # Add some fully transparent areas
    pixels = transparent_png.load()
    for x in range(500, 1000):
        for y in range(300, 700):
            pixels[x, y] = (0, 0, 255, 0)  # Fully transparent
    
    transparent_png_path = output_dir / "transparent_logo.png"
    transparent_png.save(transparent_png_path, format='PNG')
    
    # Create a small image (should skip compression)
    small_jpg = Image.new('RGB', (100, 100), color='green')
    small_jpg_path = output_dir / "small_icon.jpg"
    small_jpg.save(small_jpg_path, format='JPEG', quality=90)
    
    print(f"Created 4 sample images in {output_dir}")
    for img_path in [large_jpg_path, large_png_path, transparent_png_path, small_jpg_path]:
        size_mb = img_path.stat().st_size / 1024 / 1024
        print(f"  - {img_path.name}: {size_mb:.2f} MB")


def example_basic_compression():
    """Basic image compression example."""
    print("\n=== Basic Image Compression Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample images
        create_sample_images(temp_path)
        
        # Use simple compression function
        print("\nUsing simple compression function:")
        large_jpg = temp_path / "large_photo.jpg"
        
        original_size = large_jpg.stat().st_size
        print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
        
        # Compress image (default threshold 1MB)
        was_compressed = compress_image_simple(large_jpg)
        
        new_size = large_jpg.stat().st_size
        if was_compressed:
            compression_ratio = original_size / new_size
            print(f"Compressed size: {new_size / 1024 / 1024:.2f} MB")
            print(f"Compression ratio: {compression_ratio:.2f}x")
        else:
            print("Image not compressed (possibly below threshold)")


def example_advanced_compression():
    """Advanced image compression example."""
    print("\n=== Advanced Image Compression Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample images
        create_sample_images(temp_path)
        
        # Create custom compressor
        compressor = ImageCompressor(
            size_threshold_mb=0.5,  # 500KB threshold
            jpeg_quality=90,        # High quality JPEG
            png_compress_level=9,   # Maximum PNG compression
            convert_to_jpeg_threshold=0.8  # More conservative format conversion
        )
        
        print(f"\nCompressor settings:")
        print(f"  - Size threshold: {compressor.size_threshold_bytes / 1024 / 1024:.1f} MB")
        print(f"  - JPEG quality: {compressor.jpeg_quality}")
        print(f"  - PNG compression level: {compressor.png_compress_level}")
        
        # Compress single image
        print(f"\nCompressing single image:")
        large_png = temp_path / "large_diagram.png"
        
        was_compressed, ratio, message = compressor.compress_image(large_png)
        print(f"  {large_png.name}: {message}")
        
        # Compress to different output file
        print(f"\nCompressing to new file:")
        original_jpg = temp_path / "large_photo.jpg"
        compressed_jpg = temp_path / "compressed_photo.jpg"
        
        was_compressed, ratio, message = compressor.compress_image(
            original_jpg, 
            compressed_jpg
        )
        print(f"  {original_jpg.name} -> {compressed_jpg.name}: {message}")


def example_batch_compression():
    """Batch compression example."""
    print("\n=== Batch Compression Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample images
        create_sample_images(temp_path)
        
        # Create more test images
        for i in range(3):
            img = Image.new('RGB', (1600, 1200), 
                          color=(50 + i * 50, 100 + i * 30, 150 + i * 20))
            img_path = temp_path / f"batch_test_{i}.jpg"
            img.save(img_path, format='JPEG', quality=95)
        
        # Batch compression
        compressor = ImageCompressor(size_threshold_mb=0.3)
        
        # Get all image files
        image_files = list(temp_path.glob("*.jpg")) + list(temp_path.glob("*.png"))
        
        print(f"\nFound {len(image_files)} image files")
        for img_file in image_files:
            size_mb = img_file.stat().st_size / 1024 / 1024
            print(f"  - {img_file.name}: {size_mb:.2f} MB")
        
        # Execute batch compression
        print(f"\nStarting batch compression (threshold: {compressor.size_threshold_bytes / 1024 / 1024:.1f} MB)...")
        results = compressor.compress_batch(image_files)
        
        # Display results
        print(f"\nCompression results:")
        for path, was_compressed, ratio, message in results:
            status = "✓" if was_compressed else "-"
            filename = Path(path).name
            print(f"  {status} {filename}: {message}")
        
        # Display statistics
        stats = compressor.get_compression_stats(results)
        print(f"\nStatistics:")
        print(f"  - Total files: {stats['total_files']}")
        print(f"  - Compressed: {stats['compressed_files']}")
        print(f"  - Skipped: {stats['skipped_files']}")
        print(f"  - Failed: {stats['failed_files']}")
        if stats['compressed_files'] > 0:
            print(f"  - Average compression ratio: {stats['average_compression_ratio']:.2f}x")
            print(f"  - Maximum compression ratio: {stats['max_compression_ratio']:.2f}x")


def example_batch_with_output_directory():
    """Batch compression to output directory example."""
    print("\n=== Batch Compression to Output Directory Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create input and output directories
        input_dir = temp_path / "input"
        output_dir = temp_path / "compressed"
        input_dir.mkdir()
        
        # Create sample images in input directory
        create_sample_images(input_dir)
        
        # Create subdirectory structure
        subdir = input_dir / "photos"
        subdir.mkdir()
        
        # Create more images in subdirectory
        for i in range(2):
            img = Image.new('RGB', (1400, 1000), color=(200, 100 + i * 50, 150))
            img_path = subdir / f"photo_{i}.jpg"
            img.save(img_path, format='JPEG', quality=98)
        
        # Collect all images
        all_images = []
        for pattern in ["*.jpg", "*.png"]:
            all_images.extend(input_dir.rglob(pattern))
        
        print(f"\nInput directory structure:")
        for img_path in all_images:
            rel_path = img_path.relative_to(input_dir)
            size_mb = img_path.stat().st_size / 1024 / 1024
            print(f"  - {rel_path}: {size_mb:.2f} MB")
        
        # Execute batch compression to output directory
        compressor = ImageCompressor(size_threshold_mb=0.5)
        
        print(f"\nCompressing to output directory: {output_dir}")
        results = compressor.compress_batch(
            all_images,
            output_dir,
            preserve_structure=True  # Preserve directory structure
        )
        
        # Display results
        print(f"\nCompression results:")
        for path, was_compressed, ratio, message in results:
            status = "✓" if was_compressed else "-"
            rel_path = Path(path).relative_to(input_dir)
            print(f"  {status} {rel_path}: {message}")
        
        # Check output directory structure
        print(f"\nOutput directory structure:")
        if output_dir.exists():
            for output_file in output_dir.rglob("*"):
                if output_file.is_file():
                    rel_path = output_file.relative_to(output_dir)
                    size_mb = output_file.stat().st_size / 1024 / 1024
                    print(f"  - {rel_path}: {size_mb:.2f} MB")


def example_custom_compression_settings():
    """Custom compression settings example."""
    print("\n=== Custom Compression Settings Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test image
        large_img = Image.new('RGB', (2000, 1500), color='purple')
        img_path = temp_path / "test_image.jpg"
        large_img.save(img_path, format='JPEG', quality=95)
        
        original_size = img_path.stat().st_size
        print(f"Original image size: {original_size / 1024 / 1024:.2f} MB")
        
        # Test different compression settings
        test_configs = [
            {
                'name': 'High Quality',
                'jpeg_quality': 95,
                'png_compress_level': 6
            },
            {
                'name': 'Balanced',
                'jpeg_quality': 85,
                'png_compress_level': 9
            },
            {
                'name': 'High Compression',
                'jpeg_quality': 70,
                'png_compress_level': 9
            }
        ]
        
        print(f"\nTesting different compression settings:")
        
        for config in test_configs:
            # Copy original file for testing
            test_img_path = temp_path / f"test_{config['name']}.jpg"
            shutil.copy2(img_path, test_img_path)
            
            # Create custom compressor
            compressor = ImageCompressor(
                size_threshold_mb=0.1,  # Low threshold to ensure compression
                jpeg_quality=config['jpeg_quality'],
                png_compress_level=config['png_compress_level']
            )
            
            # Compress
            was_compressed, ratio, message = compressor.compress_image(test_img_path)
            
            new_size = test_img_path.stat().st_size
            compression_ratio = original_size / new_size
            
            print(f"  {config['name']}:")
            print(f"    - JPEG quality: {config['jpeg_quality']}")
            print(f"    - Compressed size: {new_size / 1024 / 1024:.2f} MB")
            print(f"    - Compression ratio: {compression_ratio:.2f}x")
            
            # Check image quality (estimate by file size)
            quality_score = new_size / original_size
            if quality_score > 0.8:
                quality_desc = "High quality"
            elif quality_score > 0.5:
                quality_desc = "Medium quality"
            else:
                quality_desc = "High compression"
            print(f"    - Quality assessment: {quality_desc}")
            print()


def example_format_conversion():
    """Format conversion example."""
    print("\n=== Format Conversion Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a large PNG image (no transparency)
        large_png = Image.new('RGB', (1500, 1000), color='orange')
        # Add some details
        pixels = large_png.load()
        for x in range(large_png.width):
            for y in range(large_png.height):
                if (x + y) % 100 < 10:  # Add stripes
                    pixels[x, y] = (255, 255, 255)
        
        png_path = temp_path / "large_graphic.png"
        large_png.save(png_path, format='PNG')
        
        original_size = png_path.stat().st_size
        print(f"Original PNG size: {original_size / 1024 / 1024:.2f} MB")
        
        # Use aggressive conversion settings
        compressor = ImageCompressor(
            size_threshold_mb=0.1,
            convert_to_jpeg_threshold=0.9,  # Convert if JPEG saves 90%
            jpeg_quality=85
        )
        
        # Check original format
        with Image.open(png_path) as img:
            print(f"Original format: {img.format}, mode: {img.mode}")
        
        # Compress
        was_compressed, ratio, message = compressor.compress_image(png_path)
        
        # Check compressed format
        with Image.open(png_path) as img:
            print(f"Compressed format: {img.format}, mode: {img.mode}")
        
        new_size = png_path.stat().st_size
        print(f"Compressed size: {new_size / 1024 / 1024:.2f} MB")
        print(f"Compression ratio: {ratio:.2f}x")
        print(f"Result: {message}")


def main():
    """Run all image compression examples."""
    print("Nerif Automatic Image Compression Service Examples")
    print("=" * 40)
    print("These examples demonstrate how to use Nerif to automatically compress")
    print("images larger than 1MB using lossless compression strategies")
    print("and intelligent compression method selection")
    print()
    
    # Run all examples
    example_basic_compression()
    example_advanced_compression()
    example_batch_compression()
    example_batch_with_output_directory()
    example_custom_compression_settings()
    example_format_conversion()
    
    print("\n" + "=" * 40)
    print("All examples completed!")
    print("\nCompression Strategy Summary:")
    print("- PNG: Lossless optimization + smart conversion to JPEG (if beneficial)")
    print("- JPEG: Quality optimization + progressive encoding")
    print("- Transparent PNG: Preserve transparency, lossless optimization only")
    print("- Small files: Automatically skipped (below threshold)")
    print("- Batch processing: Supports directory structure preservation")


if __name__ == "__main__":
    main()