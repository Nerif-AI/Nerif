#!/usr/bin/env python3
"""
Example 12: VLM Automatic Image Compression

Demonstrates how to automatically compress images when calling Vision Language Models (VLM),
reducing API call costs and improving transmission efficiency while maintaining image quality.
"""

import tempfile
from pathlib import Path
from PIL import Image
import requests
import base64
import io

from nerif.model.vision_model_enhanced import VisionModelWithCompression
from nerif.utils import MessageType


def create_sample_images(output_dir: Path) -> dict:
    """Create different types of sample images for testing."""
    print("Creating test images...")
    
    images = {}
    
    # Create high-resolution photo (simulate user uploaded large image)
    print("  - Creating high-resolution photo...")
    large_photo = Image.new('RGB', (3000, 2000), color='skyblue')
    pixels = large_photo.load()
    # Add gradient and details
    for x in range(large_photo.width):
        for y in range(large_photo.height):
            r = int(135 + 120 * (x / large_photo.width))
            g = int(206 + 49 * (y / large_photo.height))
            b = int(235 - 20 * ((x + y) / (large_photo.width + large_photo.height)))
            pixels[x, y] = (min(255, r), min(255, g), min(255, b))
    
    photo_path = output_dir / "high_res_photo.jpg"
    large_photo.save(photo_path, format='JPEG', quality=95)
    images['large_photo'] = photo_path
    
    # Create screenshot (usually large)
    print("  - Creating screenshot...")
    screenshot = Image.new('RGB', (1920, 1080), color='white')
    pixels = screenshot.load()
    # Simulate UI elements
    for x in range(0, screenshot.width, 100):
        for y in range(screenshot.height):
            pixels[x, y] = (200, 200, 200)
    for y in range(0, screenshot.height, 50):
        for x in range(screenshot.width):
            pixels[x, y] = (220, 220, 220)
    
    screenshot_path = output_dir / "screenshot.png"
    screenshot.save(screenshot_path, format='PNG')
    images['screenshot'] = screenshot_path
    
    # Create diagram/chart (PNG format)
    print("  - Creating technical diagram...")
    diagram = Image.new('RGB', (1600, 1200), color='white')
    pixels = diagram.load()
    # Add diagram elements
    for x in range(100, 1500, 200):
        for y in range(100, 1100):
            if y % 100 < 20:
                pixels[x, y] = (50, 100, 200)
    
    diagram_path = output_dir / "technical_diagram.png"
    diagram.save(diagram_path, format='PNG')
    images['diagram'] = diagram_path
    
    # Create artistic image (with transparency)
    print("  - Creating artistic image...")
    art = Image.new('RGBA', (1200, 800), (255, 255, 255, 0))
    pixels = art.load()
    # Create semi-transparent shapes
    for x in range(200, 1000):
        for y in range(150, 650):
            if ((x - 600) ** 2 + (y - 400) ** 2) < 200000:
                alpha = max(0, 255 - int(((x - 600) ** 2 + (y - 400) ** 2) / 1000))
                pixels[x, y] = (255, 100, 150, alpha)
    
    art_path = output_dir / "art_with_transparency.png"
    art.save(art_path, format='PNG')
    images['art'] = art_path
    
    # Print image information
    print("\nCreated test images:")
    for name, path in images.items():
        size_mb = path.stat().st_size / 1024 / 1024
        with Image.open(path) as img:
            print(f"  - {name}: {img.size[0]}x{img.size[1]}, {img.format}, {size_mb:.2f} MB")
    
    return images


def example_basic_vlm_compression():
    """Basic VLM image compression example."""
    print("\n=== Basic VLM Image Compression Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        images = create_sample_images(temp_path)
        
        # Create VLM with compression enabled
        vlm = VisionModelWithCompression(
            model="gpt-4-vision-preview",
            enable_compression=True,
            compression_threshold_mb=1.0,  # 1MB threshold
            jpeg_quality=85,
            max_image_size=(2048, 2048)  # GPT-4V recommended size
        )
        
        print(f"\nVLM compression settings:")
        stats = vlm.get_compression_stats()
        print(f"  - Model: {vlm.model}")
        print(f"  - Compression threshold: {stats['compression_threshold_mb']} MB")
        print(f"  - JPEG quality: {stats['jpeg_quality']}")
        print(f"  - Max size: {stats['max_image_size']}")
        print(f"  - VLM optimized settings: {stats['vlm_optimized_settings']}")
        
        # Add large image
        print(f"\nAdding high-resolution photo...")
        original_size = images['large_photo'].stat().st_size
        print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
        
        vlm.add_image(images['large_photo'])
        vlm.add_text("Please describe the content and colors in this image in detail.")
        
        # Simulate API call (actual call requires API key)
        print("Image has been processed and added to VLM conversation")
        print("Compressed image will be sent during actual API call")
        
        # Reset for next example
        vlm.reset()


def example_different_image_sources():
    """Compression example for different image sources."""
    print("\n=== Different Image Sources Compression Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        images = create_sample_images(temp_path)
        
        vlm = VisionModelWithCompression(
            model="claude-3-vision",  # Claude has stricter file size limits
            enable_compression=True,
            compression_threshold_mb=0.5
        )
        
        print(f"\nUsing Claude's optimized settings:")
        claude_settings = vlm._get_vlm_optimized_settings()
        print(f"  - Max size: {claude_settings['max_size']}")
        print(f"  - Max file size: {claude_settings['max_file_size_mb']} MB")
        
        # 1. File path
        print(f"\n1. Adding image from file path:")
        vlm.add_image(images['screenshot'], source_type="path")
        print(f"   Added screenshot: {images['screenshot']}")
        
        # 2. Base64 data
        print(f"\n2. Adding image from Base64 data:")
        with open(images['diagram'], 'rb') as f:
            img_bytes = f.read()
        base64_data = base64.b64encode(img_bytes).decode('utf-8')
        data_url = f"data:image/png;base64,{base64_data}"
        
        original_base64_size = len(base64_data) * 3 / 4  # Estimate original byte size
        vlm.add_image(data_url, source_type="base64")
        print(f"   Added Base64 image (original ~{original_base64_size / 1024 / 1024:.2f} MB)")
        
        # 3. Byte data
        print(f"\n3. Adding image from byte data:")
        with open(images['art'], 'rb') as f:
            art_bytes = f.read()
        
        vlm.add_image(art_bytes, source_type="bytes")
        print(f"   Added artistic image byte data ({len(art_bytes) / 1024 / 1024:.2f} MB)")
        
        # 4. Auto-detection
        print(f"\n4. Auto-detecting image type:")
        vlm.add_image(images['large_photo'])  # Auto-detected as file path
        print(f"   Auto-detected and added photo")
        
        vlm.add_text("Please analyze the characteristics and differences of these images.")
        
        print(f"\nTotal images added: {len([c for c in vlm.content_cache if c['type'] == 'image_url'])}")


def example_compression_strategies():
    """Compression strategy examples for different VLMs."""
    print("\n=== Different VLM Compression Strategy Examples ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        images = create_sample_images(temp_path)
        
        # Test compression strategies for different models
        models = [
            ("gpt-4-vision-preview", "GPT-4 Vision"),
            ("claude-3-sonnet", "Claude 3"),
            ("gemini-pro-vision", "Gemini Pro Vision")
        ]
        
        for model_name, display_name in models:
            print(f"\n--- {display_name} Compression Strategy ---")
            
            vlm = VisionModelWithCompression(
                model=model_name,
                enable_compression=True,
                compression_threshold_mb=0.5
            )
            
            settings = vlm._get_vlm_optimized_settings()
            print(f"Optimized settings: {settings}")
            
            # Test compressing large image
            large_img_path = images['large_photo']
            original_size = large_img_path.stat().st_size
            
            print(f"Processing image: {large_img_path.name} ({original_size / 1024 / 1024:.2f} MB)")
            
            # Process image (without actual API call)
            vlm.add_image(large_img_path)
            
            # Display compression strategy to be applied
            print(f"  - Max size limit: {settings['max_size']}")
            print(f"  - Quality setting: {settings['quality']}")
            print(f"  - File size limit: {settings['max_file_size_mb']} MB")
            
            vlm.reset()


def example_compression_comparison():
    """Compression effect comparison example."""
    print("\n=== Compression Effect Comparison Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        images = create_sample_images(temp_path)
        
        # Create VLMs with different compression levels
        compression_configs = [
            {
                'name': 'No Compression',
                'enable_compression': False,
                'quality': 95
            },
            {
                'name': 'Light Compression',
                'enable_compression': True,
                'compression_threshold_mb': 2.0,
                'quality': 90
            },
            {
                'name': 'Medium Compression',
                'enable_compression': True,
                'compression_threshold_mb': 1.0,
                'quality': 85
            },
            {
                'name': 'Heavy Compression',
                'enable_compression': True,
                'compression_threshold_mb': 0.5,
                'quality': 75
            }
        ]
        
        test_image = images['large_photo']
        original_size = test_image.stat().st_size
        
        print(f"\nTest image: {test_image.name}")
        print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
        print(f"Original dimensions: {Image.open(test_image).size}")
        
        print(f"\nCompression effect comparison:")
        
        for config in compression_configs:
            vlm = VisionModelWithCompression(
                enable_compression=config['enable_compression'],
                compression_threshold_mb=config.get('compression_threshold_mb', 1.0),
                jpeg_quality=config['quality'],
                max_image_size=(1024, 1024)  # Unified size limit
            )
            
            # Manually test compression effect
            if config['enable_compression']:
                with open(test_image, 'rb') as f:
                    original_data = f.read()
                
                compressed_data = vlm._compress_image_data(original_data)
                compression_ratio = len(original_data) / len(compressed_data)
                compressed_size_mb = len(compressed_data) / 1024 / 1024
                
                print(f"  {config['name']:20} - {compressed_size_mb:.2f} MB (compression ratio: {compression_ratio:.2f}x)")
            else:
                print(f"  {config['name']:20} - {original_size / 1024 / 1024:.2f} MB (no compression)")


def example_smart_format_conversion():
    """Smart format conversion example."""
    print("\n=== Smart Format Conversion Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        images = create_sample_images(temp_path)
        
        vlm = VisionModelWithCompression(
            enable_compression=True,
            compression_threshold_mb=0.3,
            jpeg_quality=85
        )
        
        print(f"\nTesting format conversion for different image types:")
        
        test_cases = [
            {
                'name': 'Photo (JPEG)',
                'path': images['large_photo'],
                'description': 'Usually keeps JPEG format'
            },
            {
                'name': 'Screenshot (PNG)',
                'path': images['screenshot'],
                'description': 'May convert to JPEG to reduce size'
            },
            {
                'name': 'Diagram (PNG)',
                'path': images['diagram'],
                'description': 'May convert to JPEG (if no transparency)'
            },
            {
                'name': 'Art (PNG+Transparency)',
                'path': images['art'],
                'description': 'Keeps PNG format to preserve transparency'
            }
        ]
        
        for case in test_cases:
            print(f"\n{case['name']}:")
            print(f"  File: {case['path'].name}")
            
            original_size = case['path'].stat().st_size
            print(f"  Original size: {original_size / 1024 / 1024:.2f} MB")
            
            with Image.open(case['path']) as img:
                print(f"  Original format: {img.format}, mode: {img.mode}")
                has_transparency = img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                print(f"  Transparency: {'Yes' if has_transparency else 'No'}")
            
            # Simulate compression processing
            with open(case['path'], 'rb') as f:
                original_data = f.read()
            
            compressed_data = vlm._compress_image_data(original_data)
            
            # Check compressed format
            try:
                compressed_img = Image.open(io.BytesIO(compressed_data))
                print(f"  Compressed format: {compressed_img.format}")
                print(f"  Compressed size: {len(compressed_data) / 1024 / 1024:.2f} MB")
                compression_ratio = len(original_data) / len(compressed_data)
                print(f"  Compression ratio: {compression_ratio:.2f}x")
            except Exception as e:
                print(f"  Processing failed: {e}")
            
            print(f"  Strategy: {case['description']}")


def example_batch_vlm_processing():
    """Batch VLM image processing example."""
    print("\n=== Batch VLM Image Processing Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        images = create_sample_images(temp_path)
        
        # Create multiple VLM instances for different tasks
        tasks = [
            {
                'name': 'Image Description Task',
                'images': [images['large_photo'], images['art']],
                'prompt': 'Please describe the content and artistic style of these images in detail.',
                'model': 'gpt-4-vision-preview'
            },
            {
                'name': 'Technical Analysis Task',
                'images': [images['screenshot'], images['diagram']],
                'prompt': 'Please analyze the technical features of these interfaces and diagrams.',
                'model': 'claude-3-sonnet'
            }
        ]
        
        for task in tasks:
            print(f"\n--- {task['name']} ---")
            print(f"Model: {task['model']}")
            print(f"Number of images: {len(task['images'])}")
            
            vlm = VisionModelWithCompression(
                model=task['model'],
                enable_compression=True,
                compression_threshold_mb=0.8
            )
            
            total_original_size = 0
            
            # Add all images
            for img_path in task['images']:
                original_size = img_path.stat().st_size
                total_original_size += original_size
                
                print(f"  Adding image: {img_path.name} ({original_size / 1024 / 1024:.2f} MB)")
                vlm.add_image(img_path)
            
            # Add task prompt
            vlm.add_text(task['prompt'])
            
            print(f"  Total original size: {total_original_size / 1024 / 1024:.2f} MB")
            print(f"  Images compressed and ready to send")
            
            # Display VLM cache status
            image_count = len([c for c in vlm.content_cache if c['type'] == 'image_url'])
            text_count = len([c for c in vlm.content_cache if c['type'] == 'text'])
            print(f"  Cache content: {image_count} images, {text_count} text messages")


def main():
    """Run all VLM automatic compression examples."""
    print("Nerif VLM Automatic Image Compression Examples")
    print("=" * 50)
    print("Demonstrates how to automatically compress images when calling Vision Language Models")
    print("Reducing API costs and improving transmission efficiency")
    print()
    
    # Run all examples
    example_basic_vlm_compression()
    example_different_image_sources()
    example_compression_strategies()
    example_compression_comparison()
    example_smart_format_conversion()
    example_batch_vlm_processing()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nVLM Image Compression Feature Summary:")
    print("📸 Auto-detection: Supports file paths, URLs, Base64, byte data")
    print("🗜️  Smart compression: Optimizes compression strategy based on VLM model")
    print("🎯 Model adaptation: Optimized for different models like GPT-4V, Claude, Gemini")
    print("🔄 Format conversion: Intelligently selects best format (JPEG/PNG)")
    print("🛡️  Quality protection: Maintains visual quality while reducing file size")
    print("⚡ Performance boost: Reduces transmission time and API costs")
    print("🎨 Transparency protection: Automatically preserves images with transparency")
    print("\nUsage recommendations:")
    print("- Web image analysis: Compression threshold 0.5-1MB")
    print("- Document processing: Compression threshold 1-2MB") 
    print("- Real-time applications: Enable aggressive compression and size limits")


if __name__ == "__main__":
    main()