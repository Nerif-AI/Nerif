import unittest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import io

from nerif.utils.image_compress import ImageCompressor, compress_image_simple


class TestImageCompressor(unittest.TestCase):
    """Test cases for ImageCompressor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.compressor = ImageCompressor(size_threshold_mb=0.01)  # 10KB for testing
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_image(self, format_name: str, size: tuple, mode: str = 'RGB') -> Path:
        """Create a test image file."""
        img = Image.new(mode, size, color='red')
        filename = f"test.{format_name.lower()}"
        filepath = self.temp_dir / filename
        
        # Save with different quality to control file size
        if format_name == 'JPEG':
            img.save(filepath, format=format_name, quality=95)
        else:
            img.save(filepath, format=format_name)
        
        return filepath
    
    def test_compress_large_jpeg(self):
        """Test compressing a large JPEG image."""
        # Create a large JPEG image
        large_img_path = self.create_test_image('JPEG', (1000, 1000))
        original_size = large_img_path.stat().st_size
        
        # Compress it
        was_compressed, ratio, message = self.compressor.compress_image(large_img_path)
        
        # Verify compression occurred
        self.assertTrue(was_compressed)
        self.assertGreater(ratio, 1.0)
        self.assertIn("Compressed from", message)
        
        # Verify file still exists and is smaller
        new_size = large_img_path.stat().st_size
        self.assertLess(new_size, original_size)
    
    def test_compress_large_png(self):
        """Test compressing a large PNG image."""
        # Create a large PNG image
        large_img_path = self.create_test_image('PNG', (800, 800))
        original_size = large_img_path.stat().st_size
        
        # Compress it
        was_compressed, ratio, message = self.compressor.compress_image(large_img_path)
        
        # Verify compression occurred
        self.assertTrue(was_compressed)
        self.assertGreater(ratio, 1.0)
        
        # Verify file still exists and is smaller or equal
        new_size = large_img_path.stat().st_size
        self.assertLessEqual(new_size, original_size)
    
    def test_skip_small_image(self):
        """Test that small images are skipped."""
        # Create a small image
        small_img_path = self.create_test_image('JPEG', (50, 50))
        
        # Try to compress it
        was_compressed, ratio, message = self.compressor.compress_image(small_img_path)
        
        # Verify it was skipped
        self.assertFalse(was_compressed)
        self.assertEqual(ratio, 1.0)
        self.assertIn("below threshold", message)
    
    def test_compress_with_output_path(self):
        """Test compressing to a different output path."""
        # Create a large image
        large_img_path = self.create_test_image('JPEG', (1000, 1000))
        output_path = self.temp_dir / "compressed.jpg"
        
        # Compress to different path
        was_compressed, ratio, message = self.compressor.compress_image(
            large_img_path, output_path
        )
        
        # Verify compression occurred
        self.assertTrue(was_compressed)
        self.assertTrue(output_path.exists())
        
        # Original should be unchanged
        with Image.open(large_img_path) as img1, Image.open(output_path) as img2:
            self.assertEqual(img1.size, img2.size)
    
    def test_png_with_transparency(self):
        """Test PNG image with transparency."""
        # Create PNG with transparency
        img = Image.new('RGBA', (500, 500), (255, 0, 0, 128))  # Semi-transparent red
        png_path = self.temp_dir / "transparent.png"
        img.save(png_path, format='PNG')
        
        # Compress it
        was_compressed, ratio, message = self.compressor.compress_image(png_path)
        
        # Should be compressed but stay as PNG
        self.assertTrue(was_compressed)
        
        # Verify it's still PNG and has transparency
        with Image.open(png_path) as compressed_img:
            self.assertEqual(compressed_img.format, 'PNG')
            self.assertEqual(compressed_img.mode, 'RGBA')
    
    def test_png_to_jpeg_conversion(self):
        """Test PNG to JPEG conversion when beneficial."""
        # Create a PNG without transparency
        img = Image.new('RGB', (800, 800), color='blue')
        png_path = self.temp_dir / "no_transparency.png"
        img.save(png_path, format='PNG')
        
        # Use a compressor that favors JPEG conversion
        compressor = ImageCompressor(
            size_threshold_mb=0.01,
            convert_to_jpeg_threshold=0.9  # Very aggressive conversion
        )
        
        # Compress it
        was_compressed, ratio, message = compressor.compress_image(png_path)
        
        # Should be compressed
        self.assertTrue(was_compressed)
    
    def test_batch_compression(self):
        """Test batch compression functionality."""
        # Create multiple large images
        image_paths = []
        for i in range(3):
            path = self.create_test_image('JPEG', (800, 800))
            # Rename to unique names
            new_path = self.temp_dir / f"test_{i}.jpg"
            path.rename(new_path)
            image_paths.append(new_path)
        
        # Add one small image that should be skipped
        small_path = self.create_test_image('JPEG', (50, 50))
        small_path = small_path.rename(self.temp_dir / "small.jpg")
        image_paths.append(small_path)
        
        # Compress batch
        results = self.compressor.compress_batch(image_paths)
        
        # Verify results
        self.assertEqual(len(results), 4)
        
        # First 3 should be compressed
        for i in range(3):
            path, was_compressed, ratio, message = results[i]
            self.assertTrue(was_compressed)
            self.assertGreater(ratio, 1.0)
        
        # Last one should be skipped
        path, was_compressed, ratio, message = results[3]
        self.assertFalse(was_compressed)
        self.assertEqual(ratio, 1.0)
    
    def test_batch_with_output_directory(self):
        """Test batch compression with output directory."""
        # Create test images
        image_paths = []
        for i in range(2):
            path = self.create_test_image('PNG', (600, 600))
            new_path = self.temp_dir / f"input_{i}.png"
            path.rename(new_path)
            image_paths.append(new_path)
        
        # Create output directory
        output_dir = self.temp_dir / "compressed"
        
        # Compress batch
        results = self.compressor.compress_batch(image_paths, output_dir)
        
        # Verify output files exist
        self.assertTrue(output_dir.exists())
        for i in range(2):
            output_file = output_dir / f"input_{i}.png"
            self.assertTrue(output_file.exists())
    
    def test_compression_stats(self):
        """Test compression statistics generation."""
        # Mock results
        results = [
            ("file1.jpg", True, 2.5, "Compressed"),
            ("file2.png", True, 1.8, "Compressed"),
            ("file3.jpg", False, 1.0, "Below threshold"),
            ("file4.png", False, 1.0, "Error: Test error")
        ]
        
        stats = self.compressor.get_compression_stats(results)
        
        self.assertEqual(stats['total_files'], 4)
        self.assertEqual(stats['compressed_files'], 2)
        self.assertEqual(stats['failed_files'], 1)
        self.assertEqual(stats['skipped_files'], 1)
        self.assertAlmostEqual(stats['average_compression_ratio'], 2.15, places=2)
        self.assertEqual(stats['max_compression_ratio'], 2.5)
        self.assertEqual(stats['min_compression_ratio'], 1.8)
    
    def test_has_transparency(self):
        """Test transparency detection."""
        # Image with full transparency
        rgba_transparent = Image.new('RGBA', (100, 100), (255, 0, 0, 0))
        self.assertTrue(self.compressor._has_transparency(rgba_transparent))
        
        # Image with partial transparency
        rgba_partial = Image.new('RGBA', (100, 100), (255, 0, 0, 128))
        self.assertTrue(self.compressor._has_transparency(rgba_partial))
        
        # Image without transparency (fully opaque)
        rgba_opaque = Image.new('RGBA', (100, 100), (255, 0, 0, 255))
        self.assertFalse(self.compressor._has_transparency(rgba_opaque))
        
        # RGB image (no alpha channel)
        rgb_image = Image.new('RGB', (100, 100), (255, 0, 0))
        self.assertFalse(self.compressor._has_transparency(rgb_image))
    
    def test_format_size(self):
        """Test size formatting function."""
        self.assertEqual(self.compressor._format_size(512), "512.00 B")
        self.assertEqual(self.compressor._format_size(1024), "1.00 KB")
        self.assertEqual(self.compressor._format_size(1024 * 1024), "1.00 MB")
        self.assertEqual(self.compressor._format_size(1024 * 1024 * 1024), "1.00 GB")
    
    def test_file_not_found_error(self):
        """Test error handling for non-existent files."""
        non_existent_path = self.temp_dir / "does_not_exist.jpg"
        
        with self.assertRaises(FileNotFoundError):
            self.compressor.compress_image(non_existent_path)
    
    def test_simple_compress_function(self):
        """Test the simple compress function."""
        # Create a large image
        large_img_path = self.create_test_image('JPEG', (1000, 1000))
        
        # Use simple function
        was_compressed = compress_image_simple(large_img_path, size_threshold_mb=0.01)
        
        self.assertTrue(was_compressed)
    
    def test_different_image_modes(self):
        """Test compression with different image modes."""
        # Test grayscale image
        gray_img = Image.new('L', (500, 500), color=128)
        gray_path = self.temp_dir / "gray.jpg"
        gray_img.save(gray_path, format='JPEG')
        
        was_compressed, ratio, message = self.compressor.compress_image(gray_path)
        self.assertTrue(was_compressed)
        
        # Test palette image
        palette_img = Image.new('P', (500, 500))
        palette_path = self.temp_dir / "palette.png"
        palette_img.save(palette_path, format='PNG')
        
        was_compressed, ratio, message = self.compressor.compress_image(palette_path)
        # May or may not be compressed depending on size, but should not error


class TestImageCompressionIntegration(unittest.TestCase):
    """Integration tests for image compression."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_real_image_compression(self):
        """Test compression with a real-looking image."""
        # Create a more realistic test image with gradient
        img = Image.new('RGB', (1920, 1080), color='white')
        
        # Add some content to make it more realistic
        pixels = img.load()
        for x in range(img.width):
            for y in range(img.height):
                r = int(255 * x / img.width)
                g = int(255 * y / img.height)
                b = 128
                pixels[x, y] = (r, g, b)
        
        img_path = self.temp_dir / "realistic.jpg"
        img.save(img_path, format='JPEG', quality=95)
        
        original_size = img_path.stat().st_size
        
        # Compress with realistic settings
        compressor = ImageCompressor(
            size_threshold_mb=0.1,  # 100KB threshold
            jpeg_quality=85
        )
        
        was_compressed, ratio, message = compressor.compress_image(img_path)
        
        if original_size > compressor.size_threshold_bytes:
            self.assertTrue(was_compressed)
            new_size = img_path.stat().st_size
            self.assertLess(new_size, original_size)
            
            # Verify image is still valid
            with Image.open(img_path) as compressed_img:
                self.assertEqual(compressed_img.size, (1920, 1080))


if __name__ == '__main__':
    unittest.main()