import unittest
import tempfile
import base64
import io
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from nerif.model.vision_model_enhanced import VisionModelWithCompression
from nerif.utils import MessageType


class TestVisionModelWithCompression(unittest.TestCase):
    """Test cases for VisionModelWithCompression."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.vision_model = VisionModelWithCompression(
            enable_compression=True,
            compression_threshold_mb=0.01,  # 10KB for testing
            max_image_size=(512, 512),
            jpeg_quality=80
        )
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_image(self, size: tuple, format_name: str = 'JPEG', mode: str = 'RGB') -> Path:
        """Create a test image file."""
        img = Image.new(mode, size, color='red')
        # Add some pattern to make it more realistic
        pixels = img.load()
        for x in range(0, size[0], 10):
            for y in range(size[1]):
                if mode == 'RGBA':
                    pixels[x, y] = (255, 255, 255, 255)
                else:
                    pixels[x, y] = (255, 255, 255)
        
        filename = f"test.{format_name.lower()}"
        filepath = self.temp_dir / filename
        
        if format_name == 'JPEG':
            img.save(filepath, format=format_name, quality=95)
        else:
            img.save(filepath, format=format_name)
        
        return filepath
    
    def create_large_image_bytes(self, size: tuple = (1000, 1000)) -> bytes:
        """Create large image as bytes."""
        img = Image.new('RGB', size, color='blue')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        return buffer.getvalue()
    
    def test_vlm_optimized_settings(self):
        """Test VLM-specific optimization settings."""
        # Test GPT-4 settings
        gpt4_model = VisionModelWithCompression(model="gpt-4-vision-preview")
        gpt4_settings = gpt4_model._get_vlm_optimized_settings()
        self.assertEqual(gpt4_settings["max_size"], (2048, 2048))
        self.assertEqual(gpt4_settings["max_file_size_mb"], 20)
        
        # Test Claude settings
        claude_model = VisionModelWithCompression(model="claude-3-vision")
        claude_settings = claude_model._get_vlm_optimized_settings()
        self.assertEqual(claude_settings["max_size"], (1568, 1568))
        self.assertEqual(claude_settings["max_file_size_mb"], 5)
        
        # Test Gemini settings
        gemini_model = VisionModelWithCompression(model="gemini-pro-vision")
        gemini_settings = gemini_model._get_vlm_optimized_settings()
        self.assertEqual(gemini_settings["max_size"], (3072, 3072))
    
    def test_image_resizing_for_vlm(self):
        """Test image resizing for VLM optimization."""
        # Create vision model with VLM-specific settings (not test settings)
        vlm_model = VisionModelWithCompression(
            model="gpt-4-vision-preview",
            auto_resize_for_vlm=True
        )
        
        # Create a large image
        large_img = Image.new('RGB', (3000, 2000), color='green')
        
        # Test resizing
        resized_img = vlm_model._resize_image_for_vlm(large_img)
        
        # Should be resized to fit within VLM optimized size (2048x2048 for GPT-4V)
        vlm_settings = vlm_model._get_vlm_optimized_settings()
        max_size = vlm_settings["max_size"]
        self.assertLessEqual(resized_img.size[0], max_size[0])
        self.assertLessEqual(resized_img.size[1], max_size[1])
        
        # Test with image already within limits
        small_img = Image.new('RGB', (400, 300), color='yellow')
        not_resized = vlm_model._resize_image_for_vlm(small_img)
        self.assertEqual(not_resized.size, (400, 300))
    
    def test_image_compression(self):
        """Test image compression functionality."""
        # Create large image bytes
        large_image_data = self.create_large_image_bytes((1500, 1500))
        original_size = len(large_image_data)
        
        # Compress the image
        compressed_data = self.vision_model._compress_image_data(large_image_data)
        compressed_size = len(compressed_data)
        
        # Should be smaller (unless compression failed)
        self.assertLessEqual(compressed_size, original_size)
        
        # Verify it's still a valid image
        compressed_img = Image.open(io.BytesIO(compressed_data))
        self.assertIsInstance(compressed_img, Image.Image)
    
    def test_process_image_path(self):
        """Test processing image from file path."""
        # Create test image
        img_path = self.create_test_image((800, 600), 'PNG')
        
        # Process the image
        processed_url = self.vision_model._process_image_path(str(img_path))
        
        # Should return data URL
        self.assertTrue(processed_url.startswith('data:image/'))
        self.assertIn('base64,', processed_url)
        
        # Verify we can decode it back
        header, base64_data = processed_url.split(',', 1)
        decoded_data = base64.b64decode(base64_data)
        decoded_img = Image.open(io.BytesIO(decoded_data))
        self.assertIsInstance(decoded_img, Image.Image)
    
    def test_process_image_base64(self):
        """Test processing base64 image."""
        # Create test image and convert to base64
        img_data = self.create_large_image_bytes((600, 400))
        base64_data = base64.b64encode(img_data).decode('utf-8')
        
        # Process the base64 image
        processed_url = self.vision_model._process_image_base64(base64_data)
        
        # Should return data URL
        self.assertTrue(processed_url.startswith('data:image/'))
        self.assertIn('base64,', processed_url)
    
    @patch('requests.get')
    def test_process_image_url(self, mock_get):
        """Test processing image from URL."""
        # Mock HTTP response
        img_data = self.create_large_image_bytes((700, 500))
        mock_response = Mock()
        mock_response.content = img_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Process the URL
        test_url = "https://example.com/test.jpg"
        processed_url = self.vision_model._process_image_url(test_url)
        
        # Should return data URL
        self.assertTrue(processed_url.startswith('data:image/'))
        mock_get.assert_called_once_with(test_url, timeout=30)
    
    def test_add_image_auto_detection(self):
        """Test automatic image type detection."""
        # Test file path
        img_path = self.create_test_image((400, 300))
        self.vision_model.add_image(str(img_path))
        self.assertEqual(len(self.vision_model.content_cache), 1)
        self.assertEqual(self.vision_model.content_cache[0]["type"], "image_url")
        
        # Reset for next test
        self.vision_model.reset()
        
        # Test URL (mock)
        with patch.object(self.vision_model, '_process_image_url') as mock_process:
            mock_process.return_value = "data:image/jpeg;base64,fake_data"
            self.vision_model.add_image("https://example.com/image.jpg")
            mock_process.assert_called_once()
        
        # Reset for next test
        self.vision_model.reset()
        
        # Test base64
        img_data = self.create_large_image_bytes((300, 200))
        base64_data = f"data:image/jpeg;base64,{base64.b64encode(img_data).decode('utf-8')}"
        self.vision_model.add_image(base64_data)
        self.assertEqual(len(self.vision_model.content_cache), 1)
        
        # Reset for next test
        self.vision_model.reset()
        
        # Test bytes
        self.vision_model.add_image(img_data, source_type="bytes")
        self.assertEqual(len(self.vision_model.content_cache), 1)
    
    def test_add_text(self):
        """Test adding text messages."""
        self.vision_model.add_text("Analyze this image")
        self.assertEqual(len(self.vision_model.content_cache), 1)
        self.assertEqual(self.vision_model.content_cache[0]["type"], "text")
        self.assertEqual(self.vision_model.content_cache[0]["text"], "Analyze this image")
    
    def test_compression_settings_update(self):
        """Test updating compression settings."""
        # Initial settings
        self.assertTrue(self.vision_model.enable_compression)
        self.assertEqual(self.vision_model.jpeg_quality, 80)
        
        # Update settings
        self.vision_model.set_compression_settings(
            enable_compression=False,
            jpeg_quality=95,
            compression_threshold_mb=2.0
        )
        
        # Verify updates
        self.assertFalse(self.vision_model.enable_compression)
        self.assertEqual(self.vision_model.jpeg_quality, 95)
        self.assertEqual(self.vision_model.compression_threshold_mb, 2.0)
        self.assertIsNone(self.vision_model.compressor)
    
    def test_compression_disabled(self):
        """Test behavior when compression is disabled."""
        # Create model with compression disabled
        no_compression_model = VisionModelWithCompression(enable_compression=False)
        
        # Process an image
        img_data = self.create_large_image_bytes((1000, 800))
        processed_data = no_compression_model._compress_image_data(img_data)
        
        # Should return original data
        self.assertEqual(processed_data, img_data)
    
    def test_small_image_skip_compression(self):
        """Test that small images skip compression."""
        # Create small image
        small_img_path = self.create_test_image((100, 100))
        
        # Process it
        with patch.object(self.vision_model, '_compress_image_data', wraps=self.vision_model._compress_image_data) as mock_compress:
            processed_url = self.vision_model._process_image_path(str(small_img_path))
            
            # Should still process but compression should note it's below threshold
            mock_compress.assert_called_once()
    
    def test_transparency_preservation(self):
        """Test that images with transparency are handled correctly."""
        # Create RGBA image with transparency
        rgba_img = Image.new('RGBA', (400, 300), (255, 0, 0, 128))
        rgba_path = self.temp_dir / "transparent.png"
        rgba_img.save(rgba_path, format='PNG')
        
        # Process the image
        processed_url = self.vision_model._process_image_path(str(rgba_path))
        
        # Should maintain PNG format for transparency
        self.assertTrue(processed_url.startswith('data:image/'))
        
        # Decode and verify
        header, base64_data = processed_url.split(',', 1)
        decoded_data = base64.b64decode(base64_data)
        decoded_img = Image.open(io.BytesIO(decoded_data))
        
        # Should preserve transparency capability
        self.assertIn(decoded_img.mode, ['RGBA', 'P'])
    
    @patch('nerif.model.vision_model_enhanced.get_litellm_response')
    def test_chat_functionality(self, mock_get_response):
        """Test the chat functionality with images."""
        # Mock LLM response
        mock_result = Mock()
        mock_result.choices = [Mock(message=Mock(content="I can see a red image with white stripes."))]
        mock_get_response.return_value = mock_result
        
        # Add an image and text
        img_path = self.create_test_image((300, 200))
        self.vision_model.add_image(str(img_path))
        self.vision_model.add_text("What do you see in this image?")
        
        # Chat
        response = self.vision_model.chat()
        
        # Verify response
        self.assertEqual(response, "I can see a red image with white stripes.")
        mock_get_response.assert_called_once()
        
        # Verify the request was made with compressed image
        call_args = mock_get_response.call_args
        messages = call_args[0][0]
        self.assertEqual(len(messages), 2)  # System + user message
        
        user_message = messages[1]
        self.assertEqual(user_message["role"], "user")
        self.assertIsInstance(user_message["content"], list)
    
    def test_get_compression_stats(self):
        """Test compression statistics."""
        stats = self.vision_model.get_compression_stats()
        
        self.assertIn("compression_enabled", stats)
        self.assertIn("compression_threshold_mb", stats)
        self.assertIn("jpeg_quality", stats)
        self.assertIn("max_image_size", stats)
        self.assertIn("vlm_optimized_settings", stats)
        
        self.assertTrue(stats["compression_enabled"])
        self.assertEqual(stats["jpeg_quality"], 80)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test non-existent file
        with self.assertRaises(FileNotFoundError):
            self.vision_model._process_image_path("nonexistent.jpg")
        
        # Test invalid base64
        with self.assertRaises(ValueError):
            self.vision_model._process_image_base64("invalid_base64_data")
        
        # Test unsupported message type
        with self.assertRaises(ValueError):
            self.vision_model.append_message("INVALID_TYPE", "content")
    
    def test_reset_functionality(self):
        """Test conversation reset."""
        # Add some content
        img_path = self.create_test_image((200, 150))
        self.vision_model.add_image(str(img_path))
        self.vision_model.add_text("Test message")
        
        self.assertEqual(len(self.vision_model.content_cache), 2)
        
        # Reset
        self.vision_model.reset()
        
        self.assertEqual(len(self.vision_model.content_cache), 0)
        self.assertEqual(len(self.vision_model.messages), 1)  # Only system message


class TestVisionModelIntegration(unittest.TestCase):
    """Integration tests for VisionModel with compression."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_image_processing(self):
        """Test complete image processing workflow."""
        # Create a realistic large image
        img = Image.new('RGB', (2000, 1500), color='white')
        
        # Add gradient and patterns
        pixels = img.load()
        for x in range(img.width):
            for y in range(img.height):
                r = int(255 * x / img.width)
                g = int(255 * y / img.height)
                b = 128
                pixels[x, y] = (r, g, b)
        
        img_path = self.temp_dir / "large_test.jpg"
        img.save(img_path, format='JPEG', quality=95)
        
        original_size = img_path.stat().st_size
        
        # Process with VisionModel
        vision_model = VisionModelWithCompression(
            enable_compression=True,
            compression_threshold_mb=0.1,
            max_image_size=(1024, 1024),
            jpeg_quality=85
        )
        
        vision_model.add_image(str(img_path))
        
        # Verify image was processed and likely compressed
        self.assertEqual(len(vision_model.content_cache), 1)
        
        # The processed image should be in the cache
        image_content = vision_model.content_cache[0]
        self.assertEqual(image_content["type"], "image_url")
        self.assertTrue(image_content["image_url"]["url"].startswith("data:image/"))


if __name__ == '__main__':
    unittest.main()