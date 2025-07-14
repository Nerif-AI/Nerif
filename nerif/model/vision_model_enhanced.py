import base64
import io
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Union
from PIL import Image
import requests

from ..utils import (
    LOGGER,
    NERIF_DEFAULT_LLM_MODEL,
    MessageType,
    NerifTokenCounter,
    ImageCompressor,
    get_litellm_response,
)


class VisionModelWithCompression:
    """
    Enhanced Vision Model with automatic image compression for VLM tasks.
    
    This model automatically compresses images before sending them to VLM APIs,
    reducing costs and improving performance while maintaining visual quality.
    
    Features:
    - Automatic image compression for large images
    - Smart compression strategies for different VLM providers
    - Support for various input formats (file path, URL, base64)
    - Configurable compression settings
    - Fallback to original image if compression fails
    """
    
    def __init__(
        self,
        model: str = NERIF_DEFAULT_LLM_MODEL,
        default_prompt: str = "You are a helpful assistant that can analyze images. Describe what you see in detail.",
        temperature: float = 0.0,
        counter: Optional[NerifTokenCounter] = None,
        max_tokens: Optional[int] = None,
        # Image compression settings
        enable_compression: bool = True,
        compression_threshold_mb: float = 1.0,
        max_image_size: tuple = (2048, 2048),
        jpeg_quality: int = 85,
        preserve_aspect_ratio: bool = True,
        auto_resize_for_vlm: bool = True,
    ):
        """
        Initialize VisionModel with compression capabilities.
        
        Args:
            model: The VLM model to use
            default_prompt: Default system prompt
            temperature: Temperature for response generation
            counter: Token counter instance
            max_tokens: Maximum tokens to generate
            enable_compression: Whether to enable automatic compression
            compression_threshold_mb: Compress images larger than this (MB)
            max_image_size: Maximum image dimensions for VLM
            jpeg_quality: JPEG quality for compression (1-100)
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
            auto_resize_for_vlm: Automatically resize images for optimal VLM performance
        """
        self.model = model
        self.temperature = temperature
        self.default_prompt = default_prompt
        self.counter = counter
        self.agent_max_tokens = max_tokens
        
        # Compression settings
        self.enable_compression = enable_compression
        self.compression_threshold_mb = compression_threshold_mb
        self.max_image_size = max_image_size
        self.jpeg_quality = jpeg_quality
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.auto_resize_for_vlm = auto_resize_for_vlm
        
        # Initialize conversation
        self.messages: List[Any] = [
            {"role": "system", "content": default_prompt},
        ]
        self.content_cache = []
        
        # Initialize image compressor
        if self.enable_compression:
            self.compressor = ImageCompressor(
                size_threshold_mb=compression_threshold_mb,
                jpeg_quality=jpeg_quality,
                png_compress_level=9,
                convert_to_jpeg_threshold=0.8  # More aggressive for VLM
            )
        else:
            self.compressor = None
    
    def _get_vlm_optimized_settings(self) -> dict:
        """Get optimized compression settings based on the VLM model."""
        # Different VLM models have different requirements
        if "gpt-4" in self.model.lower():
            return {
                "max_size": (2048, 2048),
                "quality": 85,
                "max_file_size_mb": 20  # GPT-4V limit
            }
        elif "claude" in self.model.lower():
            return {
                "max_size": (1568, 1568),
                "quality": 90,
                "max_file_size_mb": 5   # Claude limit
            }
        elif "gemini" in self.model.lower():
            return {
                "max_size": (3072, 3072),
                "quality": 80,
                "max_file_size_mb": 10  # Gemini limit
            }
        else:
            # Default settings
            return {
                "max_size": (2048, 2048),
                "quality": 85,
                "max_file_size_mb": 10
            }
    
    def _download_image_from_url(self, url: str) -> bytes:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            LOGGER.error(f"Failed to download image from {url}: {e}")
            raise
    
    def _resize_image_for_vlm(self, image: Image.Image) -> Image.Image:
        """Resize image for optimal VLM performance."""
        if not self.auto_resize_for_vlm:
            return image
        
        vlm_settings = self._get_vlm_optimized_settings()
        max_size = vlm_settings["max_size"]
        
        # Check if resizing is needed
        if image.size[0] <= max_size[0] and image.size[1] <= max_size[1]:
            return image
        
        LOGGER.info(f"Resizing image from {image.size} to fit {max_size}")
        
        if self.preserve_aspect_ratio:
            # Calculate the scaling ratio
            ratio = min(max_size[0] / image.size[0], max_size[1] / image.size[1])
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        else:
            new_size = max_size
        
        # Use high-quality resampling
        return image.resize(new_size, Image.Resampling.LANCZOS)
    
    def _compress_image_data(self, image_data: bytes, format_hint: str = None) -> bytes:
        """Compress image data and return compressed bytes."""
        if not self.enable_compression or not self.compressor:
            return image_data
        
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            original_format = image.format or format_hint or 'JPEG'
            
            LOGGER.info(f"Processing image: {image.size}, format: {original_format}")
            
            # Resize if needed for VLM optimization
            image = self._resize_image_for_vlm(image)
            
            # Get VLM-specific settings
            vlm_settings = self._get_vlm_optimized_settings()
            target_quality = vlm_settings["quality"]
            max_file_size_mb = vlm_settings["max_file_size_mb"]
            
            # Check if compression is needed
            current_size_mb = len(image_data) / (1024 * 1024)
            if current_size_mb <= self.compression_threshold_mb:
                LOGGER.info(f"Image size {current_size_mb:.2f}MB is below threshold, skipping compression")
                # Still resize if needed
                if image.size != Image.open(io.BytesIO(image_data)).size:
                    output_buffer = io.BytesIO()
                    # Choose format based on content
                    if image.mode == 'RGBA' or (image.mode == 'P' and 'transparency' in image.info):
                        image.save(output_buffer, format='PNG', optimize=True)
                    else:
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        image.save(output_buffer, format='JPEG', quality=target_quality, optimize=True)
                    return output_buffer.getvalue()
                return image_data
            
            # Determine best format and compress
            output_buffer = io.BytesIO()
            
            # For VLM, prefer JPEG for most cases to reduce file size
            if image.mode == 'RGBA' or (image.mode == 'P' and 'transparency' in image.info):
                # Keep PNG for transparency
                image.save(output_buffer, format='PNG', optimize=True, compress_level=9)
                compressed_data = output_buffer.getvalue()
                LOGGER.info(f"Compressed PNG: {len(image_data)} -> {len(compressed_data)} bytes")
            else:
                # Convert to RGB and use JPEG
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Try different quality levels if file is still too large
                for quality in [target_quality, target_quality - 10, target_quality - 20]:
                    output_buffer = io.BytesIO()
                    image.save(output_buffer, format='JPEG', quality=max(quality, 50), optimize=True, progressive=True)
                    compressed_data = output_buffer.getvalue()
                    compressed_size_mb = len(compressed_data) / (1024 * 1024)
                    
                    LOGGER.info(f"Compressed JPEG (quality {quality}): {len(image_data)} -> {len(compressed_data)} bytes ({compressed_size_mb:.2f}MB)")
                    
                    if compressed_size_mb <= max_file_size_mb or quality <= 50:
                        break
            
            # Verify compression was beneficial
            if len(compressed_data) < len(image_data):
                return compressed_data
            else:
                LOGGER.warning("Compression increased file size, using original")
                return image_data
                
        except Exception as e:
            LOGGER.error(f"Image compression failed: {e}")
            LOGGER.info("Falling back to original image")
            return image_data
    
    def _process_image_path(self, path: str) -> str:
        """Process image from file path with compression."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        
        with open(path, 'rb') as f:
            image_data = f.read()
        
        # Compress if enabled
        compressed_data = self._compress_image_data(image_data, path_obj.suffix[1:])
        
        # Convert to base64
        base64_data = base64.b64encode(compressed_data).decode('utf-8')
        
        # Determine MIME type
        try:
            image = Image.open(io.BytesIO(compressed_data))
            mime_type = f"image/{image.format.lower()}" if image.format else "image/jpeg"
        except:
            mime_type = "image/jpeg"
        
        return f"data:{mime_type};base64,{base64_data}"
    
    def _process_image_url(self, url: str) -> str:
        """Process image from URL with compression."""
        # Download image
        image_data = self._download_image_from_url(url)
        
        # Compress if enabled
        compressed_data = self._compress_image_data(image_data)
        
        # Convert to base64
        base64_data = base64.b64encode(compressed_data).decode('utf-8')
        
        # Determine MIME type
        try:
            image = Image.open(io.BytesIO(compressed_data))
            mime_type = f"image/{image.format.lower()}" if image.format else "image/jpeg"
        except:
            mime_type = "image/jpeg"
        
        return f"data:{mime_type};base64,{base64_data}"
    
    def _process_image_base64(self, base64_content: str) -> str:
        """Process base64 image with compression."""
        # Decode base64
        try:
            # Handle data URL format
            if base64_content.startswith('data:'):
                header, base64_content = base64_content.split(',', 1)
            
            image_data = base64.b64decode(base64_content)
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {e}")
        
        # Compress if enabled
        compressed_data = self._compress_image_data(image_data)
        
        # Convert back to base64
        base64_data = base64.b64encode(compressed_data).decode('utf-8')
        
        # Determine MIME type
        try:
            image = Image.open(io.BytesIO(compressed_data))
            mime_type = f"image/{image.format.lower()}" if image.format else "image/jpeg"
        except:
            mime_type = "image/jpeg"
        
        return f"data:{mime_type};base64,{base64_data}"
    
    def append_message(self, message_type: MessageType, content: str):
        """Add a message to the conversation with automatic image compression."""
        if message_type == MessageType.IMAGE_PATH:
            # Process and compress image from file path
            processed_content = self._process_image_path(content)
            self.content_cache.append({"type": "image_url", "image_url": {"url": processed_content}})
            
        elif message_type == MessageType.IMAGE_URL:
            # Process and compress image from URL
            processed_content = self._process_image_url(content)
            self.content_cache.append({"type": "image_url", "image_url": {"url": processed_content}})
            
        elif message_type == MessageType.IMAGE_BASE64:
            # Process and compress base64 image
            processed_content = self._process_image_base64(content)
            self.content_cache.append({"type": "image_url", "image_url": {"url": processed_content}})
            
        elif message_type == MessageType.TEXT:
            self.content_cache.append({"type": "text", "text": content})
            
        else:
            raise ValueError(f"Message type {message_type} not supported")
    
    def add_image(
        self, 
        image_source: Union[str, Path, bytes], 
        source_type: str = "auto"
    ):
        """
        Convenient method to add an image with automatic type detection.
        
        Args:
            image_source: Image file path, URL, base64 string, or bytes
            source_type: Type of source ("auto", "path", "url", "base64", "bytes")
        """
        if source_type == "auto":
            if isinstance(image_source, (str, Path)):
                image_source = str(image_source)
                if image_source.startswith(('http://', 'https://')):
                    source_type = "url"
                elif image_source.startswith('data:image'):
                    source_type = "base64"
                else:
                    source_type = "path"
            elif isinstance(image_source, bytes):
                source_type = "bytes"
            else:
                raise ValueError(f"Cannot auto-detect type for {type(image_source)}")
        
        if source_type == "path":
            self.append_message(MessageType.IMAGE_PATH, str(image_source))
        elif source_type == "url":
            self.append_message(MessageType.IMAGE_URL, str(image_source))
        elif source_type == "base64":
            self.append_message(MessageType.IMAGE_BASE64, str(image_source))
        elif source_type == "bytes":
            # Convert bytes to base64
            base64_data = base64.b64encode(image_source).decode('utf-8')
            processed_content = self._process_image_base64(base64_data)
            self.content_cache.append({"type": "image_url", "image_url": {"url": processed_content}})
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def add_text(self, text: str):
        """Add text message to the conversation."""
        self.append_message(MessageType.TEXT, text)
    
    def reset(self):
        """Reset the conversation history."""
        self.messages = [{"role": "system", "content": self.default_prompt}]
        self.content_cache = []
    
    def set_max_tokens(self, max_tokens: Optional[int] = None):
        """Set maximum tokens for responses."""
        self.agent_max_tokens = max_tokens
    
    def set_compression_settings(
        self,
        enable_compression: bool = None,
        compression_threshold_mb: float = None,
        jpeg_quality: int = None,
        max_image_size: tuple = None,
    ):
        """Update compression settings."""
        if enable_compression is not None:
            self.enable_compression = enable_compression
        if compression_threshold_mb is not None:
            self.compression_threshold_mb = compression_threshold_mb
        if jpeg_quality is not None:
            self.jpeg_quality = jpeg_quality
        if max_image_size is not None:
            self.max_image_size = max_image_size
        
        # Update compressor
        if self.enable_compression:
            self.compressor = ImageCompressor(
                size_threshold_mb=self.compression_threshold_mb,
                jpeg_quality=self.jpeg_quality,
                png_compress_level=9,
                convert_to_jpeg_threshold=0.8
            )
        else:
            self.compressor = None
    
    def chat(
        self,
        input_content: Optional[List[Any]] = None,
        append: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from the VLM.
        
        Args:
            input_content: Optional direct input content
            append: Whether to append response to conversation
            max_tokens: Maximum tokens for this response
            
        Returns:
            Generated response text
        """
        if input_content is None:
            # Use cached content
            content = self.content_cache
        else:
            # Use provided content
            content = input_content
        
        message = {
            "role": "user",
            "content": content,
        }
        self.messages.append(message)
        
        req_max_tokens = self.agent_max_tokens if max_tokens is None else max_tokens
        
        try:
            result = get_litellm_response(
                self.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=req_max_tokens,
                counter=self.counter,
            )
            
            text_result = result.choices[0].message.content
            
            if append:
                self.content_cache.append({"type": "text", "text": text_result})
            else:
                self.reset()
                
            return text_result
            
        except Exception as e:
            LOGGER.error(f"VLM chat failed: {e}")
            raise
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics for the current session."""
        return {
            "compression_enabled": self.enable_compression,
            "compression_threshold_mb": self.compression_threshold_mb,
            "jpeg_quality": self.jpeg_quality,
            "max_image_size": self.max_image_size,
            "vlm_optimized_settings": self._get_vlm_optimized_settings(),
        }