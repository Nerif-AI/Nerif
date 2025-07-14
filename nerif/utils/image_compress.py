"""
Image compression utility for Nerif.
Automatically compresses images larger than 1MB using lossless and lossy techniques.
"""

import io
import os
from pathlib import Path
from typing import Optional, Tuple, Union, List
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageCompressor:
    """
    Image compression utility that automatically compresses images over 1MB.
    
    Features:
    - Lossless optimization for PNG images
    - Smart JPEG quality adjustment
    - Format conversion when beneficial
    - Batch processing support
    """
    
    def __init__(
        self,
        size_threshold_mb: float = 1.0,
        jpeg_quality: int = 85,
        png_compress_level: int = 9,
        convert_to_jpeg_threshold: float = 0.7
    ):
        """
        Initialize the image compressor.
        
        Args:
            size_threshold_mb: Size threshold in MB for compression (default: 1.0)
            jpeg_quality: Quality for JPEG compression (1-100, default: 85)
            png_compress_level: PNG compression level (0-9, default: 9)
            convert_to_jpeg_threshold: If PNG->JPEG saves more than this ratio, convert (default: 0.7)
        """
        self.size_threshold_bytes = size_threshold_mb * 1024 * 1024
        self.jpeg_quality = jpeg_quality
        self.png_compress_level = png_compress_level
        self.convert_to_jpeg_threshold = convert_to_jpeg_threshold
    
    def compress_image(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        preserve_metadata: bool = True
    ) -> Tuple[bool, float, str]:
        """
        Compress an image if it exceeds the size threshold.
        
        Args:
            image_path: Path to the input image
            output_path: Path for output image (if None, overwrites original)
            preserve_metadata: Whether to preserve EXIF data
            
        Returns:
            Tuple of (was_compressed, compression_ratio, message)
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Check file size
        original_size = image_path.stat().st_size
        if original_size <= self.size_threshold_bytes:
            return False, 1.0, f"Image size ({self._format_size(original_size)}) is below threshold"
        
        # Open and analyze image
        try:
            with Image.open(image_path) as img:
                format_name = img.format
                mode = img.mode
                
                # Get EXIF data if needed
                exif_data = None
                if preserve_metadata and hasattr(img, '_getexif'):
                    try:
                        exif_data = img.info.get('exif', img._getexif())
                    except:
                        exif_data = None
                
                # Determine compression strategy
                compressed_img, compressed_format = self._compress_strategy(
                    img, format_name, mode
                )
                
                # Save compressed image
                if output_path is None:
                    output_path = image_path
                else:
                    output_path = Path(output_path)
                
                save_kwargs = self._get_save_kwargs(
                    compressed_format, exif_data
                )
                
                compressed_img.save(output_path, **save_kwargs)
                
                # Calculate compression ratio
                new_size = output_path.stat().st_size
                compression_ratio = original_size / new_size
                
                message = (
                    f"Compressed from {self._format_size(original_size)} "
                    f"to {self._format_size(new_size)} "
                    f"(ratio: {compression_ratio:.2f}x)"
                )
                
                return True, compression_ratio, message
                
        except Exception as e:
            logger.error(f"Error compressing image {image_path}: {e}")
            raise
    
    def _compress_strategy(
        self,
        img: Image.Image,
        format_name: str,
        mode: str
    ) -> Tuple[Image.Image, str]:
        """
        Determine the best compression strategy for the image.
        
        Args:
            img: PIL Image object
            format_name: Original format (PNG, JPEG, etc.)
            mode: Image mode (RGB, RGBA, etc.)
            
        Returns:
            Tuple of (compressed_image, target_format)
        """
        # For PNG images
        if format_name == 'PNG':
            # Try lossless PNG optimization first
            png_buffer = io.BytesIO()
            img.save(
                png_buffer,
                format='PNG',
                optimize=True,
                compress_level=self.png_compress_level
            )
            png_size = png_buffer.tell()
            
            # If image has no transparency, try JPEG conversion
            if mode in ['RGB', 'L'] or (mode == 'RGBA' and not self._has_transparency(img)):
                jpeg_buffer = io.BytesIO()
                
                # Convert RGBA to RGB if needed
                if mode == 'RGBA':
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3])
                    img = rgb_img
                
                img.save(
                    jpeg_buffer,
                    format='JPEG',
                    quality=self.jpeg_quality,
                    optimize=True
                )
                jpeg_size = jpeg_buffer.tell()
                
                # If JPEG is significantly smaller, use it
                if jpeg_size < png_size * self.convert_to_jpeg_threshold:
                    return img, 'JPEG'
            
            return img, 'PNG'
        
        # For JPEG images
        elif format_name in ['JPEG', 'JPG']:
            # Progressive JPEG with optimization
            return img, 'JPEG'
        
        # For other formats, convert to appropriate format
        else:
            if mode == 'RGBA' or self._has_transparency(img):
                return img, 'PNG'
            else:
                # Convert to RGB if needed
                if mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                return img, 'JPEG'
    
    def _get_save_kwargs(
        self,
        format_name: str,
        exif_data: Optional[bytes]
    ) -> dict:
        """
        Get save keyword arguments for the specified format.
        
        Args:
            format_name: Target format
            exif_data: EXIF data to preserve
            
        Returns:
            Dictionary of save parameters
        """
        kwargs = {
            'format': format_name,
            'optimize': True
        }
        
        if format_name == 'PNG':
            kwargs['compress_level'] = self.png_compress_level
        elif format_name == 'JPEG':
            kwargs['quality'] = self.jpeg_quality
            kwargs['progressive'] = True
            
        if exif_data:
            kwargs['exif'] = exif_data
            
        return kwargs
    
    def _has_transparency(self, img: Image.Image) -> bool:
        """
        Check if an image has actual transparency.
        
        Args:
            img: PIL Image object
            
        Returns:
            True if image has transparent pixels
        """
        if img.mode != 'RGBA':
            return False
        
        # Check alpha channel
        alpha = img.split()[-1]
        return alpha.getextrema()[0] < 255
    
    def _format_size(self, size_bytes: int) -> str:
        """
        Format file size in human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted size string
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def compress_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        preserve_structure: bool = True
    ) -> List[Tuple[str, bool, float, str]]:
        """
        Compress multiple images in batch.
        
        Args:
            image_paths: List of image paths
            output_dir: Output directory (if None, overwrites originals)
            preserve_structure: Preserve directory structure in output
            
        Returns:
            List of tuples (path, was_compressed, ratio, message)
        """
        results = []
        
        for image_path in image_paths:
            image_path = Path(image_path)
            
            try:
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    if preserve_structure:
                        # Preserve relative path structure
                        try:
                            # Try to get relative path from common parent
                            common_parent = Path(*image_path.parts[:-2]) if len(image_path.parts) > 2 else image_path.parent
                            rel_path = image_path.relative_to(common_parent)
                        except ValueError:
                            # Fallback to just the filename if paths don't share common parent
                            rel_path = image_path.name
                        output_path = output_dir / rel_path
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        output_path = output_dir / image_path.name
                else:
                    output_path = None
                
                was_compressed, ratio, message = self.compress_image(
                    image_path, output_path
                )
                results.append((str(image_path), was_compressed, ratio, message))
                
            except Exception as e:
                results.append((str(image_path), False, 1.0, f"Error: {e}"))
                logger.error(f"Failed to compress {image_path}: {e}")
        
        return results
    
    def get_compression_stats(
        self,
        results: List[Tuple[str, bool, float, str]]
    ) -> dict:
        """
        Generate statistics from batch compression results.
        
        Args:
            results: List of compression results
            
        Returns:
            Dictionary with compression statistics
        """
        total_files = len(results)
        compressed_files = sum(1 for _, was_compressed, _, _ in results if was_compressed)
        failed_files = sum(1 for _, _, ratio, msg in results if "Error" in msg)
        
        compression_ratios = [
            ratio for _, was_compressed, ratio, _ in results 
            if was_compressed and ratio > 1
        ]
        
        stats = {
            'total_files': total_files,
            'compressed_files': compressed_files,
            'failed_files': failed_files,
            'skipped_files': total_files - compressed_files - failed_files,
            'average_compression_ratio': (
                sum(compression_ratios) / len(compression_ratios) 
                if compression_ratios else 0
            ),
            'max_compression_ratio': max(compression_ratios) if compression_ratios else 0,
            'min_compression_ratio': min(compression_ratios) if compression_ratios else 0
        }
        
        return stats


def compress_image_simple(
    image_path: Union[str, Path],
    size_threshold_mb: float = 1.0,
    output_path: Optional[Union[str, Path]] = None
) -> bool:
    """
    Simple function to compress an image if it's larger than the threshold.
    
    Args:
        image_path: Path to the image
        size_threshold_mb: Size threshold in MB (default: 1.0)
        output_path: Output path (if None, overwrites original)
        
    Returns:
        True if image was compressed, False otherwise
    """
    compressor = ImageCompressor(size_threshold_mb=size_threshold_mb)
    was_compressed, _, message = compressor.compress_image(image_path, output_path)
    
    if was_compressed:
        logger.info(message)
    
    return was_compressed