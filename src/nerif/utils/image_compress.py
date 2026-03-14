"""
Image compression utility for Nerif.
Automatically compresses images larger than 1MB using lossless and lossy techniques.

Uses the Rust-based `nerif_native` backend when available for faster processing,
falling back to PIL/Pillow otherwise.
"""

import io
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image

logger = logging.getLogger(__name__)

# Try to import the Rust native backend
try:
    import nerif_native as _native

    _HAS_NATIVE = True
    logger.debug("nerif_native Rust backend loaded")
except ImportError:
    _native = None
    _HAS_NATIVE = False
    logger.debug("nerif_native not available, using PIL fallback")


class ImageCompressor:
    """
    Image compression utility that automatically compresses images over 1MB.

    Features:
    - Lossless optimization for PNG images
    - Smart JPEG quality adjustment
    - Format conversion when beneficial
    - Transparency detection
    - Batch processing support
    - Uses Rust backend (nerif_native) when available for faster processing
    """

    def __init__(
        self,
        size_threshold_mb: float = 1.0,
        jpeg_quality: int = 85,
        png_compress_level: int = 9,
        convert_to_jpeg_threshold: float = 0.7,
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
        preserve_metadata: bool = True,
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

        if output_path is None:
            output_path = image_path
        else:
            output_path = Path(output_path)

        # Try Rust backend first
        if _HAS_NATIVE:
            try:
                return self._compress_native(image_path, output_path, original_size)
            except Exception as e:
                logger.warning(f"Native compression failed, falling back to PIL: {e}")

        # PIL fallback
        return self._compress_pil(image_path, output_path, original_size, preserve_metadata)

    def _compress_native(
        self,
        image_path: Path,
        output_path: Path,
        original_size: int,
    ) -> Tuple[bool, float, str]:
        """Compress using the Rust nerif_native backend."""
        raw = image_path.read_bytes()

        # Detect if image has transparency to choose format
        has_alpha = _native.has_transparency(raw)
        fmt = "png" if has_alpha else "jpeg"

        compressed = _native.compress_image(
            raw,
            quality=self.jpeg_quality,
            force_format=fmt,
        )

        output_path.write_bytes(compressed)
        new_size = len(compressed)
        ratio = original_size / new_size if new_size > 0 else 1.0

        message = (
            f"Compressed from {self._format_size(original_size)} "
            f"to {self._format_size(new_size)} "
            f"(ratio: {ratio:.2f}x, backend: native)"
        )
        return True, ratio, message

    def _compress_pil(
        self,
        image_path: Path,
        output_path: Path,
        original_size: int,
        preserve_metadata: bool,
    ) -> Tuple[bool, float, str]:
        """Compress using PIL/Pillow."""
        try:
            with Image.open(image_path) as img:
                format_name = img.format
                mode = img.mode

                # Get EXIF data if needed
                exif_data = None
                if preserve_metadata and hasattr(img, "_getexif"):
                    try:
                        exif_data = img.info.get("exif", img._getexif())
                    except Exception:
                        exif_data = None

                # Determine compression strategy
                compressed_img, compressed_format = self._compress_strategy(img, format_name, mode)

                save_kwargs = self._get_save_kwargs(compressed_format, exif_data)
                compressed_img.save(output_path, **save_kwargs)

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

    def _compress_strategy(self, img: Image.Image, format_name: str, mode: str) -> Tuple[Image.Image, str]:
        """Determine the best compression strategy for the image."""
        # For PNG images
        if format_name == "PNG":
            png_buffer = io.BytesIO()
            img.save(png_buffer, format="PNG", optimize=True, compress_level=self.png_compress_level)
            png_size = png_buffer.tell()

            if mode in ["RGB", "L"] or (mode == "RGBA" and not self._has_transparency(img)):
                jpeg_buffer = io.BytesIO()
                if mode == "RGBA":
                    rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3])
                    img = rgb_img
                img.save(jpeg_buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
                jpeg_size = jpeg_buffer.tell()
                if jpeg_size < png_size * self.convert_to_jpeg_threshold:
                    return img, "JPEG"
            return img, "PNG"
        elif format_name in ["JPEG", "JPG"]:
            return img, "JPEG"
        else:
            if mode == "RGBA" or self._has_transparency(img):
                return img, "PNG"
            else:
                if mode not in ["RGB", "L"]:
                    img = img.convert("RGB")
                return img, "JPEG"

    def _get_save_kwargs(self, format_name: str, exif_data: Optional[bytes]) -> dict:
        """Get save keyword arguments for the specified format."""
        kwargs = {"format": format_name, "optimize": True}
        if format_name == "PNG":
            kwargs["compress_level"] = self.png_compress_level
        elif format_name == "JPEG":
            kwargs["quality"] = self.jpeg_quality
            kwargs["progressive"] = True
        if exif_data:
            kwargs["exif"] = exif_data
        return kwargs

    def _has_transparency(self, img: Image.Image) -> bool:
        """Check if an image has actual transparency."""
        if img.mode != "RGBA":
            return False
        alpha = img.split()[-1]
        return alpha.getextrema()[0] < 255

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    def compress_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        preserve_structure: bool = True,
    ) -> List[Tuple[str, bool, float, str]]:
        """Compress multiple images in batch."""
        results = []
        for image_path in image_paths:
            image_path = Path(image_path)
            try:
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    if preserve_structure:
                        try:
                            common_parent = (
                                Path(*image_path.parts[:-2]) if len(image_path.parts) > 2 else image_path.parent
                            )
                            rel_path = image_path.relative_to(common_parent)
                        except ValueError:
                            rel_path = image_path.name
                        output_path = output_dir / rel_path
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        output_path = output_dir / image_path.name
                else:
                    output_path = None

                was_compressed, ratio, message = self.compress_image(image_path, output_path)
                results.append((str(image_path), was_compressed, ratio, message))
            except Exception as e:
                results.append((str(image_path), False, 1.0, f"Error: {e}"))
                logger.error(f"Failed to compress {image_path}: {e}")
        return results

    def get_compression_stats(self, results: List[Tuple[str, bool, float, str]]) -> dict:
        """Generate statistics from batch compression results."""
        total_files = len(results)
        compressed_files = sum(1 for _, was_compressed, _, _ in results if was_compressed)
        failed_files = sum(1 for _, _, ratio, msg in results if "Error" in msg)
        compression_ratios = [ratio for _, was_compressed, ratio, _ in results if was_compressed and ratio > 1]

        stats = {
            "total_files": total_files,
            "compressed_files": compressed_files,
            "failed_files": failed_files,
            "skipped_files": total_files - compressed_files - failed_files,
            "average_compression_ratio": (
                sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0
            ),
            "max_compression_ratio": max(compression_ratios) if compression_ratios else 0,
            "min_compression_ratio": min(compression_ratios) if compression_ratios else 0,
            "backend": "native" if _HAS_NATIVE else "pil",
        }
        return stats


def compress_image_simple(
    image_path: Union[str, Path], size_threshold_mb: float = 1.0, output_path: Optional[Union[str, Path]] = None
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
