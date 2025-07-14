#!/usr/bin/env python3
"""
Command-line interface for image compression.
"""

import argparse
import sys
from pathlib import Path
from typing import List

from ..utils.image_compress import ImageCompressor


def main():
    """Main CLI entry point for image compression."""
    parser = argparse.ArgumentParser(
        description="Compress images larger than specified threshold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress a single image if larger than 1MB
  nerif-compress image.jpg
  
  # Compress with custom threshold (2MB)
  nerif-compress image.png --threshold 2.0
  
  # Compress to a different file
  nerif-compress large.jpg -o compressed.jpg
  
  # Compress all images in a directory
  nerif-compress images/*.jpg -o compressed/
  
  # Adjust JPEG quality (1-100)
  nerif-compress photo.jpg --jpeg-quality 90
  
  # Force PNG compression level (0-9)
  nerif-compress diagram.png --png-level 9
"""
    )
    
    parser.add_argument(
        'images',
        nargs='+',
        type=str,
        help='Image file(s) to compress'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output path (file for single image, directory for multiple)'
    )
    
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=1.0,
        help='Size threshold in MB (default: 1.0)'
    )
    
    parser.add_argument(
        '--jpeg-quality',
        type=int,
        default=85,
        help='JPEG compression quality 1-100 (default: 85)'
    )
    
    parser.add_argument(
        '--png-level',
        type=int,
        default=9,
        help='PNG compression level 0-9 (default: 9)'
    )
    
    parser.add_argument(
        '--no-preserve-metadata',
        action='store_true',
        help='Do not preserve EXIF metadata'
    )
    
    parser.add_argument(
        '--convert-threshold',
        type=float,
        default=0.7,
        help='PNG to JPEG conversion threshold (default: 0.7)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed compression information'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be compressed without actually doing it'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        parser.error("JPEG quality must be between 1 and 100")
    
    if args.png_level < 0 or args.png_level > 9:
        parser.error("PNG compression level must be between 0 and 9")
    
    # Expand glob patterns and collect image paths
    image_paths = []
    for pattern in args.images:
        path = Path(pattern)
        if path.is_file():
            image_paths.append(path)
        else:
            # Try glob pattern
            matches = list(Path().glob(pattern))
            if not matches:
                print(f"Warning: No files found matching '{pattern}'", file=sys.stderr)
            image_paths.extend(matches)
    
    if not image_paths:
        parser.error("No image files found")
    
    # Initialize compressor
    compressor = ImageCompressor(
        size_threshold_mb=args.threshold,
        jpeg_quality=args.jpeg_quality,
        png_compress_level=args.png_level,
        convert_to_jpeg_threshold=args.convert_threshold
    )
    
    # Determine output mode
    single_file_mode = len(image_paths) == 1 and args.output and not Path(args.output).is_dir()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified\n")
    
    # Process images
    if single_file_mode:
        # Single file mode
        image_path = image_paths[0]
        output_path = Path(args.output) if args.output else None
        
        if args.verbose:
            print(f"Processing: {image_path}")
        
        if not args.dry_run:
            try:
                was_compressed, ratio, message = compressor.compress_image(
                    image_path,
                    output_path,
                    preserve_metadata=not args.no_preserve_metadata
                )
                
                if was_compressed:
                    print(f"✓ {image_path}: {message}")
                else:
                    if args.verbose:
                        print(f"- {image_path}: {message}")
                        
            except Exception as e:
                print(f"✗ {image_path}: Error - {e}", file=sys.stderr)
                sys.exit(1)
        else:
            size = image_path.stat().st_size
            if size > compressor.size_threshold_bytes:
                print(f"Would compress: {image_path} ({size / 1024 / 1024:.2f} MB)")
            else:
                print(f"Would skip: {image_path} ({size / 1024 / 1024:.2f} MB)")
    
    else:
        # Batch mode
        output_dir = Path(args.output) if args.output else None
        
        if args.verbose:
            print(f"Processing {len(image_paths)} images...")
            if output_dir:
                print(f"Output directory: {output_dir}")
            print()
        
        if not args.dry_run:
            results = compressor.compress_batch(
                image_paths,
                output_dir,
                preserve_structure=True
            )
            
            # Display results
            for path, was_compressed, ratio, message in results:
                if was_compressed:
                    print(f"✓ {path}: {message}")
                elif "Error" in message:
                    print(f"✗ {path}: {message}", file=sys.stderr)
                else:
                    if args.verbose:
                        print(f"- {path}: {message}")
            
            # Show statistics
            if len(results) > 1:
                stats = compressor.get_compression_stats(results)
                print(f"\nSummary:")
                print(f"  Total files: {stats['total_files']}")
                print(f"  Compressed: {stats['compressed_files']}")
                print(f"  Skipped: {stats['skipped_files']}")
                print(f"  Failed: {stats['failed_files']}")
                
                if stats['compressed_files'] > 0:
                    print(f"  Average compression ratio: {stats['average_compression_ratio']:.2f}x")
                    
        else:
            # Dry run for batch
            total_size = 0
            would_compress = 0
            
            for image_path in image_paths:
                try:
                    size = image_path.stat().st_size
                    total_size += size
                    
                    if size > compressor.size_threshold_bytes:
                        would_compress += 1
                        print(f"Would compress: {image_path} ({size / 1024 / 1024:.2f} MB)")
                    else:
                        if args.verbose:
                            print(f"Would skip: {image_path} ({size / 1024 / 1024:.2f} MB)")
                except Exception as e:
                    print(f"Error checking {image_path}: {e}", file=sys.stderr)
            
            print(f"\nDry run summary:")
            print(f"  Total files: {len(image_paths)}")
            print(f"  Would compress: {would_compress}")
            print(f"  Total size: {total_size / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    main()