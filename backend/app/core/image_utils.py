"""
Image utilities for the Saree Virtual Try-On pipeline.

Provides functions for:
- Resizing images for API compatibility
- Converting between PIL and bytes
- Image format handling
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

from PIL import Image

logger = logging.getLogger(__name__)

# Maximum dimension for API calls (prevents timeouts)
MAX_API_DIMENSION = 2048


def resize_for_api(
    image: Union[Path, Image.Image],
    max_size: int = MAX_API_DIMENSION,
    maintain_aspect: bool = True,
) -> Image.Image:
    """
    Resize an image to fit within max_size while preserving aspect ratio.
    
    Args:
        image: Path to image or PIL Image object
        max_size: Maximum dimension (width or height)
        maintain_aspect: If True, preserve aspect ratio
        
    Returns:
        Resized PIL Image
    """
    if isinstance(image, Path):
        img = Image.open(image)
    else:
        img = image
    
    original_size = img.size
    width, height = original_size
    
    # Check if resizing is needed
    if width <= max_size and height <= max_size:
        logger.debug(f"Image {original_size} within limits, no resize needed")
        return img
    
    # Calculate new dimensions
    if maintain_aspect:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
    else:
        new_width = min(width, max_size)
        new_height = min(height, max_size)
    
    # Resize using high-quality resampling
    resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    logger.info(f"Resized image from {original_size} to {resized.size}")
    
    return resized


def image_to_bytes(
    image: Image.Image,
    format: str = "PNG",
    quality: int = 95,
) -> Tuple[bytes, str]:
    """
    Convert a PIL Image to bytes.
    
    Args:
        image: PIL Image object
        format: Output format (PNG, JPEG, WEBP)
        quality: Quality for lossy formats
        
    Returns:
        Tuple of (image bytes, mime type)
    """
    buffer = io.BytesIO()
    
    # Handle format-specific options
    save_kwargs = {}
    if format.upper() in ["JPEG", "JPG"]:
        # JPEG doesn't support alpha, convert to RGB
        if image.mode == "RGBA":
            image = image.convert("RGB")
        save_kwargs["quality"] = quality
        mime_type = "image/jpeg"
    elif format.upper() == "WEBP":
        save_kwargs["quality"] = quality
        mime_type = "image/webp"
    else:
        format = "PNG"
        mime_type = "image/png"
    
    image.save(buffer, format=format, **save_kwargs)
    
    return buffer.getvalue(), mime_type


def bytes_to_image(data: bytes) -> Image.Image:
    """
    Convert bytes to a PIL Image.
    
    Args:
        data: Image bytes
        
    Returns:
        PIL Image object
    """
    return Image.open(io.BytesIO(data))


def load_and_resize(
    image_path: Path,
    max_size: int = MAX_API_DIMENSION,
) -> Tuple[Image.Image, bool]:
    """
    Load an image and resize if needed.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum dimension
        
    Returns:
        Tuple of (PIL Image, was_resized)
    """
    img = Image.open(image_path)
    original_size = img.size
    
    resized_img = resize_for_api(img, max_size)
    was_resized = resized_img.size != original_size
    
    return resized_img, was_resized


def ensure_rgba(image: Image.Image) -> Image.Image:
    """
    Ensure image has RGBA mode (with alpha channel).
    
    Args:
        image: PIL Image
        
    Returns:
        Image in RGBA mode
    """
    if image.mode == "RGBA":
        return image
    
    return image.convert("RGBA")


def ensure_rgb(image: Image.Image) -> Image.Image:
    """
    Ensure image has RGB mode (no alpha channel).
    
    Args:
        image: PIL Image
        
    Returns:
        Image in RGB mode
    """
    if image.mode == "RGB":
        return image
    
    # Handle RGBA by compositing on white background
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # Use alpha as mask
        return background
    
    return image.convert("RGB")


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """
    Get image dimensions without fully loading the image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (width, height)
    """
    with Image.open(image_path) as img:
        return img.size
