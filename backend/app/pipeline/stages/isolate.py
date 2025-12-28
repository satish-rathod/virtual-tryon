"""
Saree Isolation Stage - AI-Powered Background and Person Removal.

Uses the AI adapter to extract only the saree from an image,
removing backgrounds, persons holding the saree, and any other elements.
Produces S_clean.png (RGBA) from the original image.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.ai.adapter import get_adapter
from app.core.config import settings

logger = logging.getLogger(__name__)


def isolate_saree_opencv(
    input_path: Path,
    output_path: Path,
    threshold: int = 240,
) -> dict:
    """
    Fallback: Remove background from saree image using OpenCV.
    
    Uses a deterministic approach:
    1. Convert to grayscale
    2. Apply Otsu's thresholding to find background
    3. Morphological operations to clean edges
    4. Create alpha mask
    
    Args:
        input_path: Path to original image
        output_path: Path to save S_clean.png
        threshold: Background threshold (lighter = background)
        
    Returns:
        dict with metadata about the operation
    """
    logger.info(f"Isolating saree (OpenCV fallback) from {input_path}")
    
    # Load image
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"Could not load image: {input_path}")
    
    original_shape = img.shape
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Otsu's thresholding for automatic threshold selection
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Close small holes
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Open to remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find the largest contour (assume it's the saree)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Keep only the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a clean mask with only the largest contour
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
        
        # Fill any holes in the contour
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        mask = clean_mask
    
    # Apply slight edge feathering for smoother edges
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # Create RGBA image with alpha channel
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), rgba)
    
    # Calculate metadata
    foreground_pixels = np.sum(mask > 127)
    total_pixels = mask.shape[0] * mask.shape[1]
    coverage = foreground_pixels / total_pixels
    
    metadata = {
        "original_shape": original_shape,
        "output_shape": rgba.shape,
        "foreground_coverage": round(coverage, 4),
        "method": "opencv_otsu",
    }
    
    logger.info(f"Isolation complete (OpenCV): {coverage:.2%} foreground coverage")
    
    return metadata


def run_isolate_stage(saree_id: str, job_id: str) -> dict:
    """
    Run the isolate stage for a job.
    
    Uses the AI adapter to extract only the saree from the image,
    removing backgrounds and persons holding the saree.
    
    Args:
        saree_id: Saree identifier
        job_id: Job identifier (for logging)
        
    Returns:
        dict with stage result and output path
    """
    from app.core.storage import get_original_path, get_artifact_path, artifact_exists, get_saree_dir
    
    logger.info(f"[{job_id}] Running isolate stage for saree {saree_id}")
    
    # Get input path
    input_path = get_original_path(saree_id)
    if not input_path:
        raise ValueError(f"Original image not found for saree {saree_id}")
    
    # Define output path
    output_path = get_artifact_path(saree_id, "S_clean.png")
    
    # Check if already exists (immutable artifacts)
    if artifact_exists(saree_id, "S_clean.png"):
        logger.info(f"[{job_id}] S_clean.png already exists, skipping isolate")
        return {
            "status": "skipped",
            "output_path": str(output_path),
            "message": "Artifact already exists",
        }
    
    # Get AI adapter
    log_dir = get_saree_dir(saree_id)
    adapter = get_adapter(settings.AI_ADAPTER_TYPE, log_dir=log_dir)
    
    # Run AI-powered isolation
    try:
        logger.info(f"[{job_id}] Using AI adapter ({settings.AI_ADAPTER_TYPE}) for isolation")
        
        response = adapter.isolate_saree(
            image_path=input_path,
            output_path=output_path,
            seed=0,
        )
        
        if response.success:
            logger.info(f"[{job_id}] AI isolation complete: {output_path}")
            return {
                "status": "success",
                "output_path": str(output_path),
                "metadata": response.metadata,
                "method": "ai_adapter",
            }
        else:
            logger.warning(f"[{job_id}] AI isolation failed: {response.error}, trying OpenCV fallback")
            # Fall back to OpenCV
            metadata = isolate_saree_opencv(input_path, output_path)
            return {
                "status": "success",
                "output_path": str(output_path),
                "metadata": metadata,
                "method": "opencv_fallback",
            }
            
    except Exception as e:
        logger.warning(f"[{job_id}] AI isolation exception: {e}, trying OpenCV fallback")
        try:
            # Fall back to OpenCV
            metadata = isolate_saree_opencv(input_path, output_path)
            return {
                "status": "success",
                "output_path": str(output_path),
                "metadata": metadata,
                "method": "opencv_fallback",
            }
        except Exception as e2:
            logger.error(f"[{job_id}] Both AI and OpenCV isolation failed: {e2}")
            return {
                "status": "failed",
                "error": str(e2),
            }
