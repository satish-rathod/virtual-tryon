"""
Saree Flattening Stage - AI-Powered Flat Layout Generation.

Uses the AI adapter to generate a flat, rectangular representation of the saree
showing all parts (pallu, body, borders) without folds or wrinkles.
Produces S_flat.png from S_clean.png.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageEnhance

from app.ai.adapter import get_adapter
from app.core.config import settings

logger = logging.getLogger(__name__)


def flatten_opencv_fallback(
    input_path: Path,
    output_path: Path,
) -> dict:
    """
    Fallback: Simple copy with enhancement when AI is not available.
    
    Args:
        input_path: Path to S_clean.png
        output_path: Path to save S_flat.png
        
    Returns:
        dict with metadata
    """
    logger.info(f"Flattening (fallback) from {input_path}")
    
    img = Image.open(input_path)
    
    # Apply minor enhancements
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.02)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.05)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=95)
    
    return {
        "method": "opencv_fallback",
        "input_size": Image.open(input_path).size,
        "output_size": img.size,
    }


def run_flatten_stage(saree_id: str, job_id: str, seed: int = 0) -> dict:
    """
    Run the flatten stage for a job.
    
    Uses the AI adapter to generate a flat, rectangular layout of the saree
    showing all parts clearly without folds or wrinkles.
    
    Args:
        saree_id: Saree identifier
        job_id: Job identifier (for logging)
        seed: Deterministic seed for AI adapter
        
    Returns:
        dict with stage result and output path
    """
    from app.core.storage import get_artifact_path, artifact_exists, get_saree_dir
    
    logger.info(f"[{job_id}] Running flatten stage for saree {saree_id}")
    
    # Get input path (S_clean.png)
    input_path = get_artifact_path(saree_id, "S_clean.png")
    if not input_path.exists():
        raise ValueError(f"S_clean.png not found for saree {saree_id}")
    
    # Define output path
    output_path = get_artifact_path(saree_id, "S_flat.png")
    
    # Check if already exists (immutable artifacts)
    if artifact_exists(saree_id, "S_flat.png"):
        logger.info(f"[{job_id}] S_flat.png already exists, skipping flatten")
        return {
            "status": "skipped",
            "output_path": str(output_path),
            "message": "Artifact already exists",
        }
    
    # Get AI adapter
    log_dir = get_saree_dir(saree_id)
    adapter = get_adapter(settings.AI_ADAPTER_TYPE, log_dir=log_dir)
    
    # Run AI-powered flat layout generation
    try:
        logger.info(f"[{job_id}] Using AI adapter ({settings.AI_ADAPTER_TYPE}) for flat layout")
        
        response = adapter.generate_flat_layout(
            image_path=input_path,
            output_path=output_path,
            seed=seed,
        )
        
        if response.success:
            logger.info(f"[{job_id}] AI flatten complete: {output_path}")
            return {
                "status": "success",
                "output_path": str(output_path),
                "metadata": response.metadata,
                "seed": response.seed,
                "method": "ai_adapter",
            }
        else:
            logger.warning(f"[{job_id}] AI flatten failed: {response.error}, using fallback")
            # Fall back to simple copy
            metadata = flatten_opencv_fallback(input_path, output_path)
            return {
                "status": "success",
                "output_path": str(output_path),
                "metadata": metadata,
                "seed": seed,
                "method": "opencv_fallback",
            }
            
    except Exception as e:
        logger.warning(f"[{job_id}] AI flatten exception: {e}, using fallback")
        try:
            # Fall back to simple copy
            metadata = flatten_opencv_fallback(input_path, output_path)
            return {
                "status": "success",
                "output_path": str(output_path),
                "metadata": metadata,
                "seed": seed,
                "method": "opencv_fallback",
            }
        except Exception as e2:
            logger.error(f"[{job_id}] Both AI and fallback flatten failed: {e2}")
            return {
                "status": "failed",
                "error": str(e2),
            }
