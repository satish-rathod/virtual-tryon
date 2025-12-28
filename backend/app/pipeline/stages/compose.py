"""
Composition Stage.

Composes saree parts onto model poses using the AI adapter.
Produces final_view_XX.png for each pose.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from app.ai.adapter import get_adapter
from app.core.config import settings

logger = logging.getLogger(__name__)

# Color mapping for placement
PLACEMENT_COLOR_MAP = """
Mapping Guide:
- Green Area: Place the Main Body of the saree here.
- Yellow Area: Place the Lower Border here.
- Blue Area: Place the Upper Border here.
- Red Area: Place the Pallu here.
"""


def generate_placeholder_pose(
    pose_id: str,
    output_path: Path,
    width: int = 800,
    height: int = 1200,
) -> Path:
    """
    Generate a placeholder pose image for testing.
    
    In production, this would load from assets/model/poses/pose_XX.png.
    For mock implementation, we generate a simple silhouette.
    """
    # Create a simple gradient background with silhouette
    img = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Gradient background
    for y in range(height):
        gray = int(200 + (55 * y / height))
        img[y, :, :3] = [gray, gray, gray]
    
    img[:, :, 3] = 255  # Fully opaque
    
    # Draw a simple body silhouette for the model
    center_x = width // 2
    
    # Head (circle)
    cv2.circle(img, (center_x, 80), 50, (100, 80, 70, 255), -1)
    
    # Body (rectangle)
    body_left = center_x - 100
    body_right = center_x + 100
    cv2.rectangle(img, (body_left, 130), (body_right, 800), (120, 100, 90, 255), -1)
    
    # Arms
    cv2.line(img, (body_left, 200), (body_left - 80, 400), (120, 100, 90, 255), 30)
    cv2.line(img, (body_right, 200), (body_right + 80, 400), (120, 100, 90, 255), 30)
    
    # Legs
    cv2.rectangle(img, (body_left, 800), (center_x - 10, 1150), (120, 100, 90, 255), -1)
    cv2.rectangle(img, (center_x + 10, 800), (body_right, 1150), (120, 100, 90, 255), -1)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    
    return output_path


def get_pose_asset(pose_id: str, saree_id: str) -> Path:
    """
    Get the path to a pose asset image.
    
    Expected format: 
    - Production: assets/poses/{id}.png (e.g. 1.png, 2.png)
    - Fallback: Generate placeholder in saree assets
    """
    from app.core.storage import get_saree_dir
    
    # Clean pose_id to get just the number if it's like "pose_01" -> "1"
    # Or keep as is if it's already "1"
    clean_id = pose_id
    if pose_id.startswith("pose_"):
        try:
            clean_id = str(int(pose_id.split("_")[1]))
        except (ValueError, IndexError):
            clean_id = pose_id

    # Try assets/poses/{clean_id}.png
    assets_path = settings.ASSETS_ROOT / "poses" / f"{clean_id}.png"
    
    if assets_path.exists():
        return assets_path
    
    # Fallback to model directory for backward compatibility or different structure
    model_path = settings.ASSETS_ROOT / "model" / f"{pose_id}_model.png"
    if model_path.exists():
        return model_path
    
    # Generate placeholder in the saree's assets directory
    saree_assets = get_saree_dir(saree_id) / "assets"
    placeholder_path = saree_assets / f"{pose_id}.png"
    
    if not placeholder_path.exists():
        logger.warning(f"Pose asset not found at {assets_path}, generating placeholder")
        generate_placeholder_pose(pose_id, placeholder_path)
    
    return placeholder_path


def get_placement_map(pose_id: str) -> Optional[Path]:
    """
    Get the placement map (layout guide) for a pose.
    
    Expected format: assets/overly/{id}.png
    """
    # Clean pose_id to get just the number
    clean_id = pose_id
    if pose_id.startswith("pose_"):
        try:
            clean_id = str(int(pose_id.split("_")[1]))
        except (ValueError, IndexError):
            clean_id = pose_id

    # Try assets/overly/{clean_id}.png (Note: 'overly' directory name as per user)
    map_path = settings.ASSETS_ROOT / "overly" / f"{clean_id}.png"
    
    if map_path.exists():
        return map_path

    # Fallback/Backward compat
    old_map_path = settings.ASSETS_ROOT / "poses" / f"{pose_id}_map.png"
    if old_map_path.exists():
        return old_map_path
    
    return None


def compose_pose(
    saree_id: str,
    pose_id: str,
    generation_name: str,
    seed: int,
    retry_instruction: Optional[str] = None,
) -> dict:
    """
    Compose the saree onto a single pose.
    
    Args:
        saree_id: Saree identifier
        pose_id: Pose identifier (pose_01, pose_02, etc.)
        generation_name: Generation folder name
        seed: Deterministic seed
        retry_instruction: Optional instruction from previous failure
        
    Returns:
        dict with result status and output path
    """
    from app.core.storage import get_artifact_path, get_generation_dir, get_saree_dir
    
    logger.info(f"Composing pose {pose_id} for saree {saree_id}")
    
    # Get the flattened saree (or use body part for composition)
    saree_path = get_artifact_path(saree_id, "S_flat.png")
    if not saree_path.exists():
        # Fallback to S_clean if S_flat doesn't exist
        saree_path = get_artifact_path(saree_id, "S_clean.png")
    
    if not saree_path.exists():
        raise ValueError(f"No saree image found for {saree_id}")
    
    # Get pose asset
    pose_path = get_pose_asset(pose_id, saree_id)
    
    # Get placement map (optional)
    placement_map = get_placement_map(pose_id)
    
    # Define output path
    # Map pose_XX to view_XX (pose_01 -> view_01)
    view_number = pose_id.split("_")[1]  # "01", "02", etc.
    output_filename = f"final_view_{view_number}.png"
    
    gen_dir = get_generation_dir(saree_id, generation_name)
    output_path = gen_dir / output_filename
    
    # Get AI adapter
    log_dir = get_saree_dir(saree_id)
    adapter = get_adapter(settings.AI_ADAPTER_TYPE, log_dir=log_dir)
    
    # Get textual description if available
    description = ""
    desc_path = get_saree_dir(saree_id) / "parts" / "description.txt"
    if desc_path.exists():
        try:
            description = desc_path.read_text().strip()
            logger.info(f"Loaded textual description for composition: {description[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to read description: {e}")
    
    # Append the placement color map instruction if a map is being used
    if placement_map:
        description += "\n" + PLACEMENT_COLOR_MAP
            
    # Get available parts for higher fidelity
    parts_dir = get_saree_dir(saree_id) / "parts"
    parts_paths = {}
    if parts_dir.exists():
        for part_file in parts_dir.glob("*.png"):
            # Exclude original full files if they happen to be there (though they shouldn't)
            if part_file.name not in ["S_flat.png", "S_clean.png"]:
                parts_paths[part_file.stem] = part_file
        if parts_paths:
            logger.info(f"Found {len(parts_paths)} parts for composition: {list(parts_paths.keys())}")
    
    # Run composition
    try:
        response = adapter.compose(
            saree_path=saree_path,
            pose_path=pose_path,
            output_path=output_path,
            parts_paths=parts_paths,
            placement_map_path=placement_map,
            seed=seed,
            retry_instruction=retry_instruction,
            description=description,
        )
        
        if response.success:
            logger.info(f"Composed {pose_id}: {output_path}")
            return {
                "status": "success",
                "pose_id": pose_id,
                "output_path": str(output_path),
                "seed": response.seed,
                "metadata": response.metadata,
            }
        else:
            logger.error(f"Compose failed for {pose_id}: {response.error}")
            return {
                "status": "failed",
                "pose_id": pose_id,
                "error": response.error,
            }
            
    except Exception as e:
        logger.error(f"Compose stage failed for {pose_id}: {e}")
        return {
            "status": "failed",
            "pose_id": pose_id,
            "error": str(e),
        }


def run_compose_stage(
    saree_id: str,
    job_id: str,
    generation_name: str,
    poses: list[str],
    base_seed: int = 0,
    retry_instructions: Optional[dict[str, str]] = None,
) -> dict:
    """
    Run the compose stage for multiple poses.
    
    Args:
        saree_id: Saree identifier
        job_id: Job identifier (for logging)
        generation_name: Generation folder name
        poses: List of pose IDs to compose
        base_seed: Base seed (combined with pose for per-pose seed)
        retry_instructions: Dict of pose_id -> retry instruction
        
    Returns:
        dict with results for each pose
    """
    logger.info(f"[{job_id}] Running compose stage for {len(poses)} poses")
    
    if retry_instructions is None:
        retry_instructions = {}
    
    results = {}
    
    for pose_id in poses:
        # Compute deterministic seed from base_seed and pose
        import hashlib
        seed_input = f"{saree_id}|{generation_name}|{pose_id}|{base_seed}"
        seed_hash = hashlib.sha256(seed_input.encode()).digest()
        pose_seed = int.from_bytes(seed_hash[:4], byteorder='big')
        
        retry_instruction = retry_instructions.get(pose_id)
        
        result = compose_pose(
            saree_id=saree_id,
            pose_id=pose_id,
            generation_name=generation_name,
            seed=pose_seed,
            retry_instruction=retry_instruction,
        )
        
        results[pose_id] = result
    
    # Summarize results
    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = sum(1 for r in results.values() if r["status"] == "failed")
    
    logger.info(f"[{job_id}] Compose complete: {successful} success, {failed} failed")
    
    return {
        "status": "success" if failed == 0 else "partial" if successful > 0 else "failed",
        "poses": results,
        "successful_count": successful,
        "failed_count": failed,
    }
