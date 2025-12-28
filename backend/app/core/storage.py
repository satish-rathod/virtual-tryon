"""
Storage helpers for managing artifacts on the local filesystem.

Implements the storage layout defined in docs/STORAGE.md:
- Per-saree directories with immutable artifacts
- Canonical path builders for all artifact types
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from app.core.config import settings

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Raised when a storage operation fails."""
    pass


def generate_saree_id() -> str:
    """Generate a unique saree ID."""
    return str(uuid4())


def generate_job_id() -> str:
    """Generate a unique job ID."""
    return str(uuid4())


def get_saree_dir(saree_id: str) -> Path:
    """Get the directory for a specific saree, creating it if needed."""
    path = settings.get_storage_root() / saree_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_generations_dir(saree_id: str) -> Path:
    """Get the generations directory for a saree."""
    path = get_saree_dir(saree_id) / "generations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_generation_dir(saree_id: str, generation_name: str) -> Path:
    """Get a specific generation directory."""
    path = get_generations_dir(saree_id) / generation_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_parts_dir(saree_id: str) -> Path:
    """Get the parts directory for a saree."""
    path = get_saree_dir(saree_id) / "parts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_next_generation_number(saree_id: str) -> int:
    """Get the next generation number for a saree."""
    gen_dir = get_generations_dir(saree_id)
    if not gen_dir.exists():
        return 1
    
    existing = list(gen_dir.iterdir())
    if not existing:
        return 1
    
    # Parse existing generation numbers (gen_01_standard, gen_02_extend, etc.)
    max_num = 0
    for d in existing:
        if d.is_dir() and d.name.startswith("gen_"):
            try:
                num = int(d.name.split("_")[1])
                max_num = max(max_num, num)
            except (IndexError, ValueError):
                continue
    
    return max_num + 1


def build_generation_name(generation_number: int, mode: str) -> str:
    """Build a generation folder name like 'gen_01_standard'."""
    return f"gen_{generation_number:02d}_{mode}"


def save_original(saree_id: str, file_data: bytes, filename: str) -> Path:
    """
    Save the original uploaded image.
    
    Args:
        saree_id: Unique identifier for the saree
        file_data: Raw bytes of the uploaded file
        filename: Original filename to determine extension
        
    Returns:
        Path to the saved file
        
    Raises:
        StorageError: If file already exists or save fails
    """
    saree_dir = get_saree_dir(saree_id)
    
    # Determine extension from original filename
    ext = Path(filename).suffix.lower() or ".jpg"
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"
    
    save_path = saree_dir / f"original{ext}"
    
    if save_path.exists():
        raise StorageError(f"Original already exists for saree {saree_id}")
    
    try:
        save_path.write_bytes(file_data)
        logger.info(f"Saved original image: {save_path}")
        return save_path
    except Exception as e:
        raise StorageError(f"Failed to save original: {e}")


def save_artifact(
    saree_id: str,
    artifact_name: str,
    data: bytes,
    generation_name: Optional[str] = None,
    subdir: Optional[str] = None,
) -> Path:
    """
    Save an artifact file.
    
    Args:
        saree_id: Saree identifier
        artifact_name: Name of the artifact file (e.g., 'S_clean.png')
        data: Raw bytes to save
        generation_name: Optional generation folder (e.g., 'gen_01_standard')
        subdir: Optional subdirectory (e.g., 'parts')
        
    Returns:
        Path to the saved file
        
    Raises:
        StorageError: If artifact already exists (immutability)
    """
    if generation_name:
        base_dir = get_generation_dir(saree_id, generation_name)
    else:
        base_dir = get_saree_dir(saree_id)
    
    if subdir:
        base_dir = base_dir / subdir
        base_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = base_dir / artifact_name
    
    if save_path.exists():
        raise StorageError(f"Artifact already exists: {save_path}")
    
    try:
        save_path.write_bytes(data)
        logger.info(f"Saved artifact: {save_path}")
        return save_path
    except Exception as e:
        raise StorageError(f"Failed to save artifact: {e}")


def save_json(
    saree_id: str,
    filename: str,
    data: dict[str, Any],
    generation_name: Optional[str] = None,
    subdir: Optional[str] = None,
) -> Path:
    """
    Save a JSON file (metrics.json, job.json, etc.).
    
    Args:
        saree_id: Saree identifier
        filename: Name of the JSON file
        data: Dictionary to serialize
        generation_name: Optional generation folder
        subdir: Optional subdirectory
        
    Returns:
        Path to the saved file
    """
    if generation_name:
        base_dir = get_generation_dir(saree_id, generation_name)
    else:
        base_dir = get_saree_dir(saree_id)
    
    if subdir:
        base_dir = base_dir / subdir
        base_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = base_dir / filename
    
    # JSON files can be updated (job.json updates during processing)
    try:
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved JSON: {save_path}")
        return save_path
    except Exception as e:
        raise StorageError(f"Failed to save JSON: {e}")


def load_json(
    saree_id: str,
    filename: str,
    generation_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Load a JSON file.
    
    Returns None if file doesn't exist.
    """
    if generation_name:
        base_dir = get_generation_dir(saree_id, generation_name)
    else:
        base_dir = get_saree_dir(saree_id)
    
    file_path = base_dir / filename
    
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON {file_path}: {e}")
        return None


def get_artifact_path(
    saree_id: str,
    artifact_name: str,
    generation_name: Optional[str] = None,
    subdir: Optional[str] = None,
) -> Path:
    """Get the full path to an artifact."""
    if generation_name:
        base_dir = get_generation_dir(saree_id, generation_name)
    else:
        base_dir = get_saree_dir(saree_id)
    
    if subdir:
        base_dir = base_dir / subdir
    
    return base_dir / artifact_name


def artifact_exists(
    saree_id: str,
    artifact_name: str,
    generation_name: Optional[str] = None,
    subdir: Optional[str] = None,
) -> bool:
    """Check if an artifact exists."""
    path = get_artifact_path(saree_id, artifact_name, generation_name, subdir)
    return path.exists()


def get_original_path(saree_id: str) -> Optional[Path]:
    """Get the path to the original image for a saree."""
    saree_dir = get_saree_dir(saree_id)
    
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        path = saree_dir / f"original{ext}"
        if path.exists():
            return path
    
    return None


def list_sarees() -> list[dict[str, Any]]:
    """
    List all sarees in storage with metadata.
    
    Returns a list of dicts with: saree_id, created_at, thumbnail, generation_count, latest_status
    """
    storage_root = settings.get_storage_root()
    sarees = []
    
    for saree_dir in storage_root.iterdir():
        if not saree_dir.is_dir():
            continue
        
        saree_id = saree_dir.name
        
        # Get creation time from original file
        original = get_original_path(saree_id)
        if original:
            created_at = datetime.fromtimestamp(
                original.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        else:
            created_at = None
        
        # Count generations
        gen_dir = saree_dir / "generations"
        generation_count = 0
        if gen_dir.exists():
            generation_count = len([d for d in gen_dir.iterdir() if d.is_dir()])
        
        # Get latest job status
        job_data = load_json(saree_id, "job.json")
        latest_status = job_data.get("status") if job_data else None
        
        # Thumbnail path (relative)
        thumbnail = f"artifacts/{saree_id}/original.jpg" if original else None
        
        sarees.append({
            "saree_id": saree_id,
            "created_at": created_at,
            "thumbnail": thumbnail,
            "generation_count": generation_count,
            "latest_status": latest_status,
        })
    
    # Sort by creation date, newest first
    sarees.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    
    return sarees


def copy_file(src: Path, dst: Path) -> Path:
    """Copy a file, creating parent directories if needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst
