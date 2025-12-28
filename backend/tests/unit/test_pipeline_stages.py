"""
Unit tests for pipeline stages.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings


@pytest.fixture
def temp_storage(tmp_path):
    """Create a temporary storage directory."""
    storage_path = tmp_path / "storage"
    storage_path.mkdir()
    
    with patch.object(settings, 'STORAGE_ROOT', storage_path):
        yield storage_path


@pytest.fixture
def sample_saree_setup(temp_storage):
    """Set up a sample saree with original image."""
    saree_id = "test-saree-123"
    saree_dir = temp_storage / saree_id
    saree_dir.mkdir()
    
    # Create a sample RGB image (simulating a saree on white background)
    img = np.zeros((400, 300, 3), dtype=np.uint8)
    img[:, :] = [255, 255, 255]  # White background
    
    # Add a colored rectangle (simulating saree)
    img[50:350, 50:250] = [100, 50, 150]  # Purple saree
    
    # Add a border pattern
    img[50:350, 50:60] = [200, 180, 50]  # Gold border
    img[50:350, 240:250] = [200, 180, 50]  # Gold border
    
    cv2.imwrite(str(saree_dir / "original.jpg"), img)
    
    return saree_id


def test_isolate_stage(temp_storage, sample_saree_setup):
    """Test the isolate (background removal) stage."""
    from app.pipeline.stages.isolate import run_isolate_stage
    
    saree_id = sample_saree_setup
    result = run_isolate_stage(saree_id, "test-job-1")
    
    assert result["status"] == "success"
    
    # Verify S_clean.png was created
    s_clean_path = temp_storage / saree_id / "S_clean.png"
    assert s_clean_path.exists()
    
    # Verify it has an alpha channel
    img = cv2.imread(str(s_clean_path), cv2.IMREAD_UNCHANGED)
    assert img.shape[2] == 4  # BGRA


def test_isolate_stage_idempotent(temp_storage, sample_saree_setup):
    """Test that isolate stage is idempotent (skips if exists)."""
    from app.pipeline.stages.isolate import run_isolate_stage
    
    saree_id = sample_saree_setup
    
    # Run first time
    result1 = run_isolate_stage(saree_id, "test-job-1")
    assert result1["status"] == "success"
    
    # Run second time
    result2 = run_isolate_stage(saree_id, "test-job-2")
    assert result2["status"] == "skipped"


def test_flatten_stage(temp_storage, sample_saree_setup):
    """Test the flatten stage."""
    from app.pipeline.stages.isolate import run_isolate_stage
    from app.pipeline.stages.flatten import run_flatten_stage
    
    saree_id = sample_saree_setup
    
    # First run isolate to create S_clean.png
    run_isolate_stage(saree_id, "test-job-1")
    
    # Then run flatten
    result = run_flatten_stage(saree_id, "test-job-1", seed=42)
    
    assert result["status"] == "success"
    assert result["seed"] == 42
    
    # Verify S_flat.png was created
    s_flat_path = temp_storage / saree_id / "S_flat.png"
    assert s_flat_path.exists()


def test_extract_stage(temp_storage, sample_saree_setup):
    """Test the extract stage."""
    from app.pipeline.stages.isolate import run_isolate_stage
    from app.pipeline.stages.flatten import run_flatten_stage
    from app.pipeline.stages.extract import run_extract_stage
    
    saree_id = sample_saree_setup
    
    # Run prerequisites
    run_isolate_stage(saree_id, "test-job-1")
    run_flatten_stage(saree_id, "test-job-1", seed=42)
    
    # Run extract
    result = run_extract_stage(saree_id, "test-job-1")
    
    assert result["status"] == "success"
    
    # Verify parts were created
    parts_dir = temp_storage / saree_id / "parts"
    assert parts_dir.exists()
    assert (parts_dir / "body.png").exists()
    assert (parts_dir / "pallu.png").exists()
    assert (parts_dir / "parts.json").exists()
    
    # Verify parts.json content
    with open(parts_dir / "parts.json") as f:
        parts_data = json.load(f)
    
    assert "body" in parts_data
    assert "width" in parts_data["body"]
    assert "height" in parts_data["body"]


def test_compose_stage(temp_storage, sample_saree_setup):
    """Test the compose stage."""
    from app.pipeline.stages.isolate import run_isolate_stage
    from app.pipeline.stages.flatten import run_flatten_stage
    from app.pipeline.stages.compose import run_compose_stage
    from app.core.storage import get_generation_dir
    
    saree_id = sample_saree_setup
    generation_name = "gen_01_standard"
    
    # Run prerequisites
    run_isolate_stage(saree_id, "test-job-1")
    run_flatten_stage(saree_id, "test-job-1", seed=42)
    
    # Run compose for a single pose
    result = run_compose_stage(
        saree_id=saree_id,
        job_id="test-job-1",
        generation_name=generation_name,
        poses=["pose_01"],
        base_seed=42,
    )
    
    assert result["status"] in ["success", "partial"]
    assert "pose_01" in result["poses"]
    
    # Verify output was created
    gen_dir = temp_storage / saree_id / "generations" / generation_name
    assert gen_dir.exists()
    assert (gen_dir / "final_view_01.png").exists()


def test_deterministic_seed(temp_storage, sample_saree_setup):
    """Test that operations are deterministic with same seed."""
    from app.pipeline.stages.isolate import run_isolate_stage
    from app.pipeline.stages.flatten import run_flatten_stage
    
    saree_id = sample_saree_setup
    
    # Run isolate
    run_isolate_stage(saree_id, "test-job-1")
    
    # Run flatten twice with same seed, should produce same result
    run_flatten_stage(saree_id, "test-job-1", seed=42)
    
    # Read first result
    s_flat_path = temp_storage / saree_id / "S_flat.png"
    img1 = cv2.imread(str(s_flat_path))
    
    # Delete and re-run
    s_flat_path.unlink()
    run_flatten_stage(saree_id, "test-job-2", seed=42)
    
    img2 = cv2.imread(str(s_flat_path))
    
    # Should be identical
    np.testing.assert_array_equal(img1, img2)
