"""
Integration test for full generation flow.

Tests the complete pipeline from upload to generation.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

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


def create_sample_saree_image(output_path: Path):
    """Create a realistic sample saree image for testing."""
    # Create a 600x800 image simulating a saree on light background
    img = np.ones((800, 600, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Main saree body (deep purple/maroon)
    saree_color = [150, 40, 120]  # BGR
    img[100:700, 100:500] = saree_color
    
    # Gold border on edges
    border_color = [50, 180, 220]  # BGR (gold-ish)
    img[100:700, 100:120] = border_color  # Left border
    img[100:700, 480:500] = border_color  # Right border
    img[100:120, 100:500] = border_color  # Top border
    img[680:700, 100:500] = border_color  # Bottom border
    
    # Add some pattern dots to the body
    for y in range(150, 650, 50):
        for x in range(150, 450, 50):
            cv2.circle(img, (x, y), 5, [100, 200, 255], -1)
    
    # Pallu region (top with different pattern)
    img[100:200, 100:500] = [180, 60, 140]  # Slightly different shade
    for x in range(120, 480, 30):
        cv2.circle(img, (x, 150), 8, [50, 180, 220], -1)
    
    cv2.imwrite(str(output_path), img)
    return output_path


def test_full_standard_generation(temp_storage):
    """Test complete standard generation flow."""
    from app.core.storage import generate_saree_id, get_saree_dir
    from app.pipeline.stages.isolate import run_isolate_stage
    from app.pipeline.stages.flatten import run_flatten_stage
    from app.pipeline.stages.extract import run_extract_stage
    from app.pipeline.stages.compose import run_compose_stage
    from app.pipeline.stages.validate import run_validate_stage
    
    # Create saree
    saree_id = generate_saree_id()
    saree_dir = temp_storage / saree_id
    saree_dir.mkdir()
    
    # Create sample image
    create_sample_saree_image(saree_dir / "original.jpg")
    
    generation_name = "gen_01_standard"
    job_id = "test-integration-job"
    base_seed = 12345
    
    # Run all pipeline stages
    
    # Stage 1: Isolate
    isolate_result = run_isolate_stage(saree_id, job_id)
    assert isolate_result["status"] == "success"
    assert (saree_dir / "S_clean.png").exists()
    
    # Stage 2: Flatten
    flatten_result = run_flatten_stage(saree_id, job_id, seed=base_seed)
    assert flatten_result["status"] == "success"
    assert (saree_dir / "S_flat.png").exists()
    
    # Stage 3: Extract
    extract_result = run_extract_stage(saree_id, job_id)
    assert extract_result["status"] == "success"
    assert (saree_dir / "parts" / "parts.json").exists()
    
    # Stage 4: Compose (standard mode = 4 poses)
    poses = ["pose_01", "pose_02", "pose_03", "pose_04"]
    compose_result = run_compose_stage(
        saree_id=saree_id,
        job_id=job_id,
        generation_name=generation_name,
        poses=poses,
        base_seed=base_seed,
    )
    
    assert compose_result["status"] in ["success", "partial"]
    
    gen_dir = saree_dir / "generations" / generation_name
    assert gen_dir.exists()
    
    # Verify all 4 views were created
    for i in range(1, 5):
        view_path = gen_dir / f"final_view_{i:02d}.png"
        assert view_path.exists(), f"Missing {view_path.name}"
    
    # Stage 5: Validate
    validate_result = run_validate_stage(
        saree_id=saree_id,
        job_id=job_id,
        generation_name=generation_name,
        poses=poses,
    )
    
    # Verify metrics.json was created
    metrics_path = gen_dir / "metrics.json"
    assert metrics_path.exists()
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    assert "summary" in metrics
    assert "poses" in metrics
    assert metrics["summary"]["total_poses"] == 4
    
    # Print summary for debugging
    print(f"\nGeneration Summary:")
    print(f"  Total poses: {metrics['summary']['total_poses']}")
    print(f"  Passed: {metrics['summary']['passed']}")
    print(f"  Failed: {metrics['summary']['failed']}")
    print(f"  Pass rate: {metrics['summary']['pass_rate']}")


def test_generation_artifacts_immutable(temp_storage):
    """Test that artifacts are immutable (can't be overwritten)."""
    from app.core.storage import generate_saree_id, save_artifact, StorageError
    
    saree_id = generate_saree_id()
    saree_dir = temp_storage / saree_id
    saree_dir.mkdir()
    
    # Save an artifact
    save_artifact(saree_id, "test_artifact.png", b"test data 1")
    
    # Try to save again - should fail
    with pytest.raises(StorageError):
        save_artifact(saree_id, "test_artifact.png", b"test data 2")


def test_retry_log_created_on_failure(temp_storage):
    """Test that retry log is created when validation fails."""
    from app.pipeline.orchestrator import append_retry_log
    from app.core.storage import generate_saree_id, get_generation_dir
    
    saree_id = generate_saree_id()
    generation_name = "gen_01_standard"
    
    # Create generation directory
    gen_dir = temp_storage / saree_id / "generations" / generation_name
    gen_dir.mkdir(parents=True)
    
    # Append retry log entry
    append_retry_log(
        saree_id=saree_id,
        generation_name=generation_name,
        pose_id="pose_01",
        attempt=1,
        seed=12345,
        prompt="Test prompt",
        failure_reasons=["COLOR_SHIFT", "PATTERN_LOSS"],
        injected_instructions="Fix color and preserve patterns.",
    )
    
    # Verify retry_log.json exists
    retry_log_path = gen_dir / "retry_log.json"
    assert retry_log_path.exists()
    
    with open(retry_log_path) as f:
        log_data = json.load(f)
    
    assert "entries" in log_data
    assert len(log_data["entries"]) == 1
    
    entry = log_data["entries"][0]
    assert entry["pose_id"] == "pose_01"
    assert entry["attempt"] == 1
    assert "COLOR_SHIFT" in entry["failure_reasons"]


def test_storage_layout_matches_spec(temp_storage):
    """Test that storage layout matches docs/STORAGE.md specification."""
    from app.core.storage import (
        generate_saree_id,
        get_saree_dir,
        get_generation_dir,
        get_parts_dir,
    )
    from app.pipeline.stages.isolate import run_isolate_stage
    from app.pipeline.stages.flatten import run_flatten_stage
    from app.pipeline.stages.extract import run_extract_stage
    
    saree_id = generate_saree_id()
    saree_dir = temp_storage / saree_id
    saree_dir.mkdir()
    
    create_sample_saree_image(saree_dir / "original.jpg")
    
    # Run stages
    run_isolate_stage(saree_id, "test-job")
    run_flatten_stage(saree_id, "test-job", seed=42)
    run_extract_stage(saree_id, "test-job")
    
    # Verify structure per STORAGE.md
    # storage/<saree_id>/
    assert saree_dir.exists()
    
    # - original.jpg
    assert (saree_dir / "original.jpg").exists()
    
    # - S_clean.png
    assert (saree_dir / "S_clean.png").exists()
    
    # - S_flat.png
    assert (saree_dir / "S_flat.png").exists()
    
    # - parts/
    parts_dir = saree_dir / "parts"
    assert parts_dir.exists()
    
    # - parts/pallu.png
    assert (parts_dir / "pallu.png").exists()
    
    # - parts/body.png
    assert (parts_dir / "body.png").exists()
    
    # - parts/top_border.png
    assert (parts_dir / "top_border.png").exists()
    
    # - parts/bottom_border.png
    assert (parts_dir / "bottom_border.png").exists()
    
    # - parts/parts.json
    assert (parts_dir / "parts.json").exists()
