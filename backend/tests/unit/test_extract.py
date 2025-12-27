"""
Unit tests for the extract_parts stage.

Tests deterministic extraction of saree parts using layout JSON.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.pipeline.stages.extract import extract_parts, RegionMeta


def create_test_image(width: int = 200, height: int = 300) -> Image.Image:
    """Create a synthetic RGB test image with distinct colors for different regions."""
    # Create an array with distinct colors
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill different regions with different colors for testing
    # Pallu region (leftmost 20%): blue
    pallu_end = int(width * 0.2)
    arr[:, :pallu_end] = [50, 100, 200]  # Blue
    
    # Main body (middle): purple
    arr[:, pallu_end:] = [150, 50, 150]  # Purple
    
    # Top border (8% height): gold
    top_border_end = int(height * 0.08)
    arr[:top_border_end, pallu_end:] = [200, 180, 50]  # Gold
    
    # Bottom border (last 8% height): gold
    bottom_border_start = int(height * 0.92)
    arr[bottom_border_start:, pallu_end:] = [200, 180, 50]  # Gold
    
    return Image.fromarray(arr, mode="RGB")


def create_test_layout(regions: dict) -> dict:
    """Create a layout JSON structure."""
    return {
        "layout_id": "test_layout_v1",
        "orientation": {
            "x_axis": "outer_to_inner",
            "y_axis": "top_to_bottom"
        },
        "regions": regions
    }


@pytest.fixture
def temp_dirs(tmp_path: Path):
    """Create temporary directories for test input/output."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    return input_dir, output_dir


class TestExtractParts:
    """Test suite for extract_parts function."""

    def test_basic_extraction_two_regions(self, temp_dirs):
        """Test extraction with a simple two-region layout."""
        input_dir, output_dir = temp_dirs
        
        # Create synthetic 200x300 image
        img = create_test_image(200, 300)
        img_path = input_dir / "sample_flat.png"
        img.save(img_path)
        
        # Create simple layout with two regions
        layout = create_test_layout({
            "region_a": {
                "type": "rectangle",
                "relative_bbox": [0.0, 0.0, 0.5, 0.5],
                "description": "Top-left quarter"
            },
            "region_b": {
                "type": "rectangle",
                "relative_bbox": [0.5, 0.5, 1.0, 1.0],
                "description": "Bottom-right quarter"
            }
        })
        layout_path = input_dir / "test_layout.json"
        with open(layout_path, "w") as f:
            json.dump(layout, f)
        
        # Run extraction
        result = extract_parts(img_path, layout_path, output_dir)
        
        # Assert result structure
        assert isinstance(result, dict)
        assert "region_a" in result
        assert "region_b" in result
        
        # Assert part images created
        parts_dir = output_dir / "parts"
        assert parts_dir.exists()
        assert (parts_dir / "region_a.png").exists()
        assert (parts_dir / "region_b.png").exists()
        
        # Assert parts.json exists and has expected keys
        parts_json_path = parts_dir / "parts.json"
        assert parts_json_path.exists()
        
        with open(parts_json_path) as f:
            parts_data = json.load(f)
        
        assert "regions" in parts_data
        assert "region_a" in parts_data["regions"]
        assert "region_b" in parts_data["regions"]
        
        # Check metadata fields exist
        for region_name in ["region_a", "region_b"]:
            region_data = parts_data["regions"][region_name]
            assert "bbox_rel" in region_data
            assert "bbox_px" in region_data
            assert "width" in region_data
            assert "height" in region_data
            assert "mean_lab" in region_data
            assert "median_lab" in region_data
            assert "lbp_hist_sample" in region_data
            assert len(region_data["lbp_hist_sample"]) == 64

    def test_saree_layout_four_regions(self, temp_dirs):
        """Test extraction using saree-like layout with four regions."""
        input_dir, output_dir = temp_dirs
        
        # Create test image
        img = create_test_image(200, 300)
        img_path = input_dir / "sample_flat.png"
        img.save(img_path)
        
        # Create layout matching saree_flat_layout_v1
        layout = create_test_layout({
            "pallu": {
                "type": "vertical_strip",
                "relative_bbox": [0.00, 0.00, 0.20, 1.00],
                "description": "Leftmost vertical strip"
            },
            "top_border": {
                "type": "horizontal_strip",
                "relative_bbox": [0.20, 0.00, 1.00, 0.08],
                "description": "Upper border"
            },
            "main_body": {
                "type": "rectangle",
                "relative_bbox": [0.20, 0.08, 1.00, 0.92],
                "description": "Main body"
            },
            "bottom_border": {
                "type": "horizontal_strip",
                "relative_bbox": [0.20, 0.92, 1.00, 1.00],
                "description": "Lower border"
            }
        })
        layout_path = input_dir / "saree_layout.json"
        with open(layout_path, "w") as f:
            json.dump(layout, f)
        
        # Run extraction
        result = extract_parts(img_path, layout_path, output_dir)
        
        # Assert all four regions extracted
        expected_regions = ["pallu", "top_border", "main_body", "bottom_border"]
        for region in expected_regions:
            assert region in result
            assert (output_dir / "parts" / f"{region}.png").exists()
        
        # Verify parts.json
        with open(output_dir / "parts" / "parts.json") as f:
            parts_data = json.load(f)
        
        for region in expected_regions:
            assert region in parts_data["regions"]

    def test_region_meta_values(self, temp_dirs):
        """Test that RegionMeta values are computed correctly."""
        input_dir, output_dir = temp_dirs
        
        # Create 100x100 solid color image for predictable results
        arr = np.full((100, 100, 3), [100, 150, 200], dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        img_path = input_dir / "solid.png"
        img.save(img_path)
        
        # Create layout covering full image
        layout = create_test_layout({
            "full": {
                "type": "rectangle",
                "relative_bbox": [0.0, 0.0, 1.0, 1.0],
                "description": "Full image"
            }
        })
        layout_path = input_dir / "full_layout.json"
        with open(layout_path, "w") as f:
            json.dump(layout, f)
        
        # Run extraction
        result = extract_parts(img_path, layout_path, output_dir)
        
        # Check RegionMeta
        meta = result["full"]
        assert meta.name == "full"
        assert meta.width == 100
        assert meta.height == 100
        assert meta.bbox_px == (0, 0, 100, 100)
        assert meta.bbox_rel == (0.0, 0.0, 1.0, 1.0)
        
        # Mean and median LAB should be the same for solid color
        assert len(meta.mean_lab) == 3
        assert len(meta.median_lab) == 3
        # For solid color, mean == median
        assert meta.mean_lab == meta.median_lab
        
        # LBP histogram should exist
        assert len(meta.lbp_hist) == 256

    def test_clamping_out_of_bounds(self, temp_dirs):
        """Test that coordinates outside [0,1] are clamped."""
        input_dir, output_dir = temp_dirs
        
        img = create_test_image(100, 100)
        img_path = input_dir / "test.png"
        img.save(img_path)
        
        # Layout with out-of-bounds values
        layout = create_test_layout({
            "clamped": {
                "type": "rectangle",
                "relative_bbox": [-0.1, -0.2, 1.5, 1.3],  # All values out of bounds
                "description": "Should be clamped to full image"
            }
        })
        layout_path = input_dir / "oob_layout.json"
        with open(layout_path, "w") as f:
            json.dump(layout, f)
        
        # Should not raise error
        result = extract_parts(img_path, layout_path, output_dir)
        
        # Should clamp to full image
        meta = result["clamped"]
        assert meta.width == 100
        assert meta.height == 100

    def test_minimum_crop_size(self, temp_dirs):
        """Test that crops are at least 1x1 pixel."""
        input_dir, output_dir = temp_dirs
        
        img = create_test_image(100, 100)
        img_path = input_dir / "test.png"
        img.save(img_path)
        
        # Layout with zero-area region
        layout = create_test_layout({
            "tiny": {
                "type": "rectangle",
                "relative_bbox": [0.5, 0.5, 0.5, 0.5],  # Zero area
                "description": "Should become 1x1 pixel"
            }
        })
        layout_path = input_dir / "tiny_layout.json"
        with open(layout_path, "w") as f:
            json.dump(layout, f)
        
        # Should not raise error
        result = extract_parts(img_path, layout_path, output_dir)
        
        # Should be at least 1x1
        meta = result["tiny"]
        assert meta.width >= 1
        assert meta.height >= 1

    def test_error_on_missing_image(self, temp_dirs):
        """Test that appropriate error is raised for missing image."""
        input_dir, output_dir = temp_dirs
        
        layout_path = input_dir / "layout.json"
        with open(layout_path, "w") as f:
            json.dump(create_test_layout({"r": {"type": "r", "relative_bbox": [0,0,1,1], "description": ""}}), f)
        
        missing_img = input_dir / "missing.png"
        
        with pytest.raises(FileNotFoundError):
            extract_parts(missing_img, layout_path, output_dir)

    def test_error_on_empty_regions(self, temp_dirs):
        """Test that error is raised if layout has no regions."""
        input_dir, output_dir = temp_dirs
        
        img = create_test_image(100, 100)
        img_path = input_dir / "test.png"
        img.save(img_path)
        
        # Layout with no regions
        layout = {"layout_id": "empty", "regions": {}}
        layout_path = input_dir / "empty_layout.json"
        with open(layout_path, "w") as f:
            json.dump(layout, f)
        
        with pytest.raises(ValueError):
            extract_parts(img_path, layout_path, output_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
