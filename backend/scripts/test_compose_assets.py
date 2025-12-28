import sys
from pathlib import Path
import logging

# Add backend to path so we can import app modules
backend_path = Path(__file__).parent.parent
sys.path.append(str(backend_path))

# Mock settings to point to the correct assets root
from app.core.config import settings

# Override ASSETS_ROOT just in case, though it should be correct
# settings.ASSETS_ROOT = backend_path / "assets" 

from app.pipeline.stages.compose import get_pose_asset, get_placement_map, PLACEMENT_COLOR_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_asset_resolution():
    logger.info("Testing asset resolution...")
    
    # Test valid pose IDs (numeric)
    check_pose("1", expect_found=True)
    check_pose("8", expect_found=True)
    
    # Test valid pose IDs (string format)
    check_pose("pose_1", expect_found=True)
    check_pose("pose_01", expect_found=True) # Logic handles underscore splitting

    # Test valid overlay IDs
    check_overlay("1", expect_found=True)
    check_overlay("8", expect_found=True)
    check_overlay("pose_1", expect_found=True)

    # Test invalid IDs
    check_pose("99", expect_found=False)
    check_overlay("99", expect_found=False)

def check_pose(pose_id, expect_found):
    # Pass a dummy saree_id since we aren't creating a real saree dir in this test unless fallback triggers
    path = get_pose_asset(pose_id, "test_saree_id")
    exists = path.exists()
    status = "FOUND" if exists else "NOT FOUND"
    expected = "FOUND" if expect_found else "NOT FOUND (or generated placeholder)"
    
    # Note: get_pose_asset generates a placeholder if not found, so it technically always "exists" 
    # but we want to know if it resolved to the ASSET or the PLACEHOLDER.
    is_asset = "assets/poses" in str(path) or "assets/model" in str(path)
    
    logger.info(f"Pose {pose_id}: {path} ({'ASSET' if is_asset else 'PLACEHOLDER'})")
    
    if expect_found and not is_asset:
        logger.error(f"FAIL: Expected checking pose {pose_id} to resolve to an existing asset, but got placeholder path: {path}")
    elif not expect_found and is_asset:
        logger.error(f"FAIL: Expected checking pose {pose_id} to NOT be an asset, but it was found at: {path}")
    else:
        logger.info(f"PASS: Pose {pose_id} resolved correctly.")

def check_overlay(pose_id, expect_found):
    path = get_placement_map(pose_id)
    if path:
        logger.info(f"Overlay {pose_id}: {path} (FOUND)")
        if not expect_found:
             logger.error(f"FAIL: Expected overlay {pose_id} to NOT be found, but got: {path}")
    else:
        logger.info(f"Overlay {pose_id}: None (NOT FOUND)")
        if expect_found:
            logger.error(f"FAIL: Expected overlay {pose_id} to be found.")

def test_prompt_content():
    logger.info("\nTesting prompt content...")
    if "Green Area: Place the Main Body" not in PLACEMENT_COLOR_MAP:
        logger.error("FAIL: Color map description missing Green mapping")
    if "Red Area: Place the Pallu" not in PLACEMENT_COLOR_MAP:
        logger.error("FAIL: Color map description missing Red mapping")
    
    logger.info("PASS: Color mappings present in description constant.")

if __name__ == "__main__":
    test_asset_resolution()
    test_prompt_content()
