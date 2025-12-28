
import logging
import sys
import uuid
import json
import shutil
from pathlib import Path
from datetime import datetime
import time

# Ensure we can import app modules
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from app.core.config import settings
from app.core.storage import (
    get_saree_dir,
    get_generation_dir,
    save_json,
    save_original
)
from app.models import (
    JobState,
    JobStatus,
    PipelineStage,
    GenerationMode
)
from app.pipeline.orchestrator import run_pipeline

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger("debug_pipeline")

def setup_test_job(input_image_path: Path) -> tuple[str, str]:
    """Sets up a test job with a new saree_id and job_id."""
    saree_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    
    logger.info(f"Setting up test job. Saree ID: {saree_id}, Job ID: {job_id}")
    
    # Ensure storage exists
    storage_root = settings.get_storage_root()
    logger.info(f"Storage root: {storage_root}")
    
    # Create saree directory
    saree_dir = get_saree_dir(saree_id)
    logger.info(f"Saree directory: {saree_dir}")
    
    # Save original image
    if not input_image_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
        
    input_bytes = input_image_path.read_bytes()
    save_original(saree_id, input_bytes, input_image_path.name)
    logger.info(f"Saved original image from {input_image_path}")
    
    # Initial Job State
    generation_name = "gen_01_debug"
    poses = settings.STANDARD_POSES
    
    job_state = JobState(
        job_id=job_id,
        saree_id=saree_id,
        mode=GenerationMode.STANDARD,
        status=JobStatus.QUEUED,
        current_stage=PipelineStage.PENDING,
        progress=0,
        generation_name=generation_name,
        poses=poses,
        completed_poses=[],
        failed_poses=[],
        retry_counts={pose: 0 for pose in poses},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    
    save_json(saree_id, "job.json", job_state.to_dict())
    logger.info("Created job.json with initial state")
    
    return job_id, saree_id

def print_artifacts(saree_id: str):
    """Recursively lists all files in the saree directory."""
    saree_dir = get_saree_dir(saree_id)
    logger.info(f"Artifacts in {saree_dir}:")
    
    for path in sorted(saree_dir.rglob("*")):
        if path.is_file():
            rel_path = path.relative_to(saree_dir)
            size = path.stat().st_size
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")
            logger.info(f"  - {rel_path} ({size} bytes, modified: {mtime})")

def main():
    input_image = Path("test_input/101A6689.JPG")
    
    try:
        start_time = time.time()
        job_id, saree_id = setup_test_job(input_image)
        
        logger.info("Cannot auto-run the pipeline in standard configuration because it uses RQ.")
        logger.info("RUNNING PIPELINE SYNCHRONOUSLY NOW...")
        
        # Run pipeline directly
        result = run_pipeline(job_id, saree_id)
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Pipeline execution finished in {duration:.2f} seconds")
        logger.info(f"Result: {json.dumps(result, indent=2)}")
        
        print_artifacts(saree_id)
        
    except Exception as e:
        logger.error(f"Debug run failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
