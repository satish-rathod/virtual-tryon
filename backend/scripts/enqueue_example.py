"""
Example script to enqueue a generate job.

Usage:
    python scripts/enqueue_example.py <saree_id> [mode]

Example:
    python scripts/enqueue_example.py abc123 standard
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from redis import Redis
from rq import Queue

from app.core.config import settings
from app.core.storage import load_json, save_json, get_next_generation_number, build_generation_name, generate_job_id
from app.models import GenerationMode, JobStatus, PipelineStage, JobState
from app.pipeline.orchestrator import run_pipeline


def enqueue_job(saree_id: str, mode: str = "standard"):
    """Enqueue a generation job for a saree."""
    
    # Validate mode
    try:
        gen_mode = GenerationMode(mode)
    except ValueError:
        print(f"Invalid mode: {mode}")
        print(f"Valid modes: {[m.value for m in GenerationMode]}")
        return
    
    # Determine poses for the mode
    if gen_mode == GenerationMode.STANDARD:
        poses = settings.STANDARD_POSES.copy()
    elif gen_mode == GenerationMode.EXTEND:
        poses = settings.EXTENDED_POSES.copy()
    else:
        # For retry_failed, load from existing job
        job_data = load_json(saree_id, "job.json")
        if not job_data:
            print("No previous job found for retry_failed mode")
            return
        poses = job_data.get("failed_poses", [])
        if not poses:
            print("No failed poses to retry")
            return
    
    # Generate job ID
    job_id = generate_job_id()
    
    # Determine generation name
    if gen_mode == GenerationMode.RETRY_FAILED:
        job_data = load_json(saree_id, "job.json")
        generation_name = job_data.get("generation_name")
    else:
        gen_num = get_next_generation_number(saree_id)
        generation_name = build_generation_name(gen_num, gen_mode.value)
    
    # Create job state
    from datetime import datetime
    job_state = JobState(
        job_id=job_id,
        saree_id=saree_id,
        mode=gen_mode,
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
    
    # Save job.json
    save_json(saree_id, "job.json", job_state.to_dict())
    print(f"Created job: {job_id}")
    print(f"Generation: {generation_name}")
    print(f"Poses: {poses}")
    
    # Connect to Redis and enqueue
    try:
        redis_conn = Redis.from_url(settings.REDIS_URL)
        queue = Queue(settings.RQ_QUEUE_NAME, connection=redis_conn)
        
        job = queue.enqueue(
            run_pipeline,
            job_id=job_id,
            saree_id=saree_id,
            job_timeout='1h',
        )
        
        print(f"Enqueued job to queue '{settings.RQ_QUEUE_NAME}'")
        print(f"RQ Job ID: {job.id}")
        
    except Exception as e:
        print(f"Failed to enqueue: {e}")
        print("Make sure Redis is running")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    saree_id = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "standard"
    
    enqueue_job(saree_id, mode)


if __name__ == "__main__":
    main()
