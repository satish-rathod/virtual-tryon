"""
Generate API endpoint.

POST /api/generate - Start generation pipeline for a saree.
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from redis import Redis
from rq import Queue

from app.core.config import settings
from app.core.storage import (
    generate_job_id,
    get_next_generation_number,
    get_original_path,
    build_generation_name,
    save_json,
)
from app.models import (
    GenerateRequest,
    GenerateResponse,
    GenerationMode,
    JobState,
    JobStatus,
    PipelineStage,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["generate"])


def get_poses_for_mode(mode: GenerationMode, failed_poses: Optional[List[str]] = None, gen_num: int = 1) -> List[str]:
    """
    Get the list of poses for a given generation mode.
    
    Args:
        mode: The generation mode
        failed_poses: For retry_failed mode, the list of poses to retry
        gen_num: The generation number (used for cycling in extend mode)
        
    Returns:
        List of pose identifiers
    """
    if mode == GenerationMode.STANDARD:
        return settings.STANDARD_POSES.copy()
    elif mode == GenerationMode.EXTEND:
        # Cycle logic:
        # Gen 2 (Extend) -> Poses 5-8 (EXTENDED)
        # Gen 3 (Extend) -> Poses 1-4 (STANDARD)
        # Gen 4 (Extend) -> Poses 5-8 (EXTENDED)
        # ...
        # So even generations use EXTENDED, odd use STANDARD
        if gen_num % 2 == 0:
            return settings.EXTENDED_POSES.copy()
        else:
            return settings.STANDARD_POSES.copy()
    elif mode == GenerationMode.RETRY_FAILED:
        if not failed_poses:
            raise ValueError("retry_failed mode requires failed_poses list")
        return failed_poses
    else:
        raise ValueError(f"Unknown mode: {mode}")


@router.post("/generate", response_model=GenerateResponse)
async def generate_views(request: GenerateRequest) -> GenerateResponse:
    """
    Start generation pipeline for a saree.
    
    Modes:
    - standard: Generate initial 4 views (poses 01-04)
    - extend: Generate additional 4 views (cycling between poses 05-08 and 01-04)
    - retry_failed: Re-run only failed poses from last generation
    
    Returns:
        GenerateResponse with job_id and status
    """
    saree_id = request.saree_id
    mode = request.mode
    
    logger.info(f"Generate request: saree_id={saree_id}, mode={mode.value}")
    
    # Verify saree exists
    original_path = get_original_path(saree_id)
    if not original_path:
        raise HTTPException(
            status_code=404,
            detail=f"Saree not found: {saree_id}"
        )
    
    # For retry_failed, load previous job to get failed poses
    failed_poses: Optional[List[str]] = None
    if mode == GenerationMode.RETRY_FAILED:
        from app.core.storage import load_json
        job_data = load_json(saree_id, "job.json")
        if not job_data:
            raise HTTPException(
                status_code=400,
                detail="No previous job found for retry"
            )
        failed_poses = job_data.get("failed_poses", [])
        if not failed_poses:
            raise HTTPException(
                status_code=400,
                detail="No failed poses to retry"
            )
    
    try:
        # Determine generation information FIRST to use gen_num for pose selection
        if mode == GenerationMode.RETRY_FAILED:
            # Use the same generation folder as the failed job
            from app.core.storage import load_json
            job_data = load_json(saree_id, "job.json")
            generation_name = job_data.get("generation_name")
            gen_num = 0 # Not used for pose selection in retry
            if not generation_name:
                raise HTTPException(
                    status_code=400,
                    detail="Could not determine generation folder for retry"
                )
        else:
            # Create new generation folder
            gen_num = get_next_generation_number(saree_id)
            generation_name = build_generation_name(gen_num, mode.value)

        # Get poses for this mode (passing gen_num for cycling)
        poses = get_poses_for_mode(mode, failed_poses, gen_num)
        
        # Generate job ID
        job_id = generate_job_id()
        
        # Create job state
        job_state = JobState(
            job_id=job_id,
            saree_id=saree_id,
            mode=mode,
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
        
        # Save job.json to saree directory
        save_json(saree_id, "job.json", job_state.to_dict())
        
        # Enqueue job to RQ
        try:
            redis_conn = Redis.from_url(settings.REDIS_URL)
            queue = Queue(settings.RQ_QUEUE_NAME, connection=redis_conn)
            
            # Import the pipeline task
            from app.pipeline.orchestrator import run_pipeline
            
            # Enqueue the job - pass as args to avoid kwarg conflicts with RQ
            queue.enqueue(
                run_pipeline,
                args=(job_id, saree_id),
                job_timeout='1h',  # Allow up to 1 hour for pipeline
            )
            
            logger.info(f"Enqueued job {job_id} for saree {saree_id}")
            
        except Exception as e:
            logger.error(f"Failed to enqueue job: {e}")
            # Update job status to failed
            job_state.status = JobStatus.FAILED
            job_state.error = f"Failed to enqueue job: {e}"
            save_json(saree_id, "job.json", job_state.to_dict())
            raise HTTPException(
                status_code=500,
                detail="Failed to queue generation job"
            )
        
        return GenerateResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating generation job: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create generation job: {e}"
        )
