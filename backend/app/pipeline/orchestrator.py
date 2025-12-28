"""
Pipeline Orchestrator.

Coordinates the execution of pipeline stages and manages job state.
Implements retry logic with failure reason injection.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.core.config import settings
from app.core.storage import (
    get_saree_dir,
    get_generation_dir,
    load_json,
    save_json,
)
from app.models import (
    FailureReason,
    GenerationMode,
    JobState,
    JobStatus,
    PipelineStage,
)
from app.pipeline.stages.isolate import run_isolate_stage
from app.pipeline.stages.flatten import run_flatten_stage
from app.pipeline.stages.extract import run_extract_stage
from app.pipeline.stages.compose import run_compose_stage
from app.pipeline.stages.validate import run_validate_stage

logger = logging.getLogger(__name__)


# Failure reason to retry instruction mapping
FAILURE_INSTRUCTIONS = {
    FailureReason.BORDER_DISTORTION.value: 
        "Previous output failed due to BORDER_DISTORTION. "
        "Preserve border geometry exactly; do not alter border width or motif spacing.",
    FailureReason.COLOR_SHIFT.value: 
        "Previous output failed due to COLOR_SHIFT. "
        "Match the exact colors from the source image. Do not adjust hue or saturation.",
    FailureReason.PATTERN_LOSS.value: 
        "Previous output failed due to PATTERN_LOSS. "
        "Preserve all pattern details and motifs from the source. Do not blur or simplify patterns.",
    FailureReason.TEXTURE_SMOOTHING_EXCESS.value: 
        "Previous output failed due to TEXTURE_SMOOTHING_EXCESS. "
        "Reduce smoothing. Preserve fabric texture details.",
    FailureReason.ALIGNMENT_FAILURE.value: 
        "Previous output failed due to ALIGNMENT_FAILURE. "
        "Align the saree parts correctly on the model pose.",
    FailureReason.SHADOW_PHOTOMETRIC_MISMATCH.value: 
        "Previous output failed due to SHADOW_PHOTOMETRIC_MISMATCH. "
        "Match the lighting and shadows to the model pose image.",
}


def build_retry_instruction(failure_reasons: list[str]) -> str:
    """Build a combined retry instruction from failure reasons."""
    instructions = []
    for reason in failure_reasons:
        if reason in FAILURE_INSTRUCTIONS:
            instructions.append(FAILURE_INSTRUCTIONS[reason])
    
    return " ".join(instructions) if instructions else ""


def compute_base_seed(saree_id: str, generation_name: str) -> int:
    """Compute a deterministic base seed for a generation."""
    seed_input = f"{saree_id}|{generation_name}"
    seed_hash = hashlib.sha256(seed_input.encode()).digest()
    return int.from_bytes(seed_hash[:4], byteorder='big')


def update_job_state(
    saree_id: str,
    job_id: str,
    updates: dict[str, Any],
) -> JobState:
    """
    Update job state and save to job.json.
    
    Args:
        saree_id: Saree identifier
        job_id: Job identifier
        updates: Fields to update
        
    Returns:
        Updated JobState
    """
    job_data = load_json(saree_id, "job.json")
    
    if not job_data:
        raise ValueError(f"Job not found: {job_id}")
    
    # Apply updates
    for key, value in updates.items():
        job_data[key] = value
    
    job_data["updated_at"] = datetime.utcnow().isoformat()
    
    # Save updated state
    save_json(saree_id, "job.json", job_data)
    
    return JobState.from_dict(job_data)


def append_retry_log(
    saree_id: str,
    generation_name: str,
    pose_id: str,
    attempt: int,
    seed: int,
    prompt: str,
    failure_reasons: list[str],
    injected_instructions: str,
    adapter_response: Optional[dict] = None,
) -> None:
    """Append an entry to the retry log."""
    log_path = get_generation_dir(saree_id, generation_name) / "retry_log.json"
    
    existing = {"entries": []}
    if log_path.exists():
        with open(log_path) as f:
            existing = json.load(f)
    
    entry = {
        "pose_id": pose_id,
        "attempt": attempt,
        "timestamp": datetime.utcnow().isoformat(),
        "seed": seed,
        "prompt": prompt,
        "failure_reasons": failure_reasons,
        "injected_instructions": injected_instructions,
        "adapter_response": adapter_response,
    }
    
    existing["entries"].append(entry)
    
    with open(log_path, 'w') as f:
        json.dump(existing, f, indent=2)


def run_pipeline(job_id: str, saree_id: str) -> dict:
    """
    Run the complete pipeline for a job.
    
    This is the main entry point called by the RQ worker.
    
    Args:
        job_id: Job identifier
        saree_id: Saree identifier
        
    Returns:
        Final job result
    """
    logger.info(f"[{job_id}] Starting pipeline for saree {saree_id}")
    
    # Load job state
    job_data = load_json(saree_id, "job.json")
    if not job_data:
        logger.error(f"[{job_id}] Job state not found")
        return {"status": "failed", "error": "Job state not found"}
    
    job_state = JobState.from_dict(job_data)
    
    try:
        # Update status to running
        update_job_state(saree_id, job_id, {
            "status": JobStatus.RUNNING.value,
            "current_stage": PipelineStage.ISOLATE.value,
            "progress": 5,
        })
        
        # Stage 1: Isolate (background removal)
        logger.info(f"[{job_id}] Stage: ISOLATE")
        isolate_result = run_isolate_stage(saree_id, job_id)
        
        if isolate_result["status"] == "failed":
            raise Exception(f"Isolate stage failed: {isolate_result.get('error')}")
        
        update_job_state(saree_id, job_id, {
            "current_stage": PipelineStage.FLATTEN.value,
            "progress": 15,
        })
        
        # Stage 2: Flatten (AI-assisted)
        logger.info(f"[{job_id}] Stage: FLATTEN")
        base_seed = compute_base_seed(saree_id, job_state.generation_name)
        flatten_result = run_flatten_stage(saree_id, job_id, seed=base_seed)
        
        if flatten_result["status"] == "failed":
            raise Exception(f"Flatten stage failed: {flatten_result.get('error')}")
        
        update_job_state(saree_id, job_id, {
            "current_stage": PipelineStage.EXTRACT.value,
            "progress": 30,
        })
        
        # Stage 3: Extract parts
        logger.info(f"[{job_id}] Stage: EXTRACT")
        extract_result = run_extract_stage(saree_id, job_id)
        
        if extract_result["status"] == "failed":
            raise Exception(f"Extract stage failed: {extract_result.get('error')}")
        
        update_job_state(saree_id, job_id, {
            "current_stage": PipelineStage.COMPOSE.value,
            "progress": 45,
        })
        
        # Stage 4: Compose poses
        logger.info(f"[{job_id}] Stage: COMPOSE")
        poses_to_process = job_state.poses.copy()
        retry_instructions: dict[str, str] = {}
        max_retries = settings.MAX_RETRIES
        
        all_completed = []
        all_failed = []
        retry_counts = job_state.retry_counts.copy()
        
        # Main compose + validate loop with retries
        while poses_to_process:
            # Compose the poses
            compose_result = run_compose_stage(
                saree_id=saree_id,
                job_id=job_id,
                generation_name=job_state.generation_name,
                poses=poses_to_process,
                base_seed=base_seed,
                retry_instructions=retry_instructions,
            )
            
            update_job_state(saree_id, job_id, {
                "current_stage": PipelineStage.VALIDATE.value,
                "progress": 70,
            })
            
            # Stage 5: Validate
            logger.info(f"[{job_id}] Stage: VALIDATE")
            validate_result = run_validate_stage(
                saree_id=saree_id,
                job_id=job_id,
                generation_name=job_state.generation_name,
                poses=poses_to_process,
            )
            
            # Process validation results
            newly_failed = []
            
            for pose_id in poses_to_process:
                pose_result = validate_result["results"].get(pose_id, {})
                
                if pose_result.get("passed"):
                    all_completed.append(pose_id)
                    logger.info(f"[{job_id}] Pose {pose_id} passed validation")
                else:
                    # Check retry count
                    current_retries = retry_counts.get(pose_id, 0)
                    
                    if current_retries < max_retries:
                        # Prepare for retry
                        retry_counts[pose_id] = current_retries + 1
                        failure_reasons = pose_result.get("failure_reasons", [])
                        instruction = build_retry_instruction(failure_reasons)
                        retry_instructions[pose_id] = instruction
                        newly_failed.append(pose_id)
                        
                        # Log retry
                        pose_seed = hash(f"{saree_id}|{job_state.generation_name}|{pose_id}|{base_seed}")
                        append_retry_log(
                            saree_id=saree_id,
                            generation_name=job_state.generation_name,
                            pose_id=pose_id,
                            attempt=current_retries + 1,
                            seed=pose_seed,
                            prompt=f"Compose {pose_id}",
                            failure_reasons=failure_reasons,
                            injected_instructions=instruction,
                        )
                        
                        logger.warning(
                            f"[{job_id}] Pose {pose_id} failed validation "
                            f"(attempt {current_retries + 1}/{max_retries}): {failure_reasons}"
                        )
                    else:
                        # Max retries exhausted
                        all_failed.append(pose_id)
                        logger.error(
                            f"[{job_id}] Pose {pose_id} failed after {max_retries} retries"
                        )
            
            # Update poses to process for next iteration
            poses_to_process = newly_failed
            
            if poses_to_process:
                # Update progress for retry
                progress = min(85, 70 + (len(all_completed) / len(job_state.poses)) * 15)
                update_job_state(saree_id, job_id, {
                    "current_stage": PipelineStage.COMPOSE.value,
                    "progress": int(progress),
                    "retry_counts": retry_counts,
                })
        
        # Determine final status
        if len(all_completed) == len(job_state.poses):
            final_status = JobStatus.SUCCESS
        elif len(all_completed) > 0:
            final_status = JobStatus.PARTIAL
        else:
            final_status = JobStatus.FAILED
        
        # Update final job state
        update_job_state(saree_id, job_id, {
            "status": final_status.value,
            "current_stage": PipelineStage.COMPLETE.value,
            "progress": 100,
            "completed_poses": all_completed,
            "failed_poses": all_failed,
            "retry_counts": retry_counts,
        })
        
        logger.info(
            f"[{job_id}] Pipeline complete: {len(all_completed)} success, "
            f"{len(all_failed)} failed"
        )
        
        return {
            "status": final_status.value,
            "completed_poses": all_completed,
            "failed_poses": all_failed,
        }
        
    except Exception as e:
        logger.error(f"[{job_id}] Pipeline failed: {e}")
        
        update_job_state(saree_id, job_id, {
            "status": JobStatus.FAILED.value,
            "current_stage": PipelineStage.FAILED.value,
            "error": str(e),
        })
        
        return {
            "status": "failed",
            "error": str(e),
        }
