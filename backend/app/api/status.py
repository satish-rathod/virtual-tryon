"""
Status, gallery, artifacts, and logs API endpoints.

GET /api/status/{job_id} - Job status and progress
GET /api/gallery - List all sarees
GET /api/artifacts/{saree_id}/{artifact_path} - Serve artifact files
GET /api/logs/{job_id} - Retry logs and failure reasons
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.storage import (
    get_saree_dir,
    get_generation_dir,
    list_sarees,
    load_json,
)
from app.models import (
    GalleryItem,
    GalleryResponse,
    JobStatus,
    LogsResponse,
    PipelineStage,
    StatusResponse,
    SareeDetails,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["status"])


def find_job_by_id(job_id: str) -> Optional[Tuple[str, dict]]:
    """
    Find a job by its ID across all sarees.
    
    Returns (saree_id, job_data) or None if not found.
    """
    storage_root = settings.get_storage_root()
    
    for saree_dir in storage_root.iterdir():
        if not saree_dir.is_dir():
            continue
        
        job_path = saree_dir / "job.json"
        if job_path.exists():
            job_data = load_json(saree_dir.name, "job.json")
            if job_data and job_data.get("job_id") == job_id:
                return saree_dir.name, job_data
    
    return None


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str) -> StatusResponse:
    """
    Get the status and progress of a job.
    
    Returns job status, current stage, progress percentage,
    list of completed artifacts, and any errors.
    """
    result = find_job_by_id(job_id)
    
    if not result:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    saree_id, job_data = result
    
    # Build list of completed artifacts
    artifacts = []
    generation_name = job_data.get("generation_name")
    
    if generation_name:
        gen_dir = get_generation_dir(saree_id, generation_name)
        if gen_dir.exists():
            for f in gen_dir.iterdir():
                if f.is_file() and f.suffix in [".png", ".jpg", ".jpeg"]:
                    artifacts.append(f.name)
    
    # Build metrics URL if available
    metrics_url = None
    if generation_name:
        metrics_path = get_generation_dir(saree_id, generation_name) / "metrics.json"
        if metrics_path.exists():
            metrics_url = f"/api/artifacts/{saree_id}/generations/{generation_name}/metrics.json"
    
    return StatusResponse(
        job_id=job_id,
        saree_id=saree_id,
        status=JobStatus(job_data.get("status", "queued")),
        progress=job_data.get("progress", 0),
        current_stage=PipelineStage(job_data.get("current_stage", "pending")),
        artifacts=sorted(artifacts),
        metrics_url=metrics_url,
        failed_poses=job_data.get("failed_poses", []),
        error=job_data.get("error"),
    )


@router.get("/gallery", response_model=list[GalleryItem])
async def get_gallery() -> list[GalleryItem]:
    """
    List all sarees with metadata for gallery view.
    
    Returns list of sarees with: saree_id, created_at, thumbnail,
    generation_count, latest_status.
    """
    sarees = list_sarees()
    
    items = [
        GalleryItem(
            saree_id=s["saree_id"],
            created_at=s.get("created_at"),
            thumbnail=s.get("thumbnail"),
            generation_count=s.get("generation_count", 0),
            latest_status=s.get("latest_status"),
        )
        for s in sarees
    ]
    
    return items


    return items


@router.get("/gallery/{saree_id}", response_model=SareeDetails)
async def get_saree_details(saree_id: str):
    """
    Get detailed information about a saree.
    """
    from app.models import SareeDetails, SareeArtifacts, Generation, ViewArtifact
    
    saree_dir = get_saree_dir(saree_id)
    if not saree_dir.exists():
        raise HTTPException(status_code=404, detail="Saree not found")
        
    # Check for core artifacts
    artifacts = SareeArtifacts(
        original=f"/api/artifacts/{saree_id}/original.jpg"
    )
    
    if (saree_dir / "S_clean.png").exists():
        artifacts.cleaned = f"/api/artifacts/{saree_id}/S_clean.png"
    
    if (saree_dir / "S_flat.png").exists():
        artifacts.flattened = f"/api/artifacts/{saree_id}/S_flat.png"
        
    parts_dir = saree_dir / "parts"
    if parts_dir.exists():
        for part in parts_dir.iterdir():
            if part.is_file() and part.suffix == ".png":
                artifacts.parts[part.stem] = f"/api/artifacts/{saree_id}/parts/{part.name}"
    
    # Job data loaded earlier
    job_data = load_json(saree_id, "job.json")
    
    # Initialize created_at with fallback, then refine from job_data
    created_at = datetime.utcnow().isoformat()
    # If there is an active job but its folder hasn't been created yet
    if job_data:
        created_at = job_data.get("created_at", created_at)
        
    # List generations
    generations = []
    has_failures = False
    
    gens_dir = saree_dir / "generations"
    if gens_dir.exists():
        # Sort generations by creation time (inferred from name or stat)
        # Assuming names like gen_01_standard, gen_02_extend
        gen_folders = sorted([d for d in gens_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        
        for gen_folder in gen_folders:
            # Try to load metrics for status
            metrics_path = gen_folder / "metrics.json"
            metrics = load_json(saree_id, "metrics.json", gen_folder.name)
            
            # Determine status and views
            views = []
            status = "success"
            timestamp = datetime.utcnow().isoformat() # Fallback
            
            if metrics:
                timestamp = metrics.get("timestamp", timestamp)
                # Parse views from metrics or file listing
                # File listing is safer
            
            # List view files
            view_files = sorted([f for f in gen_folder.iterdir() if f.name.startswith("final_view_") and f.suffix == ".png"])
            for view_file in view_files:
                # final_view_01.png
                try:
                    view_num = int(view_file.stem.split("_")[-1])
                    views.append(ViewArtifact(
                        view_number=view_num,
                        image_url=f"/api/artifacts/{saree_id}/generations/{gen_folder.name}/{view_file.name}",
                        status="success"
                    ))
                except ValueError:
                    continue
            
            # Check if this is the active job
            is_active_job = job_data and job_data.get("generation_name") == gen_folder.name
            
            if is_active_job:
                status = job_data.get("status", "queued")
                
                # If running/queued and no views yet, create placeholders
                if not views and status in ["queued", "running", "partial"]:
                    poses = job_data.get("poses", [])
                    for i, pose in enumerate(poses):
                        # Construct a placeholder view
                        # Use pose index mapping if possible, or just sequential
                        view_num = i + 1
                        views.append(ViewArtifact(
                            view_number=view_num,
                            image_url=f"placeholder_{view_num}", # Placeholder
                            status="pending"
                        ))
            
            # Determine status if not already set by active job logic
            if not is_active_job and not views:
                status = "failed"
            elif not is_active_job and views:
                status = "success"
            
            generations.append(Generation(
                generation_id=gen_folder.name,
                label=gen_folder.name.replace("_", " ").title(),
                mode="standard" if "standard" in gen_folder.name else "extend",
                status=status,
                timestamp=timestamp,
                views=views,
                retry_count=0, 
                metrics_url=f"/api/artifacts/{saree_id}/generations/{gen_folder.name}/metrics.json" if metrics_path.exists() else None
            ))
            
    # Load job.json for the *latest* status and potential failures
    # Job data loaded earlier
    # If there is an active job but its folder hasn't been created yet (e.g. queued very recently),
    # we must synthetically add it so the frontend knows something is happening.
    if job_data:
        active_gen_name = job_data.get("generation_name")
        active_status = job_data.get("status", "queued")
        
        # Check if we already processed this generation in the loop above
        already_listed = any(g.generation_id == active_gen_name for g in generations)
        
        if active_gen_name and not already_listed and active_status in ["queued", "running"]:
            # Create synthetic views for placeholders
            views = []
            poses = job_data.get("poses", [])
            for i, pose in enumerate(poses):
                view_num = i + 1
                views.append(ViewArtifact(
                    view_number=view_num,
                    image_url=f"placeholder_{view_num}",
                    status="pending"
                ))
            
            generations.append(Generation(
                generation_id=active_gen_name,
                label=active_gen_name.replace("_", " ").title(),
                mode="standard" if "standard" in active_gen_name else "extend",
                status=active_status,
                timestamp=job_data.get("created_at", created_at),
                views=views,
                retry_count=0,
                metrics_url=None
            ))

    return SareeDetails(
        saree_id=saree_id,
        created_at=created_at,
        artifacts=artifacts,
        generations=generations,
        has_failures=has_failures
    )
@router.get("/artifacts/{saree_id}/{artifact_path:path}")
async def get_artifact(saree_id: str, artifact_path: str) -> FileResponse:
    """
    Serve an artifact file.
    
    The artifact_path may include subdirectories like:
    - original.jpg
    - S_clean.png
    - generations/gen_01_standard/final_view_01.png
    - parts/pallu.png
    """
    saree_dir = get_saree_dir(saree_id)
    
    if not saree_dir.exists():
        raise HTTPException(status_code=404, detail=f"Saree not found: {saree_id}")
    
    # Build full path and validate it's within the saree directory
    file_path = saree_dir / artifact_path
    
    # Security: ensure path is within saree directory
    try:
        file_path = file_path.resolve()
        saree_dir_resolved = saree_dir.resolve()
        
        if not str(file_path).startswith(str(saree_dir_resolved)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_path}")
    
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    
    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".json": "application/json",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_path.name,
    )


@router.get("/logs/{job_id}", response_model=LogsResponse)
async def get_job_logs(job_id: str) -> LogsResponse:
    """
    Get retry logs and failure reasons for a job.
    
    Returns the retry_log.json contents and a summary of failure reasons.
    """
    result = find_job_by_id(job_id)
    
    if not result:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    saree_id, job_data = result
    generation_name = job_data.get("generation_name")
    
    retry_log: list[dict] = []
    failure_reasons: list[str] = []
    
    if generation_name:
        log_data = load_json(saree_id, "retry_log.json", generation_name)
        if log_data:
            retry_log = log_data.get("entries", [])
            
            # Extract unique failure reasons
            for entry in retry_log:
                reasons = entry.get("failure_reasons", [])
                for reason in reasons:
                    if reason not in failure_reasons:
                        failure_reasons.append(reason)
    
    return LogsResponse(
        job_id=job_id,
        retry_log=retry_log,
        failure_reasons=failure_reasons,
    )
