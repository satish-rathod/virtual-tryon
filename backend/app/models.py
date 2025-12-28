"""
Pydantic models for API requests and responses.

Defines enums for generation modes, job statuses, and pipeline stages,
plus complete request/response models matching docs/API.md contracts.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class GenerationMode(str, Enum):
    """Supported generation modes."""
    STANDARD = "standard"  # Initial generation → poses 01-04
    EXTEND = "extend"      # Additional views → poses 05-12
    RETRY_FAILED = "retry_failed"  # Re-run only failed poses


class JobStatus(str, Enum):
    """Job execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # Some poses failed after retries


class PipelineStage(str, Enum):
    """Pipeline processing stages."""
    PENDING = "pending"
    ISOLATE = "isolate"
    FLATTEN = "flatten"
    EXTRACT = "extract"
    COMPOSE = "compose"
    VALIDATE = "validate"
    COMPLETE = "complete"
    FAILED = "failed"


class FailureReason(str, Enum):
    """Failure reason taxonomy from docs/VALIDATION.md."""
    BORDER_DISTORTION = "BORDER_DISTORTION"
    COLOR_SHIFT = "COLOR_SHIFT"
    PATTERN_LOSS = "PATTERN_LOSS"
    TEXTURE_SMOOTHING_EXCESS = "TEXTURE_SMOOTHING_EXCESS"
    ALIGNMENT_FAILURE = "ALIGNMENT_FAILURE"
    SHADOW_PHOTOMETRIC_MISMATCH = "SHADOW_PHOTOMETRIC_MISMATCH"


# === API Request Models ===

class GenerateRequest(BaseModel):
    """Request body for POST /api/generate."""
    saree_id: str = Field(..., description="UUID of the uploaded saree")
    mode: GenerationMode = Field(
        default=GenerationMode.STANDARD,
        description="Generation mode: standard, extend, or retry_failed"
    )


# === API Response Models ===

class UploadResponse(BaseModel):
    """Response for POST /api/upload."""
    saree_id: str
    upload_path: str


class GenerateResponse(BaseModel):
    """Response for POST /api/generate."""
    job_id: str
    status: JobStatus


class StatusResponse(BaseModel):
    """Response for GET /api/status/{job_id}."""
    job_id: str
    saree_id: str
    status: JobStatus
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    current_stage: PipelineStage
    artifacts: list[str] = Field(default_factory=list)
    metrics_url: Optional[str] = None
    failed_poses: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class GalleryItem(BaseModel):
    """Single item in gallery list."""
    saree_id: str
    created_at: Optional[str] = None
    thumbnail: Optional[str] = None
    generation_count: int = 0
    latest_status: Optional[str] = None


class GalleryResponse(BaseModel):
    """Response for GET /api/gallery."""
    items: list[GalleryItem]


class LogsResponse(BaseModel):
    """Response for GET /api/logs/{job_id}."""
    job_id: str
    retry_log: list[dict[str, Any]] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)


class ViewArtifact(BaseModel):
    """Single view in a generation."""
    view_number: int
    image_url: str
    status: str  # success, failed, pending


class Generation(BaseModel):
    """Generation history item."""
    generation_id: str
    label: str
    mode: str
    status: str
    timestamp: str
    views: list[ViewArtifact]
    retry_count: int
    metrics_url: Optional[str] = None


class SareeArtifacts(BaseModel):
    """Artifacts paths for SareeDetails."""
    original: str
    cleaned: Optional[str] = None
    flattened: Optional[str] = None
    parts: dict[str, str] = Field(default_factory=dict)


class SareeDetails(BaseModel):
    """Response for GET /api/gallery/{saree_id}."""
    saree_id: str
    created_at: str
    artifacts: SareeArtifacts
    generations: list[Generation]
    has_failures: bool


# === Internal Models ===

class JobState(BaseModel):
    """Job state model persisted to job.json."""
    job_id: str
    saree_id: str
    mode: GenerationMode
    status: JobStatus = JobStatus.QUEUED
    current_stage: PipelineStage = PipelineStage.PENDING
    progress: int = 0
    generation_name: Optional[str] = None
    poses: list[str] = Field(default_factory=list)
    completed_poses: list[str] = Field(default_factory=list)
    failed_poses: list[str] = Field(default_factory=list)
    retry_counts: dict[str, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "saree_id": self.saree_id,
            "mode": self.mode.value,
            "status": self.status.value,
            "current_stage": self.current_stage.value,
            "progress": self.progress,
            "generation_name": self.generation_name,
            "poses": self.poses,
            "completed_poses": self.completed_poses,
            "failed_poses": self.failed_poses,
            "retry_counts": self.retry_counts,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobState":
        """Create from dictionary."""
        return cls(
            job_id=data["job_id"],
            saree_id=data["saree_id"],
            mode=GenerationMode(data["mode"]),
            status=JobStatus(data["status"]),
            current_stage=PipelineStage(data["current_stage"]),
            progress=data.get("progress", 0),
            generation_name=data.get("generation_name"),
            poses=data.get("poses", []),
            completed_poses=data.get("completed_poses", []),
            failed_poses=data.get("failed_poses", []),
            retry_counts=data.get("retry_counts", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            error=data.get("error"),
        )


class RegionMetrics(BaseModel):
    """Validation metrics for a single region."""
    region: str
    ssim: float
    delta_e: float
    pattern_score: float
    passed: bool


class ValidationResult(BaseModel):
    """Complete validation result for a pose."""
    pose_id: str
    timestamp: str
    regions: list[RegionMetrics]
    overall_passed: bool
    failure_reasons: list[FailureReason] = Field(default_factory=list)


class RetryLogEntry(BaseModel):
    """Entry in retry_log.json."""
    pose_id: str
    attempt: int
    timestamp: str
    seed: int
    prompt: str
    failure_reasons: list[str]
    injected_instructions: str
    adapter_response: Optional[dict[str, Any]] = None
