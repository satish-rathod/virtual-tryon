"""
Configuration settings for the Saree Virtual Try-On backend.

Uses Pydantic BaseSettings for environment variable management with sensible defaults.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Storage configuration
    STORAGE_ROOT: Path = Path("./storage")
    
    # Redis configuration
    REDIS_URL: str = "redis://localhost:6379"
    
    # Worker configuration
    WORKER_CONCURRENCY: int = 2
    RQ_QUEUE_NAME: str = "saree_pipeline"
    
    # Retry configuration
    MAX_RETRIES: int = 3
    
    # Validation thresholds (per docs/VALIDATION.md)
    SSIM_THRESHOLD: float = 0.35
    DELTA_E_THRESHOLD: float = 35.0
    PATTERN_MATCH_THRESHOLD: float = 0.10
    
    # AI Adapter configuration
    AI_ADAPTER_TYPE: str = "mock"  # "mock" or "gemini"
    AI_ADAPTER_API_KEY: Optional[str] = None
    
    # Gemini-specific configuration
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash-image"  # or "gemini-3-pro-image-preview"
    
    # Pose mapping configuration
    STANDARD_POSES: list[str] = ["pose_01", "pose_02", "pose_03", "pose_04"]
    EXTENDED_POSES: list[str] = [
        "pose_05", "pose_06", "pose_07", "pose_08",
        "pose_09", "pose_10", "pose_11", "pose_12"
    ]
    
    # Assets directory
    ASSETS_ROOT: Path = Path("./assets")

    
    def get_storage_root(self) -> Path:
        """Get the absolute storage root path, creating it if needed."""
        path = self.STORAGE_ROOT.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_saree_dir(self, saree_id: str) -> Path:
        """Get the directory for a specific saree."""
        return self.get_storage_root() / saree_id
    
    def get_generation_dir(self, saree_id: str, generation_name: str) -> Path:
        """Get the directory for a specific generation."""
        return self.get_saree_dir(saree_id) / "generations" / generation_name


# Global settings instance
settings = Settings()
