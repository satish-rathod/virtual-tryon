# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-12-27

### Added

#### Backend Implementation
- **FastAPI Application** (`app/main.py`)
  - CORS middleware configured
  - Health check endpoint at `/health`
  - API routes registered under `/api` prefix

- **Configuration** (`app/core/config.py`)
  - Pydantic BaseSettings for environment variables
  - Configurable thresholds for SSIM, ΔE, pattern matching
  - Pose mappings for standard (01-04) and extended (05-12) modes

- **Storage Layer** (`app/core/storage.py`)
  - Canonical path builders matching `docs/STORAGE.md`
  - Immutable artifact storage (no overwrites)
  - JSON persistence for job state and metrics

- **API Endpoints**
  - `POST /api/upload` - Upload floor-shot saree images
  - `POST /api/generate` - Start generation pipeline (standard/extend/retry_failed modes)
  - `GET /api/status/{job_id}` - Poll job status and progress
  - `GET /api/gallery` - List all sarees with metadata
  - `GET /api/artifacts/{saree_id}/{path}` - Serve artifact files
  - `GET /api/logs/{job_id}` - Get retry logs and failure reasons

- **AI Adapter** (`app/ai/adapter.py`)
  - Abstract `AIAdapter` base class
  - Deterministic `MockAIAdapter` using OpenCV/Pillow
  - Seedable transformations for reproducibility
  - No external API calls in mock mode

- **Pipeline Stages**
  - `isolate.py` - Deterministic background removal using Otsu thresholding
  - `flatten.py` - AI-assisted fold flattening with bilateral filtering
  - `extract.py` - Layout-guided part extraction (pallu, body, borders)
  - `compose.py` - Pose composition with placeholder generation
  - `validate.py` - Per-region validation (SSIM, CIEDE2000 ΔE, ORB pattern match)

- **Pipeline Orchestrator** (`app/pipeline/orchestrator.py`)
  - Sequential stage execution
  - Job state management with progress tracking
  - Retry logic up to 3 attempts with failure reason injection
  - Retry log persistence

- **RQ Worker** (`app/worker/worker.py`)
  - Redis queue consumer for background jobs
  - Configurable queue name

#### Docker Support
- `Dockerfile` for FastAPI backend
- `worker.Dockerfile` for RQ worker
- `docker-compose.yml` with backend, redis, worker services
- Shared volume for storage directory

#### Tests
- Unit tests for upload endpoint
- Unit tests for generate endpoint
- Unit tests for all pipeline stages
- Unit tests for validation metrics (SSIM, ΔE, pattern match)
- Integration test for full standard generation flow

#### Documentation
- Updated `docs/SETUP.md` with run instructions
- Created `.env.example` with all environment variables documented

### Technical Details

- **Determinism**: All operations use seeded random state derived from `saree_id + generation_name + pose_id`
- **Validation Thresholds**: SSIM ≥ 0.80, ΔE ≤ 6.0, Pattern Match ≥ 0.40
- **Retry Logic**: Up to 3 retries per pose with failure-specific instructions injected
- **Storage Layout**: Per `docs/STORAGE.md` specification with immutable artifacts
