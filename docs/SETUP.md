# Developer Setup (Round 1)

## Requirements
- Docker & Docker Compose
- Python 3.10+
- Node 18+ (for Next.js frontend)
- Redis (for job queue)

## Quick Start with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Services:
# - Backend API: http://localhost:8000
# - Frontend: http://localhost:3000 (if configured)
# - Redis: localhost:6379
```

## Local Development (without Docker)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Start backend
uvicorn app.main:app --reload --port 8000
```

### Start Redis (required for job queue)

```bash
# Using Docker
docker run -d -p 6379:6379 redis:alpine

# Or install locally
brew install redis  # macOS
redis-server
```

### Start RQ Worker

```bash
cd backend
source .venv/bin/activate
python -m app.worker.worker
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## Environment Variables

See `backend/.env.example` for all options. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_ROOT` | `./storage` | Path to artifact storage |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `MAX_RETRIES` | `3` | Max retry attempts per pose |
| `SSIM_THRESHOLD` | `0.80` | Minimum SSIM for validation |
| `DELTA_E_THRESHOLD` | `6.0` | Maximum ΔE for color validation |
| `PATTERN_MATCH_THRESHOLD` | `0.40` | Minimum pattern match score |
| `AI_ADAPTER_TYPE` | `mock` | `mock` or `external` |

## Running Tests

```bash
cd backend
source .venv/bin/activate

# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit -v

# Integration tests only
pytest tests/integration -v

# With coverage
pytest tests/ -v --cov=app
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/upload` | Upload saree image |
| POST | `/api/generate` | Start generation |
| GET | `/api/status/{job_id}` | Get job status |
| GET | `/api/gallery` | List all sarees |
| GET | `/api/artifacts/{saree_id}/{path}` | Get artifact |
| GET | `/api/logs/{job_id}` | Get retry logs |

## Debugging Failed Jobs

### 1. Check Job Status

```bash
curl http://localhost:8000/api/status/<job_id>
```

### 2. Inspect job.json

```bash
cat storage/<saree_id>/job.json | jq
```

Fields to check:
- `status`: Current job status (queued/running/success/failed/partial)
- `current_stage`: Where the pipeline is/stopped
- `failed_poses`: List of poses that failed validation
- `retry_counts`: Number of retries per pose
- `error`: Error message if job failed

### 3. Inspect metrics.json

```bash
cat storage/<saree_id>/generations/gen_01_standard/metrics.json | jq
```

Check per-pose validation results:
- `ssim`: Structural similarity (should be ≥ 0.80)
- `delta_e`: Color difference (should be ≤ 6.0)
- `pattern_score`: Pattern match (should be ≥ 0.40)
- `failure_reasons`: Why validation failed

### 4. Inspect retry_log.json

```bash
cat storage/<saree_id>/generations/gen_01_standard/retry_log.json | jq
```

Shows retry history:
- `attempt`: Retry attempt number
- `failure_reasons`: What failed
- `injected_instructions`: Instructions given to AI adapter
- `seed`: Deterministic seed used

### 5. Worker Logs

```bash
# Docker
docker-compose logs worker

# Local
# Check terminal where worker is running
```

## Storage Layout

Per `docs/STORAGE.md`:

```
storage/<saree_id>/
├── original.jpg          # Uploaded image
├── S_clean.png           # Background removed
├── S_flat.png            # Flattened saree
├── parts/
│   ├── pallu.png
│   ├── body.png
│   ├── top_border.png
│   ├── bottom_border.png
│   └── parts.json
├── assets/               # Pose assets used
├── generations/
│   └── gen_01_standard/
│       ├── final_view_01.png
│       ├── final_view_02.png
│       ├── final_view_03.png
│       ├── final_view_04.png
│       ├── metrics.json
│       └── retry_log.json
└── job.json              # Job metadata
```