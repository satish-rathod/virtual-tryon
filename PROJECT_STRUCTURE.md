# Project structure and folder responsibilities

Root
├─ backend/                     # FastAPI backend + controllers + workers
│  ├─ app/
│  │  ├─ main.py
│  │  ├─ api/                   # REST endpoints (upload, generate, status)
│  │  ├─ pipeline/              # orchestration & job state machine
│  │  ├─ validators/            # automated validation logic (SSIM, ΔE, pattern)
│  │  ├─ storage/               # local filesystem interface, artifact naming
│  │  └─ ai/                    # wrappers for AI operations (flattening, warping)
│  └─ Dockerfile
├─ worker/                      # background worker (Celery/RQ) for long jobs
│  ├─ worker.py
│  └─ Dockerfile
├─ frontend/                     # Next.js minimal app for upload + results + gallery
│  ├─ pages/
│  └─ components/
├─ assets/                       # fixed model poses + placement maps + layouts
│  ├─ model/poses/pose_01.png
│  ├─ model/poses/...pose_12.png
│  └─ layout/                    # layout references & masks
├─ scripts/                      # helper scripts (evaluation, seeding, tests)
├─ docs/
│  ├─ ARCHITECTURE.md
│  ├─ PIPELINE.md
│  ├─ VALIDATION.md
│  ├─ STORAGE.md
│  └─ FRONTEND.md
├─ tests/                        # unit and integration tests
├─ docker-compose.yml
└─ README.md
