# Saree Virtual Try-On — Repository README

Reference: Product Requirements Document (PRD).

## Project Summary
Saree Virtual Try-On is a deterministic, constrained image-composition system that converts a single floor-shot saree image into a photorealistic image of one fixed model wearing that exact saree. The system prioritizes preservation of design, borders, pallu, texture and repeatability across runs.

## Goals
- Preserve saree identity (motifs, borders, color tones).
- Deterministic outputs (repeatability across runs).
- Minimal creative drift (fixed model, fixed poses, controlled AI usage).
- Local artifact storage and transparent retry/validation logs.
- **Read our detailed methodology:** [Solution Approach](SOLUTION_APPROACH.md)

## Key UI constraint
- Initial generation is fixed and **always** produces 4 standard views. The frontend does **not** expose pose selection during the initial generation. Users may request "Generate More Views" later to create the remaining additional poses.

## Repo structure
## Repo structure

```
Root
├─ backend/                     # FastAPI backend + controllers + workers
│  ├─ app/
│  │  ├─ main.py
│  │  ├─ api/                   # REST endpoints (upload, generate, status)
│  │  ├─ pipeline/              # orchestration & job state machine
│  │  ├─ validators/            # automated validation logic (SSIM, ΔE, pattern)
│  │  ├─ storage/               # local filesystem interface, artifact naming
│  │  └─ ai/                    # wrappers for AI operations (flattening, warping)
├─ frontend/                    # Next.js minimal app for upload + results + gallery
│  ├─ app/
│  └─ components/
├─ docs/                        # detailed documentation
│  ├─ ARCHITECTURE.md
│  ├─ PIPELINE.md
│  ├─ VALIDATION.md
│  ├─ STORAGE.md
│  └─ FRONTEND.md
└─ README.md
```

## Quick start (developer)
1. Clone repo.
2. Install system dependencies (see `SETUP.md`).
3. Start local services:
   - Backend: `uvicorn app.main:app --reload`
   - Worker: `python -m app.worker.worker`
   - Frontend: `npm run dev`
4. Run test upload: `curl -F "file=@/path/to/saree.jpg" http://localhost:8000/api/upload`
5. Trigger generation: `POST /api/generate` with `saree_id` (see `API.md`).

## Important notes
- Determinism is enforced by fixed assets, deterministic segmentation & layout-driven extraction steps, and seeding of any AI modules where available.
- Maximum retry per generation = 3. Fail reasons are logged and injected into subsequent AI prompts.

## Next steps / Roadmap
- Tighten validation thresholds after initial telemetry.
- Add optional internal UI for manual review of failure reasons (out of scope for Round 1).
