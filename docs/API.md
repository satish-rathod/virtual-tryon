# Backend API (HTTP) — endpoints and request/response contracts

Base: `/api`

## POST /api/upload
- Description: Upload a floor-shot saree image.
- Form data: `file` (image)
- Response: `{ "saree_id": "<uuid>", "upload_path": "artifacts/<saree_id>/original.jpg" }`

## POST /api/generate
- Description: Start generation pipeline for a saree.
- JSON body (initial generation):
  {
    "saree_id": "<uuid>",
    "mode": "standard" // optional; default if omitted is "standard"
  }
- Modes:
  - `standard` → backend maps to poses 01–04 (initial locked generation).
  - `extend` → backend maps to poses 05–12 (generate remaining views as a batch).
  - `retry_failed` → backend will re-run only failed pose outputs using logged reasons.
- Response: `{ "job_id":"<uuid>", "status":"queued" }`

## GET /api/status/{job_id}
- Description: Poll job status.
- Response:
  {
    "job_id":"<uuid>",
    "status":"queued|running|success|failed",
    "progress": 40,
    "current_stage":"ai_flatten",
    "artifacts":[ "final_pose_01.png" ],
    "metrics_url":"/api/artifacts/<job_id>/metrics.json"
  }

## GET /api/gallery
- Description: List saree folders for gallery view.
- Response: array of `{ saree_id, created_at, thumbnail, generation_count, latest_status }`

## GET /api/artifacts/{saree_id}/{artifact_path}
- Returns artifact (image or json). The artifact path may include generation folder: `generations/gen_01/final_pose_01.png`.

## GET /api/logs/{job_id}
- Returns retry log and readable failure reasons.

## Authentication
- Round 1: no auth required (out of scope). Add token-based auth in later iterations.

## Notes
- Frontend never needs to pass explicit pose ids for the initial generation.
- All generation requests create new immutable generation folders and return a `job_id` for polling.
