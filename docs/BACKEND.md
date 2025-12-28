1. Backend Role (Strict Definition)

The backend is the single source of truth for:

Pipeline orchestration

Deterministic execution

Artifact storage

Validation & retry logic

Frontend state derivation

The backend owns all decision-making.

The frontend:

Triggers jobs

Reads state

Displays artifacts

The backend:

Selects poses

Controls AI usage

Applies retries

Persists everything

2. Technology Stack

Framework: FastAPI

Language: Python 3.10+

Worker: Celery or RQ (pluggable)

Queue/State: Redis (local, optional)

Storage: Local filesystem

Image Processing:

OpenCV

Pillow

Metrics:

scikit-image (SSIM)

NumPy

AI Integration:

Adapter interface (local or API-based)

3. Backend Responsibilities
Area	Responsibility
API	Accept uploads, trigger jobs, expose artifacts
Orchestration	Enforce pipeline order
Determinism	Fixed pose mapping, seeded AI
Storage	Persist every artifact
Validation	Automated per-region checks
Retry	Controlled retry with failure injection
State	Job & generation metadata
4. High-Level Architecture
FastAPI
│
├── API Layer
│   ├── upload
│   ├── generate
│   ├── status
│   ├── gallery
│   └── artifacts
│
├── Pipeline Orchestrator
│   ├── isolate
│   ├── flatten
│   ├── extract
│   ├── compose
│   ├── validate
│   └── retry
│
├── AI Adapter Layer
│   ├── flatten_adapter
│   └── compose_adapter
│
├── Validation Engine
│   ├── ssim
│   ├── delta_e
│   └── pattern_match
│
└── Storage Layer
    └── filesystem

5. Pipeline Orchestration Model

Pipeline execution is sequential and stateful.

Each stage:

Has explicit input and output

Writes artifacts to disk

Updates job state

Cannot be skipped unless explicitly allowed

Failure at any stage:

Triggers retry logic

Or terminates job

6. Generation Modes (Locked)
Mode	Description
standard	Initial generation → poses 01–04
extend	Additional generation → poses 05–12
retry_failed	Re-run only failed outputs

Frontend never passes pose IDs.

Backend maps modes → poses internally.

7. Job State Model
{
  "job_id": "uuid",
  "saree_id": "uuid",
  "mode": "standard",
  "status": "queued | running | success | failed",
  "current_stage": "flatten",
  "retries": 1,
  "created_at": "...",
  "updated_at": "..."
}

8. Artifact Lifecycle

Every artifact is:

Written once

Immutable

Versioned by generation

Artifacts include:

Original upload

S_clean

S_flat

Extracted parts

Final outputs

Metrics JSON

Retry logs

Backend never overwrites artifacts.

9. Validation Engine
Per-Region Checks

SSIM

Color ΔE

Pattern similarity

Border continuity

Validation output:

{
  "region": "pallu",
  "ssim": 0.82,
  "delta_e": 3.1,
  "pattern_score": 0.46,
  "pass": true
}


Aggregate failures drive retry logic.

10. Retry Logic (Critical)

Maximum retries: 3

Each retry:

Logs failure reasons

Injects explicit constraints into AI prompt

Reuses deterministic inputs (S_clean, S_flat, parts)

Example injected instruction:

“Previous output failed due to border distortion. Preserve border geometry exactly.”

If all retries fail → job marked FAILED.

11. AI Adapter Contract

All AI usage must pass through adapters.

Adapter interface:

run(
  image,
  placement_map,
  seed,
  strict_instructions,
  forbidden_changes
)


Adapters must:

Accept a seed

Respect forbidden changes

Return deterministic output when possible

Log inputs and outputs

12. Storage Contract

Backend storage rules:

Filesystem paths are canonical

UI labels never depend on filenames

No cleanup in Round 1

All state is derivable from filesystem + job metadata

13. Error Handling

Backend errors are:

Categorized

Logged

Converted into user-safe messages

Frontend never receives:

Stack traces

Model errors

Raw AI output

14. Observability (Round 1)

Structured logs per job

Retry logs persisted

Metrics JSON per generation

No external monitoring in MVP.

15. Non-Goals (Explicit)

Backend will not:

Allow manual overrides

Expose AI prompts to frontend

Allow partial artifact deletion

Support concurrent mutation of a generation

Store data in cloud (Round 1)

16. Extension Points

Future-safe by design:

Replace AI models via adapters

Add new pose sets without frontend change

Add cloud storage later

Add auth and permissions later