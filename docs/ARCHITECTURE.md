# System architecture (high-level)

## Components
1. **Frontend (Next.js)** — upload, gallery and immutable generation inspector. Initial generation is locked to 4 standard views; pose identifiers are not exposed to users. Users may request additional views later through a single "Generate More Views" action.
2. **Backend (FastAPI)** — orchestrates the pipeline, exposes REST API, persists job metadata, triggers worker jobs.
3. **Worker (Celery / RQ)** — runs the deterministic pipeline stages (segmentation → flatten → extract → compose → validate).
4. **AI Wrappers** — controlled adapters for any AI used (image flattening, warping/lighting adaptation). They accept strict prompts and constraints, log prompts and responses, and accept a seed parameter when supported.
5. **Assets Storage (local)** — deterministic storage of all intermediate artifacts (S_clean, S_flat, parts, final images) and JSON metrics/logs.
6. **Evaluation Module** — computes SSIM, color ΔE, and pattern-match scores per region.

## Data flow
1. User uploads floor-shot saree image → `POST /api/upload`.
2. Backend creates job, stores original, and enqueues worker job.
3. Worker:
   - Background removal (S_clean) - deterministic algorithm.
   - AI-assisted flattening (S_flat) - constrained AI call.
   - Layout-guided part extraction — deterministic slice using fixed layout reference.
   - Compose: select fixed pose asset, run two-stage composition and constrained AI warping.
   - Validate; if failure → retry up to 3 times with failure injection.
4. Artifacts and metrics saved to local storage; final image made available via API.

## Determinism controls
- Fixed model assets and fixed pose assets.
- Deterministic segmentation and slicing.
- Seeding of AI wrapper calls where applicable; strict prompt templates that forbid motif/border changes.
- No randomness in placement maps; warping constrained by precomputed maps per pose.
