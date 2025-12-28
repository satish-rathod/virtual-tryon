# Storage layout (local filesystem)

Root storage: `storage/`

Per-saree directory: `storage/<saree_id>/`
- `original.jpg`                       # uploaded image
- `S_clean.png`                         # RGBA background-removed
- `S_flat.png`                          # flattened output
- `parts/`
  - `pallu.png`
  - `top_border.png`
  - `bottom_border.png`
  - `body.png`
  - `parts.json`                        # widths, heights, dominant LAB, texture descriptors
- `assets/`                             # chosen pose asset copy used for this job
- `generations/`
  - `gen_01_standard/`
    - `final_view_01.png` (pose_01 internally)
    - `final_view_02.png`
    - `final_view_03.png`
    - `final_view_04.png`
    - `metrics.json`
    - `retry_log.json`
  - `gen_02_extend/`
    - `final_view_05.png` (pose_05 internally)
    - ...
- `metrics.json`                         # summary metrics for saree-level operations
- `job.json`                             # job metadata, timestamps, status

## Filename conventions
- Use ISO8601 timestamps in names where needed: `final_view_01_2025-12-27T14:33:10Z.png`
- Use UUID for saree and job identification.
- UI-visible names use neutral labels ("View 1", "View 2") rather than pose IDs.

## Retention policy
- Round 1: retain all artifacts locally for manual QA.
- Future: implement TTL & archival, and optional deletions via an admin interface.
