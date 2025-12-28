# Pipeline details — per-stage contract

## Overview
Input: single saree floor-shot (JPEG/PNG)  
Output: one photorealistic image per requested pose + artifacts + metrics JSON

Stages:
1. **Saree Isolation (Background Removal)**
   - Input: `raw_saree.jpg`
   - Output: `S_clean.png` (RGBA)
   - Requirements: deterministic mask; same input → same output
   - Tools: classical segmentation (U2Net/Matting fallback) with deterministic parameters.

2. **AI-Assisted Flattening**
   - Input: `S_clean.png`
   - Output: `S_flat.png`
   - Purpose: remove folds & pose so layout slicing can be authoritative.
   - Allowed AI actions: smoothing, minor lighting normalization, planar warping.
   - Forbidden: motif redesign, border distortion, color shifts beyond ΔE tolerance.

3. **Layout-Guided Part Extraction**
   - Input: `S_flat.png`, `layout_reference.json`
   - Output: `parts/` containing `pallu.png`, `top_border.png`, `bottom_border.png`, `body.png` and `parts.json` (metadata)
   - Deterministic slicing with scale normalization.

4. **Model & Pose Selection**
   - Initial generation: programmatically mapped to poses 01–04 (standard views). Frontend does not expose pose selection or pose identifiers.
   - Additional generations (via "Generate More Views"): backend cycles through poses 05–08 (and then 01–04) with new random seeds to create endless variations.
   - Pose images are pre-captured assets with consistent lighting and framing.

5. **Two-Stage Composition**
   - Stage 1 (Optional): structure-layer with placeholder garment.
   - Stage 2: inject actual saree parts using placement maps and constrained AI warps for curvature + lighting adaptation.
   - Output: `final_poseXX.png`

6. **Validation + Retry**
   - Compute per-region metrics (SSIM, ΔE, pattern-match)
   - If any metric fails threshold → log failure reason and retry up to 3 times.
   - Each retry includes explicit instruction not to repeat prior error.

7. **Storage & Logging**
   - Persist all artifacts, metrics JSON, and retry logs.

## Retry logic
- Max retries: 3
- On error: append targeted instruction to AI prompt, e.g., "Previous output failed due to BORDER_DISTORTION. Preserve border geometry exactly; do not alter border width or motif spacing."
- On final failure: mark job as FAILED and return artifacts + failure reasons for manual review.
