# Validation, metrics and initial thresholds

## Per-region validation
Regions: `body`, `pallu`, `top_border`, `bottom_border`.

Metrics:
1. **SSIM** (Structural Similarity Index)
   - Purpose: detect structural loss (blurring, motif destruction).
   - Initial threshold (moderate): SSIM >= 0.80 per region.
   - After telemetry, tighten to SSIM >= 0.88 (example).

2. **Color ΔE (CIEDE2000)**
   - Purpose: detect unacceptable color shifts.
   - Initial threshold: ΔE <= 6 per region.
   - After iteration: ΔE <= 3–4.

3. **Pattern match score**
   - Purpose: ensure motifs/arrangements preserved (feature-match).
   - Implementation: ORB / SIFT-like keypoint matching (or deep feature cosine) between source part and rendered part.
   - Initial threshold: 40% matched keypoints; tune on dataset.

4. **Border consistency check**
   - Confirm border location and motif continuity using template correlation.

## Failure reason taxonomy (used for retry prompting)
- BORDER_DISTORTION
- COLOR_SHIFT
- PATTERN_LOSS
- TEXTURE_SMOOTHING_EXCESS
- ALIGNMENT_FAILURE
- SHADOW/PHOTOMETRIC_MISMATCH

## Logging
- `metrics.json` contains per-region metric values, pass/fail flags, and timestamp.
- `retry_log.json` contains reason and the prompt injection used for the next attempt.

## How to run evaluation locally
- `scripts/evaluate_single.py --source parts/pallu.png --rendered outputs/final_pallu.png`
- Writes `metrics.json` next to output.
