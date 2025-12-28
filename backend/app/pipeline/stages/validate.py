"""
Validation Stage.

Computes per-region validation metrics (SSIM, CIEDE2000 ΔE, pattern match).
Produces metrics.json per generation with pass/fail flags.
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from app.core.config import settings
from app.models import FailureReason

logger = logging.getLogger(__name__)


def compute_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index between two images.
    
    Args:
        image1: First image (will be converted to grayscale)
        image2: Second image (will be converted to grayscale)
        
    Returns:
        SSIM value between 0 and 1
    """
    # Convert to grayscale if needed
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = image1
        
    if len(image2.shape) == 3:
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = image2
    
    # Resize to match if needed
    if gray1.shape != gray2.shape:
        h = min(gray1.shape[0], gray2.shape[0])
        w = min(gray1.shape[1], gray2.shape[1])
        gray1 = cv2.resize(gray1, (w, h))
        gray2 = cv2.resize(gray2, (w, h))
    
    # Compute SSIM
    score, _ = ssim(gray1, gray2, full=True)
    
    return float(score)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB values to LAB color space."""
    # Normalize to 0-1
    rgb_normalized = rgb.astype(float) / 255.0
    
    # Apply gamma correction
    mask = rgb_normalized > 0.04045
    rgb_linear = np.where(mask, ((rgb_normalized + 0.055) / 1.055) ** 2.4, rgb_normalized / 12.92)
    
    # Convert to XYZ
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    xyz = np.dot(rgb_linear, matrix.T)
    
    # Reference white (D65)
    ref_white = np.array([0.95047, 1.00000, 1.08883])
    xyz_normalized = xyz / ref_white
    
    # Convert to LAB
    epsilon = 0.008856
    kappa = 903.3
    
    mask = xyz_normalized > epsilon
    f = np.where(mask, xyz_normalized ** (1/3), (kappa * xyz_normalized + 16) / 116)
    
    L = 116 * f[1] - 16
    a = 500 * (f[0] - f[1])
    b = 200 * (f[1] - f[2])
    
    return np.array([L, a, b])


def compute_delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """
    Compute CIEDE2000 color difference.
    
    Simplified implementation - for production, use colormath library.
    
    Args:
        lab1: First LAB color [L, a, b]
        lab2: Second LAB color [L, a, b]
        
    Returns:
        Delta E value (0 = identical, >6 = noticeable difference)
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    # Calculate C1, C2, C_avg
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2
    
    # G factor
    C_avg_7 = C_avg ** 7
    G = 0.5 * (1 - math.sqrt(C_avg_7 / (C_avg_7 + 25**7)))
    
    # Adjusted a values
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    # New chroma values
    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)
    
    # Hue angles
    h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360
    h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360
    
    # Delta values
    delta_L = L2 - L1
    delta_C = C2_prime - C1_prime
    
    # Delta h
    delta_h_prime = h2_prime - h1_prime
    if abs(delta_h_prime) > 180:
        if delta_h_prime > 0:
            delta_h_prime -= 360
        else:
            delta_h_prime += 360
    
    delta_H = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime / 2))
    
    # Average values
    L_avg = (L1 + L2) / 2
    C_avg_prime = (C1_prime + C2_prime) / 2
    
    if abs(C1_prime * C2_prime) == 0:
        h_avg = h1_prime + h2_prime
    elif abs(h1_prime - h2_prime) <= 180:
        h_avg = (h1_prime + h2_prime) / 2
    else:
        if h1_prime + h2_prime < 360:
            h_avg = (h1_prime + h2_prime + 360) / 2
        else:
            h_avg = (h1_prime + h2_prime - 360) / 2
    
    # Weighting functions
    T = (1 - 0.17 * math.cos(math.radians(h_avg - 30)) +
         0.24 * math.cos(math.radians(2 * h_avg)) +
         0.32 * math.cos(math.radians(3 * h_avg + 6)) -
         0.20 * math.cos(math.radians(4 * h_avg - 63)))
    
    S_L = 1 + (0.015 * (L_avg - 50)**2) / math.sqrt(20 + (L_avg - 50)**2)
    S_C = 1 + 0.045 * C_avg_prime
    S_H = 1 + 0.015 * C_avg_prime * T
    
    # Rotation term
    delta_theta = 30 * math.exp(-((h_avg - 275) / 25)**2)
    C_avg_prime_7 = C_avg_prime ** 7
    R_C = 2 * math.sqrt(C_avg_prime_7 / (C_avg_prime_7 + 25**7))
    R_T = -R_C * math.sin(math.radians(2 * delta_theta))
    
    # Final calculation
    delta_E = math.sqrt(
        (delta_L / S_L)**2 +
        (delta_C / S_C)**2 +
        (delta_H / S_H)**2 +
        R_T * (delta_C / S_C) * (delta_H / S_H)
    )
    
    return delta_E


def compute_mean_delta_e(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute mean CIEDE2000 ΔE between two images.
    
    Args:
        image1: First image (BGR)
        image2: Second image (BGR)
        
    Returns:
        Mean Delta E across sampled pixels
    """
    # Resize to match if needed
    if image1.shape[:2] != image2.shape[:2]:
        h = min(image1.shape[0], image2.shape[0])
        w = min(image1.shape[1], image2.shape[1])
        image1 = cv2.resize(image1, (w, h))
        image2 = cv2.resize(image2, (w, h))
    
    # Sample pixels (computing for all pixels is expensive)
    h, w = image1.shape[:2]
    sample_step = max(1, min(h, w) // 50)  # ~50x50 = 2500 samples
    
    delta_es = []
    
    for y in range(0, h, sample_step):
        for x in range(0, w, sample_step):
            # Get RGB values (OpenCV uses BGR)
            rgb1 = image1[y, x, :3][::-1]
            rgb2 = image2[y, x, :3][::-1]
            
            lab1 = rgb_to_lab(rgb1)
            lab2 = rgb_to_lab(rgb2)
            
            delta_e = compute_delta_e_ciede2000(lab1, lab2)
            delta_es.append(delta_e)
    
    return float(np.mean(delta_es)) if delta_es else 0.0


def compute_pattern_match_score(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute pattern match score using ORB keypoints.
    
    Args:
        image1: Source image (reference)
        image2: Generated image
        
    Returns:
        Match score between 0 and 1 (ratio of matched keypoints)
    """
    # Convert to grayscale
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = image1
        
    if len(image2.shape) == 3:
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = image2
    
    # Resize to match if needed
    if gray1.shape != gray2.shape:
        h = min(gray1.shape[0], gray2.shape[0])
        w = min(gray1.shape[1], gray2.shape[1])
        gray1 = cv2.resize(gray1, (w, h))
        gray2 = cv2.resize(gray2, (w, h))
    
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=500)
    
    # Detect and compute features
    kp1, desc1 = orb.detectAndCompute(gray1, None)
    kp2, desc2 = orb.detectAndCompute(gray2, None)
    
    if desc1 is None or desc2 is None:
        return 0.0
    
    if len(kp1) == 0 or len(kp2) == 0:
        return 0.0
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    
    # Calculate match ratio
    max_possible = min(len(kp1), len(kp2))
    if max_possible == 0:
        return 0.0
    
    # Filter good matches (distance threshold)
    good_matches = [m for m in matches if m.distance < 50]
    
    match_ratio = len(good_matches) / max_possible
    
    return float(min(1.0, match_ratio))


def determine_failure_reasons(
    ssim_val: float,
    delta_e: float,
    pattern_score: float,
    ssim_threshold: float,
    delta_e_threshold: float,
    pattern_threshold: float,
) -> list[FailureReason]:
    """Determine failure reasons based on metrics."""
    reasons = []
    
    if ssim_val < ssim_threshold:
        # Low SSIM can indicate various structural issues
        if ssim_val < ssim_threshold * 0.9:
            reasons.append(FailureReason.TEXTURE_SMOOTHING_EXCESS)
        else:
            reasons.append(FailureReason.ALIGNMENT_FAILURE)
    
    if delta_e > delta_e_threshold:
        reasons.append(FailureReason.COLOR_SHIFT)
    
    if pattern_score < pattern_threshold:
        reasons.append(FailureReason.PATTERN_LOSS)
        # Could also indicate border issues
        if pattern_score < pattern_threshold * 0.5:
            reasons.append(FailureReason.BORDER_DISTORTION)
    
    return reasons


def validate_region(
    source_image: np.ndarray,
    generated_image: np.ndarray,
    region_name: str,
    ssim_threshold: float,
    delta_e_threshold: float,
    pattern_threshold: float,
) -> dict[str, Any]:
    """
    Validate a single region.
    
    Args:
        source_image: Original saree part
        generated_image: Rendered region from final image
        region_name: Name of the region
        ssim_threshold: SSIM pass threshold
        delta_e_threshold: Max acceptable Delta E
        pattern_threshold: Minimum pattern match score
        
    Returns:
        Validation result dict
    """
    # Compute metrics
    ssim_val = compute_ssim(source_image, generated_image)
    delta_e = compute_mean_delta_e(source_image, generated_image)
    pattern_score = compute_pattern_match_score(source_image, generated_image)
    
    # Determine pass/fail
    ssim_pass = ssim_val >= ssim_threshold
    delta_e_pass = delta_e <= delta_e_threshold
    pattern_pass = pattern_score >= pattern_threshold
    
    overall_pass = ssim_pass and delta_e_pass and pattern_pass
    
    # Get failure reasons if any metric failed
    failure_reasons = []
    if not overall_pass:
        failure_reasons = determine_failure_reasons(
            ssim_val, delta_e, pattern_score,
            ssim_threshold, delta_e_threshold, pattern_threshold
        )
    
    return {
        "region": region_name,
        "ssim": round(ssim_val, 4),
        "delta_e": round(delta_e, 2),
        "pattern_score": round(pattern_score, 4),
        "ssim_pass": ssim_pass,
        "delta_e_pass": delta_e_pass,
        "pattern_pass": pattern_pass,
        "passed": overall_pass,
        "failure_reasons": [r.value for r in failure_reasons],
    }


def validate_pose(
    saree_id: str,
    pose_id: str,
    generation_name: str,
    ssim_threshold: Optional[float] = None,
    delta_e_threshold: Optional[float] = None,
    pattern_threshold: Optional[float] = None,
) -> dict[str, Any]:
    """
    Validate a single pose output.
    
    Compares the generated image regions against source parts.
    
    Args:
        saree_id: Saree identifier
        pose_id: Pose identifier
        generation_name: Generation folder name
        ssim_threshold: Override SSIM threshold
        delta_e_threshold: Override Delta E threshold
        pattern_threshold: Override pattern threshold
        
    Returns:
        Validation result with per-region metrics
    """
    from app.core.storage import get_artifact_path, get_generation_dir, get_parts_dir
    
    # Use config thresholds if not overridden
    ssim_threshold = ssim_threshold or settings.SSIM_THRESHOLD
    delta_e_threshold = delta_e_threshold or settings.DELTA_E_THRESHOLD
    pattern_threshold = pattern_threshold or settings.PATTERN_MATCH_THRESHOLD
    
    logger.info(f"Validating {pose_id} for saree {saree_id}")
    
    # Get the generated image
    view_number = pose_id.split("_")[1]
    gen_dir = get_generation_dir(saree_id, generation_name)
    generated_path = gen_dir / f"final_view_{view_number}.png"
    
    if not generated_path.exists():
        return {
            "pose_id": pose_id,
            "status": "error",
            "error": f"Generated image not found: {generated_path}",
            "passed": False,
        }
    
    generated = cv2.imread(str(generated_path))
    if generated is None:
        return {
            "pose_id": pose_id,
            "status": "error",
            "error": f"Could not load generated image: {generated_path}",
            "passed": False,
        }
    
    # Get source parts for comparison
    parts_dir = get_parts_dir(saree_id)
    
    # Define regions to validate
    regions = ["main_body", "pallu", "top_border", "bottom_border"]
    region_results = []
    overall_passed = True
    all_failure_reasons = []
    
    for region in regions:
        part_path = parts_dir / f"{region}.png"
        
        if not part_path.exists():
            logger.warning(f"Part not found: {part_path}")
            continue
        
        source_part = cv2.imread(str(part_path))
        if source_part is None:
            logger.warning(f"Could not load part: {part_path}")
            continue
        
        # For mock validation, use the generated image directly
        # In production, extract the corresponding region from the generated image
        # based on placement maps
        
        result = validate_region(
            source_image=source_part,
            generated_image=generated,  # Simplified: compare full image
            region_name=region,
            ssim_threshold=ssim_threshold,
            delta_e_threshold=delta_e_threshold,
            pattern_threshold=pattern_threshold,
        )
        
        region_results.append(result)
        
        if not result["passed"]:
            overall_passed = False
            all_failure_reasons.extend(result["failure_reasons"])
    
    return {
        "pose_id": pose_id,
        "timestamp": datetime.utcnow().isoformat(),
        "regions": region_results,
        "passed": overall_passed,
        "failure_reasons": list(set(all_failure_reasons)),  # Deduplicate
    }


def run_validate_stage(
    saree_id: str,
    job_id: str,
    generation_name: str,
    poses: list[str],
) -> dict:
    """
    Run validation for all poses in a generation.
    
    Args:
        saree_id: Saree identifier
        job_id: Job identifier (for logging)
        generation_name: Generation folder name
        poses: List of pose IDs to validate
        
    Returns:
        dict with validation results and metrics.json path
    """
    from app.core.storage import save_json, get_generation_dir
    
    logger.info(f"[{job_id}] Running validation for {len(poses)} poses")
    
    results = {}
    passed_count = 0
    failed_count = 0
    all_failure_reasons: dict[str, list[str]] = {}
    
    for pose_id in poses:
        result = validate_pose(saree_id, pose_id, generation_name)
        results[pose_id] = result
        
        if result.get("passed"):
            passed_count += 1
        else:
            failed_count += 1
            failure_reasons = result.get("failure_reasons", [])
            if failure_reasons:
                all_failure_reasons[pose_id] = failure_reasons
    
    # Compile metrics.json
    metrics_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "generation_name": generation_name,
        "thresholds": {
            "ssim": settings.SSIM_THRESHOLD,
            "delta_e": settings.DELTA_E_THRESHOLD,
            "pattern_match": settings.PATTERN_MATCH_THRESHOLD,
        },
        "summary": {
            "total_poses": len(poses),
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": round(passed_count / len(poses), 2) if poses else 0,
        },
        "poses": results,
    }
    
    # Save metrics.json
    save_json(saree_id, "metrics.json", metrics_data, generation_name)
    
    logger.info(f"[{job_id}] Validation complete: {passed_count}/{len(poses)} passed")
    
    return {
        "status": "success" if failed_count == 0 else "partial" if passed_count > 0 else "failed",
        "passed_count": passed_count,
        "failed_count": failed_count,
        "results": results,
        "failure_reasons": all_failure_reasons,
    }
