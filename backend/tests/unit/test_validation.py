"""
Unit tests for validation metrics.
"""

import pytest
import numpy as np
import cv2

from app.pipeline.stages.validate import (
    compute_ssim,
    compute_mean_delta_e,
    compute_pattern_match_score,
    rgb_to_lab,
    compute_delta_e_ciede2000,
    determine_failure_reasons,
)
from app.models import FailureReason


def test_ssim_identical_images():
    """Test SSIM with identical images."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    ssim_score = compute_ssim(img, img.copy())
    
    assert ssim_score >= 0.99  # Should be very close to 1.0


def test_ssim_different_images():
    """Test SSIM with different images."""
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    ssim_score = compute_ssim(img1, img2)
    
    assert ssim_score < 0.1  # Should be very low


def test_ssim_similar_images():
    """Test SSIM with similar images (small noise)."""
    img1 = np.full((100, 100, 3), 128, dtype=np.uint8)
    img2 = img1.copy()
    
    # Add small noise
    noise = np.random.randint(-10, 10, img2.shape, dtype=np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    ssim_score = compute_ssim(img1, img2)
    
    assert 0.7 < ssim_score < 1.0


def test_rgb_to_lab():
    """Test RGB to LAB conversion."""
    # Test with known values
    # White
    white_rgb = np.array([255, 255, 255])
    white_lab = rgb_to_lab(white_rgb)
    assert abs(white_lab[0] - 100) < 1  # L should be ~100
    assert abs(white_lab[1]) < 1  # a should be ~0
    assert abs(white_lab[2]) < 1  # b should be ~0
    
    # Black
    black_rgb = np.array([0, 0, 0])
    black_lab = rgb_to_lab(black_rgb)
    assert abs(black_lab[0]) < 1  # L should be ~0


def test_delta_e_identical_colors():
    """Test Delta E with identical colors."""
    lab = np.array([50.0, 10.0, 20.0])
    
    delta_e = compute_delta_e_ciede2000(lab, lab.copy())
    
    assert delta_e < 0.01


def test_delta_e_different_colors():
    """Test Delta E with noticeably different colors."""
    lab1 = np.array([50.0, 0.0, 0.0])  # Gray
    lab2 = np.array([50.0, 30.0, 0.0])  # Reddish
    
    delta_e = compute_delta_e_ciede2000(lab1, lab2)
    
    # Should be a noticeable difference
    assert delta_e > 10


def test_mean_delta_e_identical_images():
    """Test mean Delta E with identical images."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    
    mean_de = compute_mean_delta_e(img, img.copy())
    
    assert mean_de < 0.1


def test_pattern_match_identical_images():
    """Test pattern match with identical images."""
    # Create an image with distinct features
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Add some patterns
    cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
    cv2.circle(img, (150, 150), 30, (128, 128, 128), -1)
    cv2.line(img, (0, 100), (200, 100), (64, 64, 64), 5)
    
    score = compute_pattern_match_score(img, img.copy())
    
    assert score >= 0.9  # Should have high match


def test_pattern_match_different_images():
    """Test pattern match with different images."""
    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img1, (20, 20), (80, 80), (255, 255, 255), -1)
    
    img2 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img2, (100, 100), 50, (128, 128, 128), -1)
    
    score = compute_pattern_match_score(img1, img2)
    
    # Should have low match
    assert score < 0.5


def test_determine_failure_reasons_color_shift():
    """Test failure reason detection for color shift."""
    reasons = determine_failure_reasons(
        ssim_val=0.85,
        delta_e=10.0,  # Above threshold
        pattern_score=0.5,
        ssim_threshold=0.80,
        delta_e_threshold=6.0,
        pattern_threshold=0.40,
    )
    
    assert FailureReason.COLOR_SHIFT in reasons


def test_determine_failure_reasons_pattern_loss():
    """Test failure reason detection for pattern loss."""
    reasons = determine_failure_reasons(
        ssim_val=0.85,
        delta_e=4.0,
        pattern_score=0.2,  # Below threshold
        ssim_threshold=0.80,
        delta_e_threshold=6.0,
        pattern_threshold=0.40,
    )
    
    assert FailureReason.PATTERN_LOSS in reasons


def test_determine_failure_reasons_ssim_low():
    """Test failure reason detection for low SSIM."""
    reasons = determine_failure_reasons(
        ssim_val=0.70,  # Below threshold
        delta_e=4.0,
        pattern_score=0.5,
        ssim_threshold=0.80,
        delta_e_threshold=6.0,
        pattern_threshold=0.40,
    )
    
    assert len(reasons) > 0
    assert any(r in reasons for r in [
        FailureReason.TEXTURE_SMOOTHING_EXCESS,
        FailureReason.ALIGNMENT_FAILURE
    ])


def test_determine_failure_reasons_all_pass():
    """Test that no failure reasons when all metrics pass."""
    reasons = determine_failure_reasons(
        ssim_val=0.90,
        delta_e=3.0,
        pattern_score=0.6,
        ssim_threshold=0.80,
        delta_e_threshold=6.0,
        pattern_threshold=0.40,
    )
    
    assert len(reasons) == 0
