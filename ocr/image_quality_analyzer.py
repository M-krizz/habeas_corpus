"""
image_quality_analyzer.py — Habeas Corpus OCR Pipeline
=======================================================
Module: ocr/image_quality_analyzer.py  |  Stage 2

Responsibility:
    Analyse a scanned page image and return a QualityReport that tells
    Stage 3 (image_preprocessor) which enhancement strategy to apply.

    Metrics computed:
        - Blur         : Laplacian variance  (< 100 → blurry)
        - Skew angle   : Hough line transform (> 0.5° → needs deskew)
        - Contrast     : Pixel std deviation  (< 40  → low contrast)
        - Noise level  : High-frequency energy in the Fourier domain

    Strategy decision:
        - "standard"   : Normal scanned page
        - "aggressive" : Blurry / low-contrast / high-noise (fax, photocopy)
        - "gentle"     : Already clean — risk of over-processing

Dependencies:
    - opencv-python (cv2)
    - numpy
    - Pillow
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Thresholds — tune these per your document corpus
# ---------------------------------------------------------------------------

BLUR_THRESHOLD:       float = 100.0   # Laplacian variance below this = blurry
SKEW_THRESHOLD:       float = 0.5     # degrees absolute — above = deskew needed
CONTRAST_THRESHOLD:   float = 40.0    # pixel std deviation below = low contrast
NOISE_THRESHOLD:      float = 0.35    # high-freq energy ratio above = noisy


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Per-page image quality metrics and recommended preprocessing strategy."""
    blur_score:            float
    skew_angle:            float    # degrees; positive = counter-clockwise
    contrast_score:        float
    noise_level:           float
    is_blurry:             bool
    has_significant_skew:  bool
    has_low_contrast:      bool
    is_noisy:              bool
    recommended_strategy:  Literal["standard", "aggressive", "gentle"]

    def summary(self) -> str:
        flags = []
        if self.is_blurry:            flags.append("blurry")
        if self.has_significant_skew: flags.append(f"skewed {self.skew_angle:.1f}°")
        if self.has_low_contrast:     flags.append("low-contrast")
        if self.is_noisy:             flags.append("noisy")
        flag_str = ", ".join(flags) if flags else "clean"
        return (
            f"strategy={self.recommended_strategy}  "
            f"blur={self.blur_score:.1f}  "
            f"contrast={self.contrast_score:.1f}  "
            f"noise={self.noise_level:.2f}  "
            f"[{flag_str}]"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_gray(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV grayscale array."""
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)


def _compute_blur(gray: np.ndarray) -> float:
    """
    Laplacian variance — the standard blur metric.
    Higher = sharper.  A totally blurry image scores near 0.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _compute_skew(gray: np.ndarray) -> float:
    """
    Estimate page skew angle using the Hough line transform.

    Returns the dominant angle in degrees relative to horizontal.
    A well-aligned page returns ≈ 0.0.
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        # OpenCV 4: line[0] = [x1,y1,x2,y2]; OpenCV 5: line = [x1,y1,x2,y2]
        coords = line[0] if line.ndim == 2 else line
        x1, y1, x2, y2 = coords
        if x2 != x1:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Keep only near-horizontal lines (within ±45°)
            if abs(angle) < 45:
                angles.append(angle)

    if not angles:
        return 0.0

    # Median angle is more robust than mean against outliers
    return float(np.median(angles))


def _compute_contrast(gray: np.ndarray) -> float:
    """Standard deviation of pixel intensities — higher = more contrast."""
    return float(gray.std())


def _compute_noise(gray: np.ndarray) -> float:
    """
    High-frequency energy ratio via 2D FFT.

    A clean document has most energy at low frequencies (text strokes).
    A noisy scan has significant high-frequency energy.
    Returns a ratio in [0, 1]; higher = noisier.
    """
    # Downsample to speed up FFT — quality doesn't need full resolution
    small = cv2.resize(gray, (512, 512))
    f     = np.fft.fft2(small.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w  = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 8   # inner circle = low frequency zone

    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2

    low_energy  = magnitude[mask].sum()
    total_energy = magnitude.sum()

    if total_energy == 0:
        return 0.0

    high_energy_ratio = 1.0 - (low_energy / total_energy)
    return float(np.clip(high_energy_ratio, 0.0, 1.0))


def _decide_strategy(
    is_blurry: bool,
    has_low_contrast: bool,
    is_noisy: bool,
) -> Literal["standard", "aggressive", "gentle"]:
    """
    Choose a preprocessing strategy based on the quality flags.

    aggressive: page has serious problems — use heavy-handed enhancement
    gentle:     page is already good — minimal processing to avoid artifacts
    standard:   default for normal scanned documents
    """
    problem_count = sum([is_blurry, has_low_contrast, is_noisy])
    if problem_count >= 2:
        return "aggressive"
    if problem_count == 0:
        return "gentle"
    return "standard"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(image: Image.Image) -> QualityReport:
    """
    Analyse image quality and return a QualityReport.

    Parameters
    ----------
    image : PIL.Image
        A single document page image (any mode; converted internally).

    Returns
    -------
    QualityReport
    """
    gray = _pil_to_gray(image)

    blur_score     = _compute_blur(gray)
    skew_angle     = _compute_skew(gray)
    contrast_score = _compute_contrast(gray)
    noise_level    = _compute_noise(gray)

    is_blurry            = blur_score     < BLUR_THRESHOLD
    has_significant_skew = abs(skew_angle) > SKEW_THRESHOLD
    has_low_contrast     = contrast_score  < CONTRAST_THRESHOLD
    is_noisy             = noise_level     > NOISE_THRESHOLD

    strategy = _decide_strategy(is_blurry, has_low_contrast, is_noisy)

    return QualityReport(
        blur_score=blur_score,
        skew_angle=skew_angle,
        contrast_score=contrast_score,
        noise_level=noise_level,
        is_blurry=is_blurry,
        has_significant_skew=has_significant_skew,
        has_low_contrast=has_low_contrast,
        is_noisy=is_noisy,
        recommended_strategy=strategy,
    )
