"""
image_preprocessor.py — Habeas Corpus OCR Pipeline
====================================================
Module: ocr/image_preprocessor.py  |  Stage 3

Responsibility:
    Apply OpenCV enhancement to a scanned page image using the strategy
    recommended by Stage 2 (image_quality_analyzer).

    Three named strategies:

    standard   Normal scanned page — grayscale, deskew, denoise, OTSU
    aggressive Fax / photocopy / blurry — bilateral filter, CLAHE,
               adaptive threshold, sharpen, morphological cleanup
    gentle     Already-clean page — minimal processing to avoid artifacts

    Each strategy is a standalone function (PIL Image → PIL Image), so
    OCR engines are completely decoupled from preprocessing details.
    Swapping strategies or adding a fourth one requires no changes outside
    this module.

Dependencies:
    - opencv-python (cv2)
    - numpy
    - Pillow
"""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Strategy = Literal["standard", "aggressive", "gentle"]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _gray_to_pil(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(gray)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _deskew(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate the image to correct the skew angle.

    Uses a filled-white border so rotated corners do not produce black
    triangles that confuse the OCR engine.
    """
    if abs(angle_deg) < 0.3:
        return gray   # Not worth rotating for tiny angles

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,   # white border
    )
    return rotated


def _estimate_skew(gray: np.ndarray) -> float:
    """Quick local skew estimate for use within the preprocessor."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 100, 10)
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        coords = line[0] if line.ndim == 2 else line
        x1, y1, x2, y2 = coords
        if x2 != x1:
            a = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(a) < 45:
                angles.append(a)
    return float(np.median(angles)) if angles else 0.0


def _sharpen(gray: np.ndarray) -> np.ndarray:
    """Apply an unsharp-mask sharpening kernel."""
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ], dtype=np.float32)
    return cv2.filter2D(gray, -1, kernel)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _strategy_standard(image: Image.Image) -> Image.Image:
    """
    Standard strategy — suitable for most clean scanned documents.

    Pipeline:
        BGR → Grayscale → Deskew → fastNlMeans denoise → OTSU threshold
        → Morphological close (fill small gaps in characters)
    """
    bgr  = _pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Deskew
    angle = _estimate_skew(gray)
    gray  = _deskew(gray, angle)

    # Denoise (preserves edges better than Gaussian)
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Global OTSU binarisation
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close small gaps in strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return _gray_to_pil(binary)


def _strategy_aggressive(image: Image.Image) -> Image.Image:
    """
    Aggressive strategy — for blurry, low-contrast, or heavily noisy pages
    (old faxes, photocopies, degraded paper).

    Pipeline:
        BGR → Grayscale → Deskew → Bilateral filter (edge-preserving smooth)
        → CLAHE (local contrast enhancement) → Adaptive threshold
        → Sharpen → Dilate (thicken thin strokes) → Morphological cleanup
    """
    bgr  = _pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Deskew
    angle = _estimate_skew(gray)
    gray  = _deskew(gray, angle)

    # Bilateral filter — smooths noise while keeping text edges sharp
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # CLAHE — Contrast Limited Adaptive Histogram Equalisation
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # Adaptive threshold — handles uneven illumination (e.g., shadow across page)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8,
    )

    # Sharpen
    binary = _sharpen(binary)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    # Dilate slightly to thicken thin/broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Final morphological close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return _gray_to_pil(binary)


def _strategy_gentle(image: Image.Image) -> Image.Image:
    """
    Gentle strategy — for already-clean pages.

    Minimal processing reduces the risk of introducing artifacts that
    confuse the OCR engine (e.g., over-binarising a clean scan creates
    broken character strokes).

    Pipeline:
        BGR → Grayscale → Deskew → Light Gaussian blur → OTSU threshold
    """
    bgr  = _pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Deskew
    angle = _estimate_skew(gray)
    gray  = _deskew(gray, angle)

    # Very gentle blur to remove JPEG/scanner compression artefacts
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # OTSU binarisation
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return _gray_to_pil(binary)


# ---------------------------------------------------------------------------
# Strategy registry — adding a new strategy only requires:
#   1. Write a function _strategy_<name>(image) -> Image
#   2. Add it here
# ---------------------------------------------------------------------------

_STRATEGIES: dict[str, callable] = {
    "standard":  _strategy_standard,
    "aggressive": _strategy_aggressive,
    "gentle":    _strategy_gentle,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply(image: Image.Image, strategy: Strategy = "standard") -> Image.Image:
    """
    Apply an enhancement strategy to a scanned page image.

    Parameters
    ----------
    image : PIL.Image
        Raw page image from Stage 1 (pdf_renderer).
    strategy : "standard" | "aggressive" | "gentle"
        Enhancement strategy — typically from QualityReport.recommended_strategy.

    Returns
    -------
    PIL.Image
        Enhanced image ready for the OCR engine.

    Raises
    ------
    ValueError
        If strategy is not one of the known strategies.
    """
    fn = _STRATEGIES.get(strategy)
    if fn is None:
        known = ", ".join(sorted(_STRATEGIES))
        raise ValueError(
            f"[image_preprocessor] Unknown strategy '{strategy}'. "
            f"Known: {known}"
        )
    return fn(image)


def available_strategies() -> list[str]:
    """Return the names of all registered preprocessing strategies."""
    return sorted(_STRATEGIES.keys())
