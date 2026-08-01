"""
confidence_evaluator.py — Habeas Corpus OCR Pipeline
=====================================================
Module: ocr/confidence_evaluator.py  |  Stage 5

Responsibility:
    Run OCR on a single page image with retry logic.

    The evaluator makes up to MAX_ATTEMPTS attempts per page, escalating
    the preprocessing strategy and/or switching engines if confidence stays
    below the threshold.  Low-confidence pages are NEVER silently accepted —
    they are flagged in the result so the postprocessor and callers can
    make informed decisions.

    Retry schedule (per page):
        Attempt 1: primary engine + strategy recommended by QualityReport
        Attempt 2: primary engine + "aggressive" strategy (harder enhancement)
        Attempt 3: fallback engine + "aggressive" strategy

    If all three attempts are below threshold, the best result is kept and
    marked as low-confidence.

    Configuration via environment variable:
        OCR_CONFIDENCE_THRESHOLD=0.85   (default)

Dependencies:
    - ocr.ocr_engine
    - ocr.image_preprocessor
    - ocr.image_quality_analyzer
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from statistics import mean

from PIL import Image

from ocr.image_quality_analyzer import QualityReport
from ocr.ocr_engine import OCREngine, OCRLine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Pages with mean confidence below this are flagged as low-confidence.
#: Override via environment variable OCR_CONFIDENCE_THRESHOLD.
CONFIDENCE_THRESHOLD: float = float(
    os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.85")
)

MAX_ATTEMPTS: int = 3


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class PageOCRResult:
    """
    The result of OCR (with retries) for a single page.

    Attributes
    ----------
    page_num : int
    lines : list[OCRLine]
        All recognised lines from the winning attempt.
    confidence : float
        Mean confidence of the winning attempt [0.0, 1.0].
    attempt : int
        Which attempt succeeded (1, 2, or 3).
    strategy_used : str
        The preprocessing strategy used in the winning attempt.
    engine_used : str
        Name of the OCR engine used in the winning attempt.
    is_low_confidence : bool
        True if confidence stayed below CONFIDENCE_THRESHOLD after all retries.
    """
    page_num:           int
    lines:              list[OCRLine]
    confidence:         float
    attempt:            int
    strategy_used:      str
    engine_used:        str
    is_low_confidence:  bool = False

    @property
    def text(self) -> str:
        """Join all recognised lines into a single string."""
        return "\n".join(ln.text for ln in self.lines if ln.text.strip())

    def confidence_label(self) -> str:
        pct = int(self.confidence * 100)
        flag = " [LOW CONFIDENCE]" if self.is_low_confidence else ""
        return f"{pct}%{flag}"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _mean_confidence(lines: list[OCRLine]) -> float:
    if not lines:
        return 0.0
    return mean(ln.confidence for ln in lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_page(
    image: Image.Image,
    quality_report: QualityReport,
    primary_engine: OCREngine,
    fallback_engine: OCREngine,
    page_num: int = 1,
) -> PageOCRResult:
    """
    Run OCR on a single page with up to MAX_ATTEMPTS retry attempts.

    Parameters
    ----------
    image : PIL.Image
        Raw page image (unprocessed).
    quality_report : QualityReport
        From Stage 2.  Provides the initial recommended strategy.
    primary_engine : OCREngine
        Used for attempts 1 and 2.
    fallback_engine : OCREngine
        Used for attempt 3 if primary confidence is still too low.
    page_num : int
        1-based page index (for logging / result metadata).

    Returns
    -------
    PageOCRResult
    """
    # Lazy import here to avoid circular imports
    from ocr import image_preprocessor

    best_result: PageOCRResult | None = None

    attempt_schedule = [
        # (engine,           strategy)
        (primary_engine,  quality_report.recommended_strategy),
        (primary_engine,  "aggressive"),
        (fallback_engine, "aggressive"),
    ]

    for attempt_num, (engine, strategy) in enumerate(attempt_schedule, start=1):
        print(
            f"    [confidence_evaluator] Page {page_num} — "
            f"attempt {attempt_num}/{MAX_ATTEMPTS}  "
            f"engine={engine.name}  strategy={strategy}"
        )

        try:
            enhanced = image_preprocessor.apply(image, strategy)
            lines    = engine.run(enhanced)
            conf     = _mean_confidence(lines)
        except Exception as exc:
            print(f"    [confidence_evaluator] Attempt {attempt_num} FAILED: {exc}")
            conf  = 0.0
            lines = []

        result = PageOCRResult(
            page_num=page_num,
            lines=lines,
            confidence=conf,
            attempt=attempt_num,
            strategy_used=strategy,
            engine_used=engine.name,
        )

        print(f"    [confidence_evaluator] Confidence: {conf:.1%}")

        if conf >= CONFIDENCE_THRESHOLD:
            print(f"    [confidence_evaluator] Accepted (>= {CONFIDENCE_THRESHOLD:.0%})")
            return result

        # Keep the best attempt so far
        if best_result is None or conf > best_result.confidence:
            best_result = result

        if attempt_num < MAX_ATTEMPTS:
            print(
                f"    [confidence_evaluator] Below threshold "
                f"({CONFIDENCE_THRESHOLD:.0%}) — retrying..."
            )

    # All attempts exhausted — return best result with low-confidence flag
    assert best_result is not None
    best_result.is_low_confidence = True
    print(
        f"    [confidence_evaluator] WARNING: page {page_num} low-confidence "
        f"after {MAX_ATTEMPTS} attempts. Best: {best_result.confidence:.1%}"
    )
    return best_result
