"""
ocr_engine.py — Habeas Corpus OCR Pipeline
===========================================
Module: ocr/ocr_engine.py  |  Stage 4

Responsibility:
    Provide a stable, engine-agnostic OCR interface.

    The abstract base class (OCREngine) defines a single contract:
        run(image: PIL.Image) -> list[OCRLine]

    Concrete implementations wrap real OCR engines.  Swapping engines
    requires no changes anywhere else in the pipeline.

    Available engines (in preference order):
        PaddleOCREngine  — primary; excellent on legal documents
        EasyOCREngine    — fallback; broad Python compatibility
        MockEngine       — for unit testing / pipeline validation

    Engine selection:
        Use get_engine() to get the best available engine, or
        instantiate a specific engine class directly.

Dependencies:
    - paddleocr   (primary)
    - easyocr     (fallback)
    - Pillow
    - numpy
"""

from __future__ import annotations

import io
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

#: Default OCR languages for Indian legal documents.
#: Override via env var: OCR_LANGUAGES=en,ta,hi
_DEFAULT_LANGUAGES: list[str] = [
    lang.strip()
    for lang in os.getenv("OCR_LANGUAGES", "en,ta").split(",")
    if lang.strip()
]


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class OCRLine:
    """
    A single recognised text line from the OCR engine.

    Attributes
    ----------
    text : str
        The recognised text.
    confidence : float
        Confidence score in [0.0, 1.0].  1.0 = certain.
    bbox : tuple[int, int, int, int]
        Bounding box (x1, y1, x2, y2) in pixel coordinates.
        (0, 0, 0, 0) if the engine does not return bbox.
    """
    text:       str
    confidence: float
    bbox:       tuple[int, int, int, int] = field(default=(0, 0, 0, 0))

    def __post_init__(self) -> None:
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class OCREngine(ABC):
    """
    Abstract OCR engine interface.

    All concrete engines must implement ``run()`` and ``name``.
    They must NOT retain state between calls — each ``run()`` call is
    independent (safe for parallel use).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine identifier."""

    @abstractmethod
    def run(self, image: Image.Image) -> list[OCRLine]:
        """
        Perform OCR on a single page image.

        Parameters
        ----------
        image : PIL.Image
            Pre-processed page image (typically from image_preprocessor).

        Returns
        -------
        list[OCRLine]
            Recognised lines in top-to-bottom, left-to-right order.
            Returns [] if no text is detected.
        """

    def page_confidence(self, lines: list[OCRLine]) -> float:
        """Mean confidence across all recognised lines.  0.0 if no lines."""
        if not lines:
            return 0.0
        return float(np.mean([ln.confidence for ln in lines]))


# ---------------------------------------------------------------------------
# PaddleOCR engine (primary)
# ---------------------------------------------------------------------------

class PaddleOCREngine(OCREngine):
    """
    Primary OCR engine using PaddleOCR.

    PaddleOCR is lazy-loaded on first ``run()`` call.  The model is cached
    so subsequent pages pay no loading overhead.

    Configuration
    -------------
    lang : str
        Language code.  "en" for English.
    use_gpu : bool
        Set True if a CUDA GPU is available.
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False) -> None:
        self._lang    = lang
        self._use_gpu = use_gpu
        self._ocr     = None   # lazy

    @property
    def name(self) -> str:
        return "PaddleOCR"

    def _load(self):
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR  # type: ignore
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self._lang,
                use_gpu=self._use_gpu,
                show_log=False,
            )
            logger.info("[PaddleOCREngine] Model loaded.")
        except ImportError as exc:
            raise ImportError(
                "paddleocr is not installed. Run: uv add paddleocr"
            ) from exc

    def run(self, image: Image.Image) -> list[OCRLine]:
        self._load()

        # PaddleOCR accepts numpy arrays or file paths
        img_array = np.array(image.convert("RGB"))
        result    = self._ocr.ocr(img_array, cls=True)

        lines: list[OCRLine] = []
        if not result or result[0] is None:
            return lines

        for page_result in result:
            if page_result is None:
                continue
            for item in page_result:
                # item = [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, score)]
                bbox_pts, (text, score) = item
                xs = [pt[0] for pt in bbox_pts]
                ys = [pt[1] for pt in bbox_pts]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                lines.append(OCRLine(text=text, confidence=score, bbox=bbox))

        # Sort top-to-bottom, then left-to-right
        lines.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        return lines


# ---------------------------------------------------------------------------
# EasyOCR engine (fallback)
# ---------------------------------------------------------------------------

def _patch_easyocr_config() -> None:
    """Patch EasyOCR 1.7.2 config bugs (e.g. Tamil checkpoint shape mismatch)."""
    try:
        import easyocr.config
        gen1 = easyocr.config.recognition_models.get('gen1', {})
        if 'tamil_g1' in gen1:
            chars = gen1['tamil_g1']['characters']
            if len(chars) < 142:
                gen1['tamil_g1']['characters'] = chars + ('\u0000' * (142 - len(chars)))
    except Exception:
        pass

class EasyOCREngine(OCREngine):
    """
    Fallback OCR engine using EasyOCR.

    EasyOCR cannot mix Latin scripts (en, fr, ...) with Indic scripts
    (ta, hi, te, ...) in a single reader — they use separate recognition
    models with different output class sizes (127 vs 143+ classes).

    This engine handles it transparently via a dual-reader strategy:
      - Latin languages  → Reader A (english_g2 model, 127 classes)
      - Indic languages  → Reader B (tamil/devanagari model, 143+ classes)
    Results from both readers are merged and sorted by bounding box position.

    Configuration
    -------------
    languages : list[str]
        Language codes from OCR_LANGUAGES env var (e.g. ["en", "ta"]).
    gpu : bool
        Set True if CUDA is available.
    """

    # Indic/non-Latin scripts that must use separate readers from Latin
    _INDIC_SCRIPTS: frozenset[str] = frozenset({
        "ta", "hi", "te", "kn", "ml", "bn", "gu", "pa", "or", "ur",
        "si", "ne", "mr", "sa", "th", "ar", "fa", "he", "ja", "ko",
        "zh", "zh-TW",
    })

    def __init__(self, languages: list[str] | None = None, gpu: bool = False) -> None:
        self._languages     = languages or _DEFAULT_LANGUAGES
        self._gpu           = gpu
        self._reader_latin  = None   # lazy — Latin scripts
        self._reader_indic  = None   # lazy — Indic scripts

    @property
    def name(self) -> str:
        return "EasyOCR"

    def _split_languages(self) -> tuple[list[str], list[str]]:
        latin = [l for l in self._languages if l not in self._INDIC_SCRIPTS]
        indic = [l for l in self._languages if l in self._INDIC_SCRIPTS]
        return latin, indic

    def _load(self) -> None:
        try:
            import easyocr  # type: ignore
        except ImportError as exc:
            raise ImportError("easyocr is not installed. Run: uv add easyocr") from exc

        _patch_easyocr_config()

        latin, indic = self._split_languages()

        if latin and self._reader_latin is None:
            self._reader_latin = easyocr.Reader(latin, gpu=self._gpu, verbose=False)
            logger.info(f"[EasyOCREngine] Latin reader loaded: {latin}")

        if indic and self._reader_indic is None:
            self._reader_indic = easyocr.Reader(indic, gpu=self._gpu, verbose=False)
            logger.info(f"[EasyOCREngine] Indic reader loaded: {indic}")

    def _read_with(self, reader, img_array) -> list[OCRLine]:
        lines: list[OCRLine] = []
        for item in reader.readtext(img_array, detail=1):
            bbox_pts, text, score = item
            xs = [pt[0] for pt in bbox_pts]
            ys = [pt[1] for pt in bbox_pts]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            lines.append(OCRLine(text=text, confidence=score, bbox=bbox))
        return lines

    def run(self, image: Image.Image) -> list[OCRLine]:
        self._load()
        img_array = np.array(image.convert("RGB"))
        lines: list[OCRLine] = []

        if self._reader_latin:
            lines += self._read_with(self._reader_latin, img_array)
        if self._reader_indic:
            lines += self._read_with(self._reader_indic, img_array)

        lines.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        return lines


# ---------------------------------------------------------------------------
# Mock engine (unit testing / pipeline validation without real OCR)
# ---------------------------------------------------------------------------

class MockEngine(OCREngine):
    """
    Deterministic mock engine that returns a fixed response.

    Useful for testing the pipeline without loading heavy ML models.
    """

    def __init__(self, fixed_text: str = "Mock OCR output.", confidence: float = 0.95):
        self._text       = fixed_text
        self._confidence = confidence

    @property
    def name(self) -> str:
        return "MockOCR"

    def run(self, image: Image.Image) -> list[OCRLine]:
        return [OCRLine(
            text=self._text,
            confidence=self._confidence,
            bbox=(0, 0, image.width, image.height),
        )]


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def get_engine(prefer: str = "paddle") -> OCREngine:
    """
    Return the best available OCR engine.

    Parameters
    ----------
    prefer : "paddle" | "easy" | "mock"
        Preferred engine.  Falls back to the next available one.

    Returns
    -------
    OCREngine
    """
    if prefer == "mock":
        return MockEngine()

    if prefer in ("paddle", "auto"):
        try:
            import paddleocr  # noqa: F401
            return PaddleOCREngine()
        except ImportError:
            logger.warning("[ocr_engine] PaddleOCR not available, falling back to EasyOCR.")

    try:
        import easyocr  # noqa: F401
        return EasyOCREngine()
    except ImportError:
        pass

    logger.warning("[ocr_engine] No OCR engine available — returning MockEngine.")
    return MockEngine()
