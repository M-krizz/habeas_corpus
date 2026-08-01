"""
legal_postprocessor.py — Habeas Corpus OCR Pipeline
====================================================
Module: ocr/legal_postprocessor.py  |  Stage 6

Responsibility:
    Clean up raw OCR output using legal-domain knowledge.

    Two-part pipeline:
        Part A — Legal correction dictionary
            Apply ~200 word-level substitutions for common OCR character
            confusions in Indian legal documents (l/I, rn/m, 0/O etc.).
            Applied as word-boundary regex substitutions — not naive
            string replace — to avoid false corrections inside other words.

        Part B — Layout reconstruction
            Merge OCR lines into paragraphs based on vertical proximity.
            Detect and preserve section headings.
            Remove duplicate lines (OCR sometimes double-reads headers).
            Normalise whitespace throughout.

    Additionally, low-confidence pages receive a visible annotation
    (e.g. "[LOW_CONFIDENCE: 72%]") so downstream processing knows to
    treat that page's text with caution.

Configuration:
    Correction dictionary path defaults to resources/legal_corrections.json
    relative to the project root.  Override via env var LEGAL_CORRECTIONS_PATH.

Dependencies:
    - ocr.confidence_evaluator (PageOCRResult, OCRLine)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ocr.confidence_evaluator import PageOCRResult
from ocr.ocr_engine import OCRLine


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent

_DEFAULT_CORRECTIONS_PATH = _PROJECT_ROOT / "resources" / "legal_corrections.json"

#: Lines whose normalised text appears more than this many times in the
#: full document are treated as repeated headers/footers and removed.
_DUPLICATE_THRESHOLD: int = 3

#: Two OCR lines are merged into the same paragraph if their vertical gap
#: (in pixels) is less than this fraction of the average line height.
_LINE_MERGE_GAP_FACTOR: float = 0.8


# ---------------------------------------------------------------------------
# Correction dictionary loader (cached)
# ---------------------------------------------------------------------------

_CORRECTIONS_CACHE: dict[str, str] | None = None


def _load_corrections() -> dict[str, str]:
    global _CORRECTIONS_CACHE
    if _CORRECTIONS_CACHE is not None:
        return _CORRECTIONS_CACHE

    path = Path(os.getenv("LEGAL_CORRECTIONS_PATH", str(_DEFAULT_CORRECTIONS_PATH)))
    if not path.exists():
        _CORRECTIONS_CACHE = {}
        return _CORRECTIONS_CACHE

    with open(path, encoding="utf-8") as fh:
        raw: dict[str, str] = json.load(fh)

    _CORRECTIONS_CACHE = raw
    return _CORRECTIONS_CACHE


# ---------------------------------------------------------------------------
# Part A — Legal correction dictionary
# ---------------------------------------------------------------------------

def _apply_corrections(text: str) -> str:
    """
    Apply word-boundary regex substitutions from the legal corrections dict.

    Each key in the dict is a regex pattern (word-boundary anchored).
    Values are literal replacement strings.
    """
    corrections = _load_corrections()
    for pattern, replacement in corrections.items():
        try:
            text = re.sub(r"\b" + pattern + r"\b", replacement, text)
        except re.error:
            # If the key is not a valid regex, fall back to literal replace
            text = text.replace(pattern, replacement)
    return text


# ---------------------------------------------------------------------------
# Part B — Layout reconstruction
# ---------------------------------------------------------------------------

def _is_heading(text: str) -> bool:
    """
    Heuristic: a line is a section heading if it is ALL CAPS and
    shorter than 80 characters.  This preserves headings like
    "JUDGMENT", "FACTS", "HELD:" etc.
    """
    stripped = text.strip()
    if not stripped:
        return False
    return stripped.isupper() and len(stripped) < 80


def _reconstruct_paragraphs(lines: list[OCRLine]) -> str:
    """
    Merge OCR lines into natural paragraphs based on vertical proximity.

    Lines close together (small vertical gap) are joined with a space.
    Lines far apart (large vertical gap) start a new paragraph.
    """
    if not lines:
        return ""

    # Estimate average line height from bounding boxes
    heights = [ln.bbox[3] - ln.bbox[1] for ln in lines if ln.bbox[3] > ln.bbox[1]]
    avg_height = sum(heights) / len(heights) if heights else 20
    gap_threshold = avg_height * _LINE_MERGE_GAP_FACTOR

    paragraphs: list[list[str]] = []
    current: list[str] = []
    prev_bottom = None

    for ln in lines:
        text = ln.text.strip()
        if not text:
            continue

        top = ln.bbox[1]
        bottom = ln.bbox[3]

        if prev_bottom is None:
            # First line
            current.append(text)
        else:
            gap = top - prev_bottom
            if gap > gap_threshold or _is_heading(text):
                # Start a new paragraph
                if current:
                    paragraphs.append(current)
                current = [text]
            else:
                current.append(text)

        prev_bottom = bottom

    if current:
        paragraphs.append(current)

    # Join each paragraph
    para_strings = []
    for para in paragraphs:
        joined = " ".join(para)
        para_strings.append(joined)

    return "\n\n".join(para_strings)


def _remove_duplicate_lines(text: str) -> str:
    """
    Remove lines that appear suspiciously often (repeated headers/footers
    that the OCR engine picks up on every page).
    """
    lines = text.splitlines()
    # Count normalised occurrences
    normalised_counts: dict[str, int] = {}
    for line in lines:
        key = re.sub(r"\s+", " ", line).strip().lower()
        if key:
            normalised_counts[key] = normalised_counts.get(key, 0) + 1

    # Filter
    result = []
    for line in lines:
        key = re.sub(r"\s+", " ", line).strip().lower()
        if key and normalised_counts.get(key, 0) > _DUPLICATE_THRESHOLD:
            continue
        result.append(line)

    return "\n".join(result)


def _normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces; preserve paragraph breaks."""
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse spaces within lines
    lines = [re.sub(r" {2,}", " ", line) for line in text.splitlines()]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Low-confidence annotation
# ---------------------------------------------------------------------------

def _annotate_low_confidence(text: str, page_num: int, confidence: float) -> str:
    """Prepend a visible warning to low-confidence page text."""
    pct = int(confidence * 100)
    banner = f"[LOW_CONFIDENCE: {pct}% — Page {page_num} — verify manually]"
    return f"{banner}\n{text}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def postprocess_page(result: PageOCRResult) -> str:
    """
    Apply the full postprocessing pipeline to a single page OCR result.

    Parameters
    ----------
    result : PageOCRResult
        From Stage 5 (confidence_evaluator).

    Returns
    -------
    str
        Clean, corrected text for this page.
    """
    # Reconstruct layout from OCR lines
    text = _reconstruct_paragraphs(result.lines)

    # Apply legal correction dictionary
    text = _apply_corrections(text)

    # Normalise whitespace
    text = _normalise_whitespace(text)

    # Annotate low-confidence pages — never silently accept
    if result.is_low_confidence:
        text = _annotate_low_confidence(text, result.page_num, result.confidence)

    return text


def postprocess_document(page_results: list[PageOCRResult]) -> str:
    """
    Postprocess all pages and assemble a single clean document string.

    Parameters
    ----------
    page_results : list[PageOCRResult]
        All page results from Stage 5, in page order.

    Returns
    -------
    str
        Full document text, pages separated by double newlines.
    """
    page_texts = [postprocess_page(r) for r in page_results]

    # Join pages
    full_text = "\n\n".join(pt for pt in page_texts if pt.strip())

    # Remove cross-page duplicate lines (repeated headers/footers)
    full_text = _remove_duplicate_lines(full_text)

    # Final normalisation
    full_text = _normalise_whitespace(full_text)

    return full_text
