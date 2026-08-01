"""
pipeline.py — Habeas Corpus OCR Pipeline
=========================================
Module: ocr/pipeline.py  |  Top-level Orchestrator

Responsibility:
    Single entry point for all document text extraction.

    Auto-detects whether a document has selectable text:
        - Selectable PDF → fast path via fitz (no OCR cost)
        - Scanned PDF / image → full 6-stage OCR pipeline

    This module replaces pdf_reader.py as the entry point.
    pdf_reader.py is still used internally for the fast path.

    Stages invoked for scanned documents:
        Stage 1: pdf_renderer       → PageInput list
        Stage 2: image_quality_analyzer → QualityReport per page
        Stage 3: image_preprocessor → enhanced PIL Image
        Stage 4: ocr_engine         → OCRLine list
        Stage 5: confidence_evaluator → PageOCRResult (with retry)
        Stage 6: legal_postprocessor → clean text string

Configuration (via environment variables):
    OCR_ENGINE          = "paddle" | "easy" | "mock"  (default: "paddle")
    OCR_CONFIDENCE_THRESHOLD = 0.85                   (default)
    LEGAL_CORRECTIONS_PATH   = path to JSON            (default: resources/)

Dependencies:
    All ocr.* sub-modules, plus fitz (PyMuPDF) for the fast path.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from ocr.confidence_evaluator import PageOCRResult, evaluate_page
from ocr.image_quality_analyzer import QualityReport, analyze
from ocr.legal_postprocessor import postprocess_document
from ocr.ocr_engine import OCREngine, get_engine
from ocr.pdf_renderer import PageInput, render_document

# Load .env so OPENROUTER_API_KEY is available before config is read
load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ENGINE_PREF: str = os.getenv("OCR_ENGINE", "paddle")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PageDetail:
    """Full detail for a single processed page."""
    page_num:          int
    mode:              str          # "text" | "ocr"
    text:              str
    confidence:        float | None  # None for text pages
    is_low_confidence: bool
    strategy_used:     str | None
    engine_used:       str | None
    quality_report:    QualityReport | None = field(default=None, repr=False)


@dataclass
class DocumentResult:
    """
    Result of processing a single document through the full pipeline.

    Attributes
    ----------
    text : str
        Final clean document text (one string).
    source_path : str
    mode : "selectable" | "ocr" | "mixed"
        "selectable" — all pages had text layers.
        "ocr"        — all pages went through OCR.
        "mixed"      — some pages selectable, some scanned.
    pages : list[PageDetail]
        Per-page detail including confidence scores.
    avg_ocr_confidence : float | None
        Mean confidence across OCR'd pages.  None if mode == "selectable".
    low_confidence_pages : list[int]
        1-based page numbers flagged as low confidence.
    elapsed_seconds : float
    """
    text:                  str
    source_path:           str
    mode:                  str
    pages:                 list[PageDetail]
    avg_ocr_confidence:    float | None
    low_confidence_pages:  list[int]
    elapsed_seconds:       float

    def summary(self) -> str:
        lines = [
            f"Source:     {self.source_path}",
            f"Mode:       {self.mode}",
            f"Pages:      {len(self.pages)}",
            f"Characters: {len(self.text):,}",
            f"Elapsed:    {self.elapsed_seconds:.1f}s",
        ]
        if self.avg_ocr_confidence is not None:
            lines.append(f"OCR conf:   {self.avg_ocr_confidence:.1%}")
        if self.low_confidence_pages:
            lines.append(f"Low-conf pages: {self.low_confidence_pages}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# OCRConfig
# ---------------------------------------------------------------------------

@dataclass
class OCRConfig:
    """Runtime configuration for the OCR pipeline."""
    engine_preference:   str   = _ENGINE_PREF
    use_gpu:             bool  = False
    fallback_to_easyocr: bool  = True
    #: Send low-confidence pages to Gemini Vision.
    #: Automatically disabled if GEMINI_API_KEY is not set.
    use_llm_fallback:    bool  = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_engines(config: OCRConfig) -> tuple[OCREngine, OCREngine]:
    """Build primary and fallback engines from config."""
    primary  = get_engine(prefer=config.engine_preference)
    fallback = get_engine(prefer="easy") if config.fallback_to_easyocr else primary
    return primary, fallback


def _process_text_page(page_input: PageInput) -> PageDetail:
    """Fast path: return selectable text without OCR."""
    return PageDetail(
        page_num=page_input.page_num,
        mode="text",
        text=page_input.text,
        confidence=None,
        is_low_confidence=False,
        strategy_used=None,
        engine_used=None,
    )


def _process_image_page(
    page_input: PageInput,
    primary: OCREngine,
    fallback: OCREngine,
) -> PageDetail:
    """OCR path: analyse, enhance, OCR, evaluate confidence."""
    assert page_input.image is not None

    # Stage 2 — quality analysis
    qreport = analyze(page_input.image)
    print(
        f"  [quality] Page {page_input.page_num}: "
        f"{qreport.summary()}"
    )

    # Stages 3-5 — preprocess, OCR, confidence check (with retries)
    ocr_result: PageOCRResult = evaluate_page(
        image=page_input.image,
        quality_report=qreport,
        primary_engine=primary,
        fallback_engine=fallback,
        page_num=page_input.page_num,
    )

    # Stage 6 — postprocess this page
    clean_text = _postprocess_single(ocr_result)

    return PageDetail(
        page_num=page_input.page_num,
        mode="ocr",
        text=clean_text,
        confidence=ocr_result.confidence,
        is_low_confidence=ocr_result.is_low_confidence,
        strategy_used=ocr_result.strategy_used,
        engine_used=ocr_result.engine_used,
        quality_report=qreport,
    )


def _postprocess_single(ocr_result: PageOCRResult) -> str:
    """Run Stage 6 for a single page."""
    from ocr.legal_postprocessor import postprocess_page
    return postprocess_page(ocr_result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(
    source_path: str | Path,
    config: OCRConfig | None = None,
) -> DocumentResult:
    """
    Extract clean text from any supported document.

    Parameters
    ----------
    source_path : str | Path
        Path to a PDF, PNG, JPG, TIFF, or BMP file.
    config : OCRConfig | None
        Pipeline configuration.  Defaults are used if None.

    Returns
    -------
    DocumentResult
        Contains the final text, per-page detail, and metadata.
    """
    config = config or OCRConfig()
    path   = Path(source_path)
    start  = time.perf_counter()

    print(f"\n[ocr/pipeline] Processing: {path.name}")
    print("=" * 60)

    # Stage 1 — render document
    page_inputs: list[PageInput] = render_document(path)

    # Build engines (lazy — models not loaded until first .run() call)
    primary, fallback = _build_engines(config)

    text_pages  = [p for p in page_inputs if p.mode == "text"]
    image_pages = [p for p in page_inputs if p.mode == "image"]

    # Determine overall mode
    if not image_pages:
        overall_mode = "selectable"
    elif not text_pages:
        overall_mode = "ocr"
    else:
        overall_mode = "mixed"

    print(
        f"[ocr/pipeline] Mode: {overall_mode}  "
        f"({len(text_pages)} text, {len(image_pages)} scanned)"
    )

    # Process all pages
    page_details: list[PageDetail] = []
    ocr_results_for_postprocessing: list[PageOCRResult] = []

    # Decide if LLM fallback should be used this run
    from ocr.llm_fallback import is_available as llm_available, llm_ocr_page
    _use_llm = config.use_llm_fallback and llm_available()
    if config.use_llm_fallback and not llm_available():
        print("[ocr/pipeline] LLM fallback disabled — GEMINI_API_KEY not set.")

    for page_input in page_inputs:
        if page_input.mode == "text":
            detail = _process_text_page(page_input)
            page_details.append(detail)
        else:
            print(f"\n[ocr/pipeline] OCR: page {page_input.page_num}")
            detail = _process_image_page(page_input, primary, fallback)

            # LLM fallback: if OCR confidence is still low, ask Gemini Vision
            if detail.is_low_confidence and _use_llm:
                print(
                    f"    [ocr/pipeline] Page {page_input.page_num} low-confidence "
                    f"({detail.confidence:.0%}) — invoking LLM fallback"
                )
                assert page_input.image is not None
                improved = llm_ocr_page(
                    image=page_input.image,
                    ocr_draft=detail.text,
                    page_num=page_input.page_num,
                )
                # Replace the text; keep confidence metadata honest
                detail = PageDetail(
                    page_num=detail.page_num,
                    mode="ocr",
                    text=improved,
                    confidence=detail.confidence,
                    is_low_confidence=True,    # still flagged so callers know
                    strategy_used=detail.strategy_used,
                    engine_used=f"{detail.engine_used}+llm",
                    quality_report=detail.quality_report,
                )

            page_details.append(detail)

    # Assemble final text — text pages join naturally, OCR pages already postprocessed
    all_page_texts = [d.text for d in sorted(page_details, key=lambda d: d.page_num)]
    final_text = "\n\n".join(t for t in all_page_texts if t.strip())

    # If any OCR pages, run cross-page duplicate removal
    if image_pages:
        from ocr.legal_postprocessor import _remove_duplicate_lines, _normalise_whitespace
        final_text = _remove_duplicate_lines(final_text)
        final_text = _normalise_whitespace(final_text)

    # Compute OCR stats
    ocr_details = [d for d in page_details if d.mode == "ocr"]
    avg_conf    = (
        sum(d.confidence for d in ocr_details) / len(ocr_details)
        if ocr_details else None
    )
    low_conf_pages = [d.page_num for d in ocr_details if d.is_low_confidence]

    elapsed = time.perf_counter() - start

    result = DocumentResult(
        text=final_text,
        source_path=str(path),
        mode=overall_mode,
        pages=page_details,
        avg_ocr_confidence=avg_conf,
        low_confidence_pages=low_conf_pages,
        elapsed_seconds=elapsed,
    )

    print(f"\n[ocr/pipeline] Done.")
    print(result.summary())
    return result


# Alias for backward compatibility with upload endpoints
process_document = extract_text


# ---------------------------------------------------------------------------
# Entry point — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Reconfigure stdout for utf-8 on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    if len(sys.argv) < 2:
        print("Usage: python ocr/pipeline.py <path_to_pdf_or_image> [--engine paddle|easy|mock]")
        sys.exit(0)

    path_arg = sys.argv[1]
    engine_arg = "paddle"
    for i, arg in enumerate(sys.argv):
        if arg == "--engine" and i + 1 < len(sys.argv):
            engine_arg = sys.argv[i + 1]

    cfg = OCRConfig(engine_preference=engine_arg)
    doc = extract_text(path_arg, config=cfg)

    print("\n--- First 1000 characters of extracted text ---")
    try:
        print(doc.text[:1000])
    except UnicodeEncodeError:
        print(doc.text[:1000].encode("ascii", errors="backslashreplace").decode("ascii"))

    if len(doc.text) > 1000:
        print(f"\n... [{len(doc.text) - 1000:,} more characters] ...")
