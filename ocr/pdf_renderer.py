"""
pdf_renderer.py — Habeas Corpus OCR Pipeline
=============================================
Module: ocr/pdf_renderer.py  |  Stage 1

Responsibility:
    - Accept any supported input (PDF, PNG, JPG, TIFF, BMP, JPEG).
    - For PDFs: inspect each page for selectable text.
        • If the text layer has ≥ MIN_SELECTABLE_CHARS characters, extract
          directly via fitz (fast path — zero OCR cost).
        • If the page is essentially blank text (scanned), render it as a
          300 DPI PIL Image for the OCR pipeline.
    - For image files: load directly as a PIL Image.
    - Return a list of PageInput objects preserving page order.

Design principle:
    This module has no knowledge of OCR or image enhancement.
    It only answers: "What raw material do I hand to the next stage?"

Dependencies:
    - PyMuPDF (fitz)  — already in pyproject.toml
    - Pillow          — uv add pillow
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: A page with fewer than this many characters is treated as scanned.
MIN_SELECTABLE_CHARS: int = 50

#: Render DPI for scanned pages.  300 is the OCR industry standard.
RENDER_DPI: int = 300

#: Supported direct-image extensions.
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PageInput:
    """
    Represents a single page of a document ready for the next pipeline stage.

    Attributes
    ----------
    page_num : int
        1-based page index.
    mode : "text" | "image"
        ``"text"`` — selectable text extracted directly; no OCR needed.
        ``"image"`` — scanned page rendered as a PIL Image for OCR.
    text : str
        Populated when mode == "text".  Empty string otherwise.
    image : PIL.Image | None
        Populated when mode == "image".  None for text pages.
    source_path : str
        Absolute path to the original file.
    """
    page_num:    int
    mode:        Literal["text", "image"]
    text:        str           = ""
    image:       Image.Image | None = field(default=None, repr=False)
    source_path: str           = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fitz_page_to_pil(page: fitz.Page, dpi: int = RENDER_DPI) -> Image.Image:
    """
    Render a fitz page to a PIL Image at the specified DPI.

    fitz's default coordinate space is 72 DPI, so we scale by dpi/72.
    """
    scale  = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    # pixmap.tobytes("png") returns PNG bytes; PIL can load those directly
    return Image.open(io.BytesIO(pixmap.tobytes("png")))


def _is_selectable(page: fitz.Page) -> bool:
    """Return True if the page has enough selectable text to skip OCR."""
    text = page.get_text("text")
    return len(text.strip()) >= MIN_SELECTABLE_CHARS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_document(source_path: str | Path) -> list[PageInput]:
    """
    Inspect and render a document, returning per-page inputs.

    Parameters
    ----------
    source_path : str | Path
        Path to a PDF or supported image file.

    Returns
    -------
    list[PageInput]
        One entry per page, in document order.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"[pdf_renderer] File not found: {path}")

    suffix = path.suffix.lower()

    # ---- Image file: single "page" ----------------------------------------
    if suffix in IMAGE_EXTENSIONS:
        img = Image.open(path).convert("RGB")
        return [PageInput(
            page_num=1,
            mode="image",
            image=img,
            source_path=str(path),
        )]

    # ---- PDF file -----------------------------------------------------------
    if suffix == ".pdf":
        return _render_pdf(path)

    raise ValueError(
        f"[pdf_renderer] Unsupported file type: '{suffix}'. "
        f"Supported: .pdf, {', '.join(sorted(IMAGE_EXTENSIONS))}"
    )


def _render_pdf(path: Path) -> list[PageInput]:
    """Internal: process all pages of a PDF."""
    pages: list[PageInput] = []

    with fitz.open(str(path)) as doc:
        total = len(doc)
        print(f"[pdf_renderer] '{path.name}' — {total} page(s)")

        for page_num, page in enumerate(doc, start=1):
            if _is_selectable(page):
                text = page.get_text("text")
                pages.append(PageInput(
                    page_num=page_num,
                    mode="text",
                    text=text,
                    source_path=str(path),
                ))
                print(f"  Page {page_num:>3}: selectable text "
                      f"({len(text.strip()):,} chars)")
            else:
                img = _fitz_page_to_pil(page, dpi=RENDER_DPI)
                pages.append(PageInput(
                    page_num=page_num,
                    mode="image",
                    image=img,
                    source_path=str(path),
                ))
                print(f"  Page {page_num:>3}: scanned — rendered at {RENDER_DPI} DPI "
                      f"({img.width}x{img.height}px)")

    text_count  = sum(1 for p in pages if p.mode == "text")
    image_count = sum(1 for p in pages if p.mode == "image")
    print(f"[pdf_renderer] {text_count} text page(s), {image_count} scanned page(s)")
    return pages


# ---------------------------------------------------------------------------
# Entry point — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr/pdf_renderer.py <path_to_pdf_or_image>")
        sys.exit(0)
    pages = render_document(sys.argv[1])
    print(f"\nTotal pages returned: {len(pages)}")
    for p in pages:
        if p.mode == "text":
            print(f"  [{p.page_num}] TEXT  — {len(p.text):,} chars")
        else:
            print(f"  [{p.page_num}] IMAGE — {p.image.size}")
