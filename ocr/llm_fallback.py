"""
llm_fallback.py — Habeas Corpus OCR Pipeline
=============================================
Module: ocr/llm_fallback.py  |  Stage 4b (LLM Vision Fallback)

Responsibility:
    When the OCR engine produces a low-confidence result for a page
    (typically handwritten or very degraded scans), send the original
    page image to Gemini 2.5 Flash via OpenRouter and return a corrected
    transcription, using the OCR draft as a hint.

    This is intentionally a *fallback* — it is only invoked when:
        (a) the page is scanned (not selectable text), AND
        (b) the best OCR confidence is below OCR_CONFIDENCE_THRESHOLD, AND
        (c) OPENROUTER_API_KEY is present in the environment

    If the API key is missing the function logs a warning and returns
    the original (low-confidence) OCR text unchanged.

Configuration (via .env / environment variables):
    OPENROUTER_API_KEY      — required (get one at openrouter.ai)
    LLM_FALLBACK_MODEL      — default: "google/gemini-2.5-flash"
    LLM_FALLBACK_MAX_TOKENS — default: 2048

Dependencies:
    httpx         (already installed as transitive dep)
    python-dotenv (uv add python-dotenv)
    Pillow

Usage:
    Called internally by ocr/pipeline.py — not intended for direct use.

    from ocr.llm_fallback import llm_ocr_page

    improved_text = llm_ocr_page(
        image=pil_image,
        ocr_draft="garbled OCR text...",
        page_num=1,
    )
"""

from __future__ import annotations

import base64
import io
import logging
import os

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL   = "google/gemini-2.5-flash"
_DEFAULT_TOKENS  = 2048

_PROMPT_TEMPLATE = """\
You are a precise legal document transcription assistant.

The image is a scanned page from an Indian legal document — it may be a \
court judgment, settlement memo, or legal form — containing both printed \
text and handwritten fills.

Your task:
1. Read ALL visible text in the image — both printed and handwritten.
2. Preserve the original layout: field labels on their own lines, values \
following them on the same or next line.
3. For handwritten text that is ambiguous, make your best contextual guess \
(e.g. if a label says "Case no." transcribe what looks like a case number).
4. Replace pure signatures and wholly illegible scribbles with [signature] \
or [illegible] respectively — never invent content.
5. Output ONLY the transcribed text. No commentary, preamble, or markdown.

The OCR engine already attempted this page and produced the draft below \
(the printed parts are mostly correct; the handwritten fills are garbled — \
use the draft as a structural guide):

--- OCR DRAFT START ---
{ocr_draft}
--- OCR DRAFT END ---

Transcribe the full page now:"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """Return True if an OpenRouter API key is configured."""
    _load_env_once()
    return bool(os.getenv("OPENROUTER_API_KEY"))


def llm_ocr_page(
    image: Image.Image,
    ocr_draft: str,
    page_num: int = 1,
) -> str:
    """
    Send a page image + OCR draft to Gemini 2.5 Flash via OpenRouter
    and return an improved transcription.

    Parameters
    ----------
    image : PIL.Image.Image
        The original (unprocessed) page image at full resolution.
    ocr_draft : str
        Best text the OCR engine produced. Used as a structural hint so
        the LLM focuses on correcting handwritten fills.
    page_num : int
        For logging only.

    Returns
    -------
    str
        Improved transcription, or ``ocr_draft`` unchanged if the API
        call fails for any reason.
    """
    _load_env_once()

    api_key    = os.getenv("OPENROUTER_API_KEY")
    model      = os.getenv("LLM_FALLBACK_MODEL", _DEFAULT_MODEL)
    max_tokens = int(os.getenv("LLM_FALLBACK_MAX_TOKENS", str(_DEFAULT_TOKENS)))

    if not api_key:
        logger.warning(
            "[llm_fallback] OPENROUTER_API_KEY not set — skipping LLM fallback "
            "for page %d. Add it to .env to enable.", page_num
        )
        return ocr_draft

    print(
        f"    [llm_fallback] Page {page_num} — sending to {model} "
        f"via OpenRouter for handwriting correction..."
    )

    try:
        import httpx

        b64_image = _pil_to_base64(image)
        prompt    = _PROMPT_TEMPLATE.format(
            ocr_draft=ocr_draft.strip() or "(no OCR output)"
        )

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.1,      # low = faithful transcription
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/habeas-corpus",
            "X-Title":       "Habeas Corpus Legal Engine",
        }

        response = httpx.post(
            _OPENROUTER_BASE,
            json=payload,
            headers=headers,
            timeout=120.0,   # vision calls can be slow
        )
        response.raise_for_status()

        data        = response.json()
        result_text = data["choices"][0]["message"]["content"].strip()
        char_gain   = len(result_text) - len(ocr_draft)

        print(
            f"    [llm_fallback] Page {page_num} — received {len(result_text):,} chars "
            f"({'+'  if char_gain >= 0 else ''}{char_gain} vs OCR draft)"
        )
        return result_text

    except Exception as exc:
        logger.error(
            "[llm_fallback] Page %d — OpenRouter call failed: %s. "
            "Returning OCR text as-is.", page_num, exc
        )
        return ocr_draft


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_env_loaded: bool = False

def _load_env_once() -> None:
    """Load .env once per process (idempotent). override=True ensures
    updated keys in .env are always picked up on next process start."""
    global _env_loaded
    if _env_loaded:
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass
    _env_loaded = True


def _pil_to_base64(image: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
