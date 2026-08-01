"""
clean_text.py — Habeas Corpus Legal Search Engine
===================================================
Module: extractor/clean_text.py

Responsibility:
    - Accept raw text extracted by pdf_reader.py (or any similar extractor).
    - Remove typographic noise without destroying legal content:
        1. Collapse repeated inline spaces.
        2. Collapse repeated blank lines.
        3. Strip bare page numbers (standalone digits / simple labels).
        4. Detect and remove headers/footers that recur across pages.
    - Return a single cleaned string ready for downstream NLP or indexing.

Design notes:
    - Pages are delimited by the form-feed character (\f) that pdf_reader.py
      inserts between pages.  The cleaner respects this boundary so that per-
      page heuristics (header/footer detection) work correctly, then joins
      pages back together with a double newline.
    - All cleaning steps are exposed as individual functions so callers can
      mix-and-match or write their own pipeline.
    - No external dependencies — only the Python standard library.

Usage:
    Import and call the top-level function:

        from extractor.clean_text import clean_document

        cleaned = clean_document(raw_text)

    Or run directly to smoke-test on a .txt file produced by pdf_reader.py:

        python extractor/clean_text.py output/judgment_001.txt
"""

import re
import sys
from collections import Counter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A line is considered a "bare page number" if, after stripping whitespace,
# it matches one of these patterns.
_PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\d+$"),                           # "42"
    re.compile(r"^-\s*\d+\s*-$"),                  # "- 42 -"
    re.compile(r"^page\s+\d+(\s+of\s+\d+)?$",      # "Page 42" / "Page 42 of 100"
               re.IGNORECASE),
    re.compile(r"^\[\s*\d+\s*\]$"),                 # "[42]"
]

# pdf_reader.py inserts these annotation headers; strip them before any other
# processing so they do not pollute header/footer detection.
_READER_HEADER_PATTERN = re.compile(
    r"^---\s*Page\s+\d+\s+of\s+\d+\s*---$", re.IGNORECASE
)

# How many lines at the top / bottom of each page to inspect when looking for
# repeated headers and footers.
_HEADER_SCAN_LINES = 3
_FOOTER_SCAN_LINES = 3

# A candidate line must appear in at least this fraction of pages to be
# treated as a recurring header/footer and removed.
_REPETITION_THRESHOLD = 0.4  # 40 % of pages


# ---------------------------------------------------------------------------
# Step 1 — Strip pdf_reader annotation headers
# ---------------------------------------------------------------------------

def strip_reader_annotations(text: str) -> str:
    """
    Remove the ``--- Page N of M ---`` markers inserted by pdf_reader.py.

    These are internal bookkeeping markers, not document content.  They must
    be removed before header/footer detection so they do not skew line counts.

    Parameters
    ----------
    text : str
        Raw text as returned by ``pdf_reader.extract_text_from_pdf()``.

    Returns
    -------
    str
        Text with annotation lines removed.
    """
    cleaned_lines = [
        line for line in text.splitlines()
        if not _READER_HEADER_PATTERN.match(line.strip())
    ]
    return "\n".join(cleaned_lines)


# ---------------------------------------------------------------------------
# Step 2 — Collapse repeated spaces
# ---------------------------------------------------------------------------

def collapse_spaces(text: str) -> str:
    """
    Replace runs of two or more non-newline whitespace characters with a
    single space on each line.

    Tabs are first normalised to spaces.  This prevents words from running
    together while keeping line breaks intact.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 3 — Collapse repeated blank lines
# ---------------------------------------------------------------------------

def collapse_blank_lines(text: str, max_consecutive: int = 1) -> str:
    """
    Reduce runs of more than ``max_consecutive`` blank lines to exactly that
    many blank lines.

    Legal documents often have large vertical gaps between sections after
    extraction; this keeps single blank lines as paragraph separators while
    removing excessive whitespace.

    Parameters
    ----------
    text : str
    max_consecutive : int
        Maximum number of consecutive blank lines to allow (default ``1``).

    Returns
    -------
    str
    """
    pattern = re.compile(r"(\n[ \t]*){" + str(max_consecutive + 1) + r",}")
    replacement = "\n" * (max_consecutive + 1)
    return pattern.sub(replacement, text)


# ---------------------------------------------------------------------------
# Step 4 — Remove standalone page numbers
# ---------------------------------------------------------------------------

def remove_page_numbers(text: str) -> str:
    """
    Remove lines that consist solely of a page number or simple page label.

    The patterns recognised are listed in ``_PAGE_NUMBER_PATTERNS``.  Lines
    with additional textual content (e.g., "See page 3 for details.") are
    intentionally left untouched.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(pat.match(stripped) for pat in _PAGE_NUMBER_PATTERNS):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


# ---------------------------------------------------------------------------
# Step 5 — Detect and remove repeated headers / footers
# ---------------------------------------------------------------------------

def _normalise_candidate(line: str) -> str:
    """Return a normalised form of a line for comparison purposes."""
    return re.sub(r"\s+", " ", line).strip().lower()


def detect_repeated_headers_footers(pages: list[str]) -> set[str]:
    """
    Identify lines that appear near the top or bottom of many pages.

    Each page's first ``_HEADER_SCAN_LINES`` non-empty lines and last
    ``_FOOTER_SCAN_LINES`` non-empty lines are treated as candidates.  A
    candidate is flagged as a recurring header/footer if it appears in at
    least ``_REPETITION_THRESHOLD * len(pages)`` pages.

    Parameters
    ----------
    pages : list[str]
        List of per-page text strings (already stripped of annotation headers).

    Returns
    -------
    set[str]
        Normalised (lowercased, whitespace-collapsed) strings that should be
        removed everywhere they appear as standalone lines.
    """
    if not pages:
        return set()

    min_occurrences = max(2, int(len(pages) * _REPETITION_THRESHOLD))
    candidate_counter: Counter = Counter()

    for page_text in pages:
        non_empty = [l for l in page_text.splitlines() if l.strip()]

        head_candidates = non_empty[:_HEADER_SCAN_LINES]
        tail_candidates = (
            non_empty[-_FOOTER_SCAN_LINES:]
            if len(non_empty) > _HEADER_SCAN_LINES
            else []
        )

        seen_on_this_page: set[str] = set()
        for line in head_candidates + tail_candidates:
            norm = _normalise_candidate(line)
            if norm and norm not in seen_on_this_page:
                candidate_counter[norm] += 1
                seen_on_this_page.add(norm)

    repeated = {
        norm for norm, count in candidate_counter.items()
        if count >= min_occurrences
    }
    return repeated


def remove_repeated_headers_footers(pages: list[str]) -> list[str]:
    """
    Strip recurring header/footer lines from every page.

    Parameters
    ----------
    pages : list[str]
        Per-page text strings.

    Returns
    -------
    list[str]
        Pages with recurring header/footer lines removed.
    """
    repeated = detect_repeated_headers_footers(pages)
    if not repeated:
        return pages

    cleaned_pages = []
    for page_text in pages:
        cleaned_lines = [
            line for line in page_text.splitlines()
            if _normalise_candidate(line) not in repeated
        ]
        cleaned_pages.append("\n".join(cleaned_lines))

    return cleaned_pages


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def clean_document(raw_text: str) -> str:
    """
    Run the full cleaning pipeline on text extracted from a single PDF.

    Pipeline order:
        1. Strip pdf_reader annotation headers (``--- Page N of M ---``).
        2. Split on form-feed (``\f``) to get per-page strings.
        3. Remove recurring headers / footers across pages.
        4. Rejoin pages, then apply document-level cleaning:
            a. Collapse repeated inline spaces.
            b. Remove standalone page numbers.
            c. Collapse repeated blank lines.
        5. Strip leading/trailing whitespace from the final result.

    Parameters
    ----------
    raw_text : str
        Text as returned by ``pdf_reader.extract_text_from_pdf()``, with
        pages separated by ``\f`` characters.

    Returns
    -------
    str
        Cleaned, normalised document text ready for NLP or indexing.
    """
    if not raw_text or not raw_text.strip():
        return ""

    # --- Step 1: strip internal annotation markers ---
    text = strip_reader_annotations(raw_text)

    # --- Step 2: split into pages ---
    pages = text.split("\f")

    # --- Step 3: remove recurring headers / footers ---
    pages = remove_repeated_headers_footers(pages)

    # --- Step 4: rejoin pages with a clear separator ---
    text = "\n\n".join(pages)

    # --- Step 4a: collapse inline spaces ---
    text = collapse_spaces(text)

    # --- Step 4b: remove bare page numbers ---
    text = remove_page_numbers(text)

    # --- Step 4c: collapse blank lines ---
    text = collapse_blank_lines(text, max_consecutive=1)

    # --- Step 5: final trim ---
    return text.strip()


# ---------------------------------------------------------------------------
# Entry Point — smoke-test on a single .txt file
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extractor/clean_text.py <path_to_extracted_txt>")
        print()
        print("  Reads a .txt file produced by pdf_reader.py, runs the full")
        print("  cleaning pipeline, and prints the first 1 000 characters.")
        sys.exit(0)

    input_path = sys.argv[1]
    try:
        with open(input_path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except FileNotFoundError:
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)

    cleaned = clean_document(raw)

    char_before = len(raw)
    char_after  = len(cleaned)
    reduction   = (1 - char_after / char_before) * 100 if char_before else 0

    print(f"[clean_text] Input : {char_before:,} characters")
    print(f"[clean_text] Output: {char_after:,} characters  ({reduction:.1f}% reduction)")
    print("-" * 60)
    print(cleaned[:1_000])
    if len(cleaned) > 1_000:
        print(f"\n... [{len(cleaned) - 1_000:,} more characters] ...")
