"""
pdf_reader.py — Habeas Corpus Legal Search Engine
===================================================
Module: extractor/pdf_reader.py

Responsibility:
    - Scan the sc_English/ folder for all PDF files.
    - Extract selectable (digital) text from each PDF using PyMuPDF (fitz).
    - Preserve page order during extraction.
    - Return the extracted text as a Python string.
    - Save each extracted text to output/<filename>.txt.

Dependencies:
    - PyMuPDF  (pip install pymupdf)

Usage:
    Run directly for a quick extraction of all PDFs:
        python extractor/pdf_reader.py

    Or import and call from other modules:
        from extractor.pdf_reader import extract_text_from_pdf, process_all_pdfs
"""

import os
import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Constants — adjust paths here if the project layout ever changes
# ---------------------------------------------------------------------------

# Root of the project (one level above this file: kc/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder that holds the source PDFs
PDF_INPUT_DIR = os.path.join(PROJECT_ROOT, "sc_English")

# Folder where extracted .txt files will be saved
TEXT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract selectable text from a single PDF file using PyMuPDF.

    Pages are iterated in order (page 0 → last page).  Only the digitally
    embedded text layer is read — no OCR is performed.  If a page has no
    selectable text (e.g., a scanned image page), an empty string is
    contributed for that page.

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to the PDF file.

    Returns
    -------
    str
        Full document text, with pages separated by a form-feed character
        (``\\f``) so callers can split on ``\\f`` to recover individual pages.

    Raises
    ------
    FileNotFoundError
        If the given path does not point to an existing file.
    fitz.FileDataError
        If PyMuPDF cannot open or parse the file (corrupted / not a PDF).
    """

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    page_texts = []  # Collect text page by page to preserve order

    # Open the document — fitz.open() supports PDF, XPS, EPUB, etc.
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        print(f"  [fitz] Opened '{os.path.basename(pdf_path)}' — {total_pages} page(s)")

        for page_number, page in enumerate(doc, start=1):
            # get_text("text") returns plain text; alternatives are "html",
            # "xml", "dict" etc.  Plain text is sufficient for NLP pipelines.
            page_text = page.get_text("text")

            # Annotate each page block so downstream code can locate sources
            header = f"\n--- Page {page_number} of {total_pages} ---\n"
            page_texts.append(header + page_text)

    # Join pages with a form-feed separator; strip leading/trailing whitespace
    full_text = "\f".join(page_texts).strip()
    return full_text


def save_text_to_file(text: str, pdf_filename: str, output_dir: str) -> str:
    """
    Save extracted text to a .txt file inside the output directory.

    The output filename mirrors the PDF filename, with the extension replaced
    by ``.txt`` (e.g., ``judgment_001.pdf`` → ``judgment_001.txt``).

    Parameters
    ----------
    text : str
        The extracted text content to write.
    pdf_filename : str
        Base filename of the source PDF (e.g., ``judgment_001.pdf``).
    output_dir : str
        Path to the directory where the .txt file should be saved.

    Returns
    -------
    str
        Full path of the saved .txt file.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Build the output filename: same stem, .txt extension
    stem = os.path.splitext(pdf_filename)[0]
    output_filename = stem + ".txt"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    return output_path


def process_all_pdfs(
    input_dir: str = PDF_INPUT_DIR,
    output_dir: str = TEXT_OUTPUT_DIR,
) -> dict[str, str]:
    """
    Process every PDF in ``input_dir``, extract its text, save a .txt file,
    and return a mapping of filename → extracted text.

    Parameters
    ----------
    input_dir : str
        Directory to scan for ``.pdf`` files (case-insensitive extension
        match).  Defaults to the project's ``sc_English/`` folder.
    output_dir : str
        Directory where ``.txt`` files will be written.  Defaults to the
        project's ``output/`` folder.

    Returns
    -------
    dict[str, str]
        ``{ "judgment_001.pdf": "<full extracted text>", ... }``
        Only successfully extracted PDFs are included.
    """

    # Collect all PDF files in the input directory (non-recursive)
    all_files = os.listdir(input_dir)
    pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"[pdf_reader] No PDF files found in: {input_dir}")
        return {}

    print(f"[pdf_reader] Found {len(pdf_files)} PDF file(s) in '{input_dir}'")
    print("-" * 60)

    results: dict[str, str] = {}

    for pdf_filename in sorted(pdf_files):  # sorted for deterministic ordering
        pdf_path = os.path.join(input_dir, pdf_filename)
        print(f"[pdf_reader] Processing: {pdf_filename}")

        try:
            text = extract_text_from_pdf(pdf_path)
            output_path = save_text_to_file(text, pdf_filename, output_dir)

            char_count = len(text)
            print(f"  [OK] Extracted {char_count:,} characters → saved to '{output_path}'")

            results[pdf_filename] = text

        except FileNotFoundError as e:
            print(f"  [ERROR] File not found — {e}")
        except Exception as e:
            # Catch fitz errors and any other unexpected exceptions gracefully
            print(f"  [ERROR] Failed to process '{pdf_filename}': {e}")

        print()  # Blank line between files for readability

    print("-" * 60)
    print(f"[pdf_reader] Done. Successfully processed {len(results)} / {len(pdf_files)} PDF(s).")
    return results


# ---------------------------------------------------------------------------
# Entry Point — run directly for a standalone extraction
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    extracted = process_all_pdfs()

    if extracted:
        print("\n[pdf_reader] Extraction summary:")
        for filename, text in extracted.items():
            preview = text[:200].replace("\n", " ")
            print(f"  • {filename}: \"{preview}...\"")
