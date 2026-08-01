"""
chunker.py — Habeas Corpus / Semantic Retrieval Engine
=======================================================
Module: semantic_retrieval/chunker.py

Responsibility:
    Split a single cleaned judgment text into overlapping word-boundary
    chunks suitable for embedding.  This is the most important step in
    the semantic retrieval pipeline — embedding an entire 100-page judgment
    would produce a meaningless average vector.  Smaller, semantically
    coherent chunks let the embedding model capture the *specific* legal
    reasoning that a user query is looking for.

Chunking strategy:
    - Sliding window over word tokens (not characters).
    - Window size  : CHUNK_SIZE  words (default 400)
    - Step size    : CHUNK_SIZE − OVERLAP words  (default 350)
    - Overlap      : OVERLAP words  (default 50)
    - Minimum size : MIN_WORDS words (default 30) — skip trailing stubs

Design notes:
    - Word-boundary splitting preserves tokens; no mid-word cuts.
    - Overlap ensures that a sentence spanning two chunk boundaries is
      fully represented in at least one chunk (avoids meaning loss at
      edges).
    - The output list is positionally stable: chunk at index i in the
      returned list corresponds to FAISS row i in the index built from
      those chunks.

Usage:
    from semantic_retrieval.chunker import chunk_document

    chunks = chunk_document(case_id="2022_1_1_17_EN",
                            cleaned_text=text,
                            source_file="output/2022_1_1_17_EN.txt")
    # → [{"case_id": ..., "chunk_index": 0, "text": ..., "word_count": 400,
    #      "source_file": ...}, ...]
"""

from __future__ import annotations

import re
from typing import Iterator


# ---------------------------------------------------------------------------
# Default chunking parameters
# ---------------------------------------------------------------------------

CHUNK_SIZE = 400   # target words per chunk
OVERLAP    = 50    # words shared between consecutive chunks
MIN_WORDS  = 30    # discard trailing chunks shorter than this


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """
    Split text into word tokens, preserving punctuation attached to words.

    We split on whitespace rather than using a full tokeniser so the module
    has zero NLP dependencies (chunker must be fast and lightweight).
    """
    return text.split()


def _words_to_text(words: list[str]) -> str:
    """Rejoin a word list into a readable string."""
    return " ".join(words)


def _sliding_windows(
    words: list[str],
    chunk_size: int,
    overlap: int,
) -> Iterator[tuple[int, list[str]]]:
    """
    Yield ``(window_index, word_list)`` tuples using a sliding window.

    Parameters
    ----------
    words      : flat list of word tokens for the full document
    chunk_size : target number of words per window
    overlap    : number of words shared with the next window

    Yields
    ------
    (int, list[str])
        Zero-based window index and the corresponding word sublist.
    """
    step  = max(1, chunk_size - overlap)
    start = 0
    idx   = 0
    total = len(words)

    while start < total:
        end   = min(start + chunk_size, total)
        chunk = words[start:end]
        yield idx, chunk
        if end == total:
            break          # do not yield an empty trailing window
        start += step
        idx   += 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(
    case_id: str,
    cleaned_text: str,
    source_file: str = "",
    chunk_size: int  = CHUNK_SIZE,
    overlap: int     = OVERLAP,
    min_words: int   = MIN_WORDS,
) -> list[dict]:
    """
    Split ``cleaned_text`` into overlapping chunks and return a list of
    chunk metadata dicts.

    Parameters
    ----------
    case_id      : stable identifier for the document (e.g. filename stem)
    cleaned_text : output of ``extractor.clean_text.clean_document()``
    source_file  : path to the source .txt file (stored as provenance)
    chunk_size   : target words per chunk (default 400)
    overlap      : words of overlap between consecutive chunks (default 50)
    min_words    : minimum words for a trailing chunk to be kept (default 30)

    Returns
    -------
    list of dict
        Each dict has keys:
            case_id     (str)  — document identifier
            chunk_index (int)  — zero-based position in this document
            text        (str)  — chunk text (rejoined words)
            word_count  (int)  — actual word count of this chunk
            source_file (str)  — provenance path
    """
    if not cleaned_text or not cleaned_text.strip():
        return []

    words  = _tokenise(cleaned_text)
    chunks = []

    for idx, word_list in _sliding_windows(words, chunk_size, overlap):
        if len(word_list) < min_words:
            continue   # skip tiny trailing stubs
        chunks.append(
            {
                "case_id":     case_id,
                "chunk_index": idx,
                "text":        _words_to_text(word_list),
                "word_count":  len(word_list),
                "source_file": source_file,
            }
        )

    return chunks


# ---------------------------------------------------------------------------
# Entry point — smoke-test on a .txt file
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m semantic_retrieval.chunker <path_to_txt>")
        sys.exit(0)

    from pathlib import Path

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    text   = path.read_text(encoding="utf-8")
    result = chunk_document(case_id=path.stem, cleaned_text=text,
                             source_file=str(path))

    print(f"[chunker] {path.name}")
    print(f"          Total words : {len(text.split()):,}")
    print(f"          Chunks      : {len(result)}")
    if result:
        print(f"          First chunk ({result[0]['word_count']} words):")
        print("          " + result[0]["text"][:300] + "...")
