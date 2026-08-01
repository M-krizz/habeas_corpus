"""
metadata_store.py — Habeas Corpus / Semantic Retrieval Engine
==============================================================
Module: semantic_retrieval/metadata_store.py

Responsibility:
    Persist and reload the chunk metadata that runs parallel to the FAISS
    index.  FAISS stores only float32 vectors — it has no concept of which
    case a vector came from, what its text was, or where the source file
    lives.  This module bridges that gap.

Storage format:
    A single JSON file (``metadata.json``) in the semantic_index/
    directory.  The file contains a JSON array where position ``i``
    corresponds exactly to FAISS row ``i``.

    Example entry:
        {
            "case_id":     "2022_1_1_17_EN",
            "chunk_index": 3,
            "text":        "The court held that negligent driving...",
            "word_count":  412,
            "source_file": "output/2022_1_1_17_EN.txt"
        }

Design note:
    Positional alignment (metadata[i] ↔ FAISS row i) is the contract
    that makes retrieval work.  ``faiss_builder.py`` must build the index
    in the same order that the metadata list is assembled.  Both are
    driven by the same ``all_chunks`` list in ``pipeline.py``.

Usage:
    from semantic_retrieval.metadata_store import save_metadata, load_metadata

    save_metadata(chunks, Path("semantic_index/metadata.json"))
    chunks = load_metadata(Path("semantic_index/metadata.json"))
    chunk  = get_chunk_by_index(chunks, 42)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_metadata(chunks: List[dict], path: Path) -> None:
    """
    Persist the chunk metadata list to a JSON file.

    Parameters
    ----------
    chunks : list of chunk dicts produced by ``chunker.chunk_document()``
    path   : destination path for the JSON file

    Notes
    -----
    - The parent directory is created automatically if it does not exist.
    - The file is written atomically as UTF-8 JSON with indentation.
    - Overwrites any existing file at ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    print(f"[metadata_store] Saved {len(chunks):,} chunk records -> {path}")


def load_metadata(path: Path) -> List[dict]:
    """
    Load chunk metadata from a JSON file.

    Parameters
    ----------
    path : path to the ``metadata.json`` file

    Returns
    -------
    list of dict
        Positionally aligned with the FAISS index rows.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist (index has not been built yet).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"[metadata_store] metadata.json not found at {path}.\n"
            "Run 'build_index()' to create it."
        )
    with open(path, encoding="utf-8") as fh:
        chunks = json.load(fh)
    print(f"[metadata_store] Loaded {len(chunks):,} chunk records <- {path}")
    return chunks


def get_chunk_by_index(chunks: List[dict], index: int) -> dict:
    """
    Return the chunk metadata at position ``index``.

    This is the primary look-up used by ``faiss_search.py`` after FAISS
    returns a list of row indices.

    Parameters
    ----------
    chunks : full metadata list (as returned by ``load_metadata``)
    index  : zero-based FAISS row index

    Returns
    -------
    dict  — the chunk metadata dict at position ``index``

    Raises
    ------
    IndexError
        If ``index`` is out of range.
    """
    if index < 0 or index >= len(chunks):
        raise IndexError(
            f"[metadata_store] Index {index} out of range "
            f"(store has {len(chunks)} entries)."
        )
    return chunks[index]


def get_chunks_by_indices(chunks: List[dict], indices: List[int]) -> List[dict]:
    """
    Batch version of ``get_chunk_by_index``.

    Parameters
    ----------
    chunks  : full metadata list
    indices : list of FAISS row indices (as returned by a search)

    Returns
    -------
    list of dict, in the same order as ``indices``
    """
    return [get_chunk_by_index(chunks, i) for i in indices]
