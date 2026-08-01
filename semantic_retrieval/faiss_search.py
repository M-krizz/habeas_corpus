"""
faiss_search.py — Habeas Corpus / Semantic Retrieval Engine
============================================================
Module: semantic_retrieval/faiss_search.py

Responsibility:
    Online search — given a raw user query string, return the top-K most
    semantically similar chunks from the pre-built FAISS index.

    This module is the only component that runs on every user request.
    Everything else (chunking, embedding, index building) is offline.

Lazy singleton:
    The FAISS index and metadata are loaded from disk on the first call
    to ``search()`` and kept in memory for the lifetime of the process.
    Subsequent calls are fast (no I/O, no model re-loading).

Pipeline position:
    User Query
        │
        ▼
    encode_query()          ← embedder.py (same model, same normalisation)
        │
        ▼
    index.search()          ← FAISS IndexFlatIP
        │
        ▼
    get_chunks_by_indices() ← metadata_store.py
        │
        ▼
    list of enriched chunk dicts (with similarity score attached)
        │
        ▼
    ranker.py               ← aggregates into case-level rankings

Usage:
    from semantic_retrieval.faiss_search import search

    results = search("Someone hit my parked bike", top_k=20)
    # → [{"case_id": ..., "chunk_index": ..., "text": ..., "score": 0.91}, ...]
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np

from semantic_retrieval.embedder       import encode_query
from semantic_retrieval.faiss_builder  import load_index
from semantic_retrieval.metadata_store import load_metadata, get_chunks_by_indices


# ---------------------------------------------------------------------------
# Default paths — relative to the project root
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
INDEX_DIR     = _PROJECT_ROOT / "semantic_index"
INDEX_PATH    = INDEX_DIR / "legal.index"
METADATA_PATH = INDEX_DIR / "metadata.json"


# ---------------------------------------------------------------------------
# Lazy singletons — loaded once on first search call
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_search_state() -> tuple:
    """
    Load the FAISS index and metadata into memory exactly once.

    Returns
    -------
    tuple  (faiss.Index, list[dict])
        The loaded index and the full metadata list.
    """
    index  = load_index(INDEX_PATH)
    chunks = load_metadata(METADATA_PATH)
    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"[faiss_search] Mismatch: FAISS has {index.ntotal} vectors but "
            f"metadata has {len(chunks)} entries.  Rebuild the index."
        )
    return index, chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(query: str, top_k: int = 20) -> List[dict]:
    """
    Search the FAISS index for the ``top_k`` chunks most similar to ``query``.

    Parameters
    ----------
    query  : raw user query string (not pre-processed)
    top_k  : number of chunk results to return (default 20)
             A higher value gives the ranker more material to work with,
             but increases Neo4j look-up overhead.

    Returns
    -------
    list of dict
        Each dict is the chunk's metadata with an additional ``score`` key:

            {
                "case_id":     "2022_1_1_17_EN",
                "chunk_index": 4,
                "text":        "The court held...",
                "word_count":  400,
                "source_file": "output/2022_1_1_17_EN.txt",
                "score":       0.9121   ← cosine similarity (0–1)
            }

        Sorted descending by score.

    Raises
    ------
    FileNotFoundError
        If the index or metadata file has not been built yet.
    """
    index, all_chunks = _load_search_state()

    # Encode the query (includes BGE-M3 instruction prefix)
    q_vec = encode_query(query)             # shape (1, 1024), float32

    # Clamp top_k to the number of available vectors
    k = min(top_k, index.ntotal)

    # Run FAISS search — returns (distances, indices) each shape (1, k)
    scores_arr, indices_arr = index.search(q_vec, k)

    scores  = scores_arr[0].tolist()       # list of float
    indices = indices_arr[0].tolist()      # list of int

    # Build enriched results: shallow-copy each chunk dict and attach score
    enriched = []
    for idx, score in zip(indices, scores):
        entry = dict(all_chunks[idx])
        entry["score"]     = round(float(score), 6)
        entry["faiss_row"] = idx
        enriched.append(entry)

    return sorted(enriched, key=lambda x: x["score"], reverse=True)


def invalidate_cache() -> None:
    """
    Force the next call to ``search()`` to reload the index from disk.

    Call this after ``build_index()`` in the same process (e.g., when
    ``main.py --index`` rebuilds the index and then immediately tests it).
    """
    _load_search_state.cache_clear()


# ---------------------------------------------------------------------------
# Entry point — smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "negligent driving compensation"
    print(f"[faiss_search] Query: '{query}'")
    print("-" * 60)

    results = search(query, top_k=5)
    for i, r in enumerate(results, 1):
        print(f"  #{i}  case={r['case_id']}  chunk={r['chunk_index']}  "
              f"score={r['score']:.4f}")
        print(f"       {r['text'][:120]}...")
        print()
