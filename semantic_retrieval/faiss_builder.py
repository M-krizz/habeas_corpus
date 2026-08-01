"""
faiss_builder.py — Habeas Corpus / Semantic Retrieval Engine
=============================================================
Module: semantic_retrieval/faiss_builder.py

Responsibility:
    Build, persist, and reload the FAISS vector index.

Index type: IndexFlatIP
    - ``IP`` = Inner Product.
    - Because all vectors are L2-normalised by ``embedder.py``,
      inner product == cosine similarity.
    - ``Flat`` = exact nearest-neighbour search (no approximation).
    - No training required — vectors can be added immediately.
    - Appropriate for corpora up to ~millions of vectors on CPU.

Vector dimension: 1024  (BAAI/bge-m3 output)

Upgrade path:
    When the corpus grows to hundreds of thousands of judgments,
    replace ``IndexFlatIP`` with ``IndexIVFFlat`` for approximate
    search with much faster query times.  The rest of the pipeline
    requires no changes.

Usage:
    from semantic_retrieval.faiss_builder import build_index, save_index, load_index

    index = build_index(vectors)           # np.ndarray (n, 1024) float32
    save_index(index, Path("semantic_index/legal.index"))
    index = load_index(Path("semantic_index/legal.index"))
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 1024   # BAAI/bge-m3 output dimension


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP from a matrix of L2-normalised vectors.

    Parameters
    ----------
    vectors : np.ndarray
        Shape ``(n, 1024)``, dtype float32.
        Must already be L2-normalised (``embedder.encode_chunks`` does this).

    Returns
    -------
    faiss.IndexFlatIP
        Populated index ready for search.

    Raises
    ------
    ValueError
        If ``vectors`` has the wrong shape or dtype.
    """
    if vectors.ndim != 2:
        raise ValueError(
            f"[faiss_builder] Expected 2-D array, got shape {vectors.shape}"
        )
    n, dim = vectors.shape
    if dim != EMBEDDING_DIM:
        raise ValueError(
            f"[faiss_builder] Expected {EMBEDDING_DIM}-D vectors, got {dim}-D. "
            "Check that the embedder model matches EMBEDDING_DIM."
        )
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"[faiss_builder] Built IndexFlatIP — {n:,} vectors, dim={dim}")
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    """
    Persist a FAISS index to disk.

    Parameters
    ----------
    index : any FAISS index
    path  : destination file path (parent directory created if needed)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    print(f"[faiss_builder] Saved index ({index.ntotal:,} vectors) -> {path}")


def load_index(path: Path) -> faiss.Index:
    """
    Load a FAISS index from disk.

    Parameters
    ----------
    path : path to the ``.index`` file

    Returns
    -------
    faiss.Index

    Raises
    ------
    FileNotFoundError
        If the file does not exist (index has not been built yet).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"[faiss_builder] Index not found at {path}.\n"
            "Run 'build_index()' (offline phase) to create it."
        )
    index = faiss.read_index(str(path))
    print(f"[faiss_builder] Loaded index ({index.ntotal:,} vectors) <- {path}")
    return index


# ---------------------------------------------------------------------------
# Entry point — smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("[faiss_builder] Smoke test — building a tiny index of 5 random vectors...")
    rng  = np.random.default_rng(42)
    vecs = rng.random((5, EMBEDDING_DIM), dtype=np.float32)
    # L2-normalise
    vecs = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)

    idx = build_index(vecs)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.index"
        save_index(idx, p)
        idx2 = load_index(p)

    query  = vecs[[0]]                        # query with first vector (self-match)
    D, I   = idx2.search(query, k=3)
    print(f"[faiss_builder] Top-3 indices for row 0 query : {I[0]}")
    print(f"[faiss_builder] Corresponding scores          : {D[0]}")
    print("[faiss_builder] Smoke test passed [OK]")
