"""
embedder.py — Habeas Corpus / Semantic Retrieval Engine
========================================================
Module: semantic_retrieval/embedder.py

Responsibility:
    Provide a thin, singleton wrapper around the BAAI/bge-m3 (BGE-3)
    sentence-transformer model.  Every component in the pipeline that
    needs to produce vectors — both the offline index builder and the
    online query handler — imports from here so they always use the same
    model weights and the same normalisation strategy.

Model: BAAI/bge-m3
    - Generation  : 3rd-generation BGE (state-of-the-art as of 2024)
    - Dimension   : 1024-D dense vectors
    - Languages   : multilingual (100+ languages, including Indian legal
                    English and mixed-script documents)
    - Download    : ~570 MB on first use; cached by HuggingFace Hub
    - Device      : CPU by default; auto-detects CUDA if available

Normalisation:
    All vectors are L2-normalised before being stored or used for search.
    This converts inner-product similarity (which FAISS IndexFlatIP
    computes) into cosine similarity, which is the correct metric for
    comparing semantic embeddings.

Usage:
    from semantic_retrieval.embedder import encode_chunks, encode_query

    vectors = encode_chunks(["The truck struck...", "...the parked bike..."])
    # → np.ndarray of shape (2, 1024), float32, L2-normalised

    q_vec = encode_query("Someone hit my parked bike")
    # → np.ndarray of shape (1, 1024), float32, L2-normalised
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache
from typing import List


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "BAAI/bge-m3"

# BGE-M3 instruction prefix recommended for retrieval tasks.
# Applied to the query only (NOT to document chunks).
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Singleton loader — model is loaded once and reused
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_model():
    """
    Load the BGE-M3 sentence-transformer model exactly once.
    """
    import os
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        threads = max(1, os.cpu_count() or 4)
        try:
            torch.set_num_threads(threads)
        except Exception:
            pass

    print(f"[embedder] Loading model: {MODEL_NAME} on device '{device}' (threads: {torch.get_num_threads()})")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"[embedder] Model loaded. Embedding dim: {model.get_embedding_dimension()}")
    return model


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """
    L2-normalise each row of ``matrix`` in-place and return it.

    After normalisation, dot-product == cosine similarity, which is the
    metric used by ``faiss.IndexFlatIP``.

    Parameters
    ----------
    matrix : np.ndarray
        Shape ``(n, dim)``, dtype float32.

    Returns
    -------
    np.ndarray
        The same array, each row normalised to unit length.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero for degenerate zero vectors
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_chunks(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Encode a list of document chunk texts into L2-normalised vectors.
    GPU-optimized: batch_size=32, max_seq_length=256 on CUDA.
    """
    import torch
    model = _get_model()
    on_gpu = torch.cuda.is_available()
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = 256 if on_gpu else 256

    max_chars = 1500 if on_gpu else 1000
    truncated_texts = [t[:max_chars] for t in texts]

    # Process in sub-batches of 1000 under torch.no_grad() for fast inference
    all_vecs = []
    chunk_step = 1000
    with torch.no_grad():
        for i in range(0, len(truncated_texts), chunk_step):
            sub_batch = truncated_texts[i:i + chunk_step]
            vecs = model.encode(
                sub_batch,
                batch_size           = batch_size,
                normalize_embeddings = False,
                show_progress_bar    = False,
                convert_to_numpy     = True,
            )
            all_vecs.append(vecs)
            print(f"[embedder] Embedded {min(i + chunk_step, len(truncated_texts))}/{len(truncated_texts)} chunks...")

    combined = np.vstack(all_vecs) if all_vecs else np.empty((0, 1024), dtype=np.float32)
    return _l2_normalise(combined.astype(np.float32))


def encode_query(query: str) -> np.ndarray:
    """
    Encode a single user query string into a L2-normalised vector.

    The BGE-M3 instruction prefix is prepended to the query to improve
    retrieval performance (this is the recommended usage for the model
    in asymmetric retrieval tasks such as question-to-passage search).

    Parameters
    ----------
    query : raw user query string

    Returns
    -------
    np.ndarray
        Shape ``(1, 1024)``, dtype float32.
    """
    model        = _get_model()
    instructed   = _QUERY_INSTRUCTION + query.strip()
    vector       = model.encode(
        [instructed],
        normalize_embeddings = False,
        show_progress_bar    = False,
        convert_to_numpy     = True,
    )
    return _l2_normalise(np.array(vector, dtype=np.float32))


def embedding_dim() -> int:
    """Return the embedding dimension (1024 for BGE-M3)."""
    return _get_model().get_embedding_dimension()


# ---------------------------------------------------------------------------
# Entry point — smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[embedder] Smoke test — encoding two sample legal sentences...")

    samples = [
        "The respondent's truck collided with the petitioner's parked motorcycle.",
        "Compensation was awarded under Section 166 of the Motor Vehicles Act.",
    ]

    vecs = encode_chunks(samples)
    print(f"[embedder] Output shape   : {vecs.shape}")
    print(f"[embedder] Row 0 norm     : {np.linalg.norm(vecs[0]):.6f}  (should be ~1.0)")
    print(f"[embedder] Cosine sim 0,1 : {float(vecs[0] @ vecs[1]):.4f}")

    q = encode_query("accident involving a truck and a parked vehicle")
    print(f"[embedder] Query shape    : {q.shape}")
    print(f"[embedder] Query-doc sim  : {float(q[0] @ vecs[0]):.4f}")
