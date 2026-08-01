"""
pipeline.py — Habeas Corpus / Semantic Retrieval Engine
========================================================
Module: semantic_retrieval/pipeline.py

Responsibility:
    Top-level orchestrator for the Semantic Retrieval Engine.
    This is the **only** file that callers (including ``main.py``) need
    to import.  It wires together:

        Offline  →  chunker  →  embedder  →  faiss_builder  →  metadata_store
        Online   →  embedder  →  faiss_search  →  ranker

Public API:
    build_index(txt_dir, index_dir)    — offline: build from .txt files
    search_cases(query, top_k, n)      — online: query → ranked case list
    get_top_case_ids(query, top_k, n)  — online: query → list of case_id strings

Usage:
    # Offline (once, or whenever new judgments arrive)
    from semantic_retrieval.pipeline import build_index
    build_index()                          # reads output/*.txt by default

    # Online (every user query)
    from semantic_retrieval.pipeline import search_cases, get_top_case_ids
    results  = search_cases("Someone hit my parked bike")
    case_ids = get_top_case_ids("Someone hit my parked bike")

    Or from the command line:
        python -m semantic_retrieval.pipeline --build
        python -m semantic_retrieval.pipeline --query "negligent driving"
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import List

import numpy as np

from semantic_retrieval.chunker        import chunk_document
from semantic_retrieval.embedder       import encode_chunks
from semantic_retrieval.metadata_store import save_metadata, load_metadata
from semantic_retrieval.faiss_builder  import build_index as _build_faiss, save_index
from semantic_retrieval.faiss_search   import search as _faiss_search, invalidate_cache
from semantic_retrieval.ranker         import rank_cases, top_case_ids


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_OUTPUT_DIR   = _PROJECT_ROOT / "output"
_INDEX_DIR    = _PROJECT_ROOT / "semantic_index"
_INDEX_PATH   = _INDEX_DIR / "legal.index"
_META_PATH    = _INDEX_DIR / "metadata.json"


# ---------------------------------------------------------------------------
# Offline: build_index
# ---------------------------------------------------------------------------

def build_index(
    txt_dir:   Path | str = _OUTPUT_DIR,
    index_dir: Path | str = _INDEX_DIR,
) -> None:
    """
    Build the semantic index from all ``.txt`` files in ``txt_dir``.

    This is the **offline** phase.  Run it once when new judgments arrive.
    It will always rebuild the index from scratch to avoid stale vectors.

    Pipeline:
        1. Read all .txt files in ``txt_dir``
        2. Chunk each document (400 words, 50-word overlap)
        3. Embed all chunks using BAAI/bge-m3
        4. Save metadata (positionally aligned with vectors)
        5. Build FAISS IndexFlatIP and save to disk

    Parameters
    ----------
    txt_dir   : directory containing cleaned .txt judgment files
                (default: ``output/`` in the project root)
    index_dir : directory where ``legal.index`` and ``metadata.json``
                will be written (default: ``semantic_index/``)

    Raises
    ------
    FileNotFoundError
        If ``txt_dir`` does not exist or contains no .txt files.
    """
    txt_dir   = Path(txt_dir)
    index_dir = Path(index_dir)

    # --- 1. Collect .txt files ---
    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"[pipeline] No .txt files found in {txt_dir}. "
            "Run pdf_reader.py first to extract text from PDFs."
        )

    print(f"\n[pipeline] {'=' * 55}")
    print(f"[pipeline] OFFLINE INDEX BUILD")
    print(f"[pipeline] Source dir : {txt_dir}")
    print(f"[pipeline] Index dir  : {index_dir}")
    print(f"[pipeline] Found {len(txt_files)} .txt file(s)")
    print(f"[pipeline] {'=' * 55}")

    # --- 2. Chunk all documents ---
    all_chunks: list[dict] = []
    for txt_path in txt_files:
        case_id = txt_path.stem           # e.g. "2022_1_1_17_EN"
        text    = txt_path.read_text(encoding="utf-8")
        chunks  = chunk_document(
            case_id     = case_id,
            cleaned_text = text,
            source_file  = str(txt_path),
        )
        all_chunks.extend(chunks)
        print(f"[pipeline]   {txt_path.name:40s}  -> {len(chunks):4d} chunks")

    print(f"[pipeline] Total chunks: {len(all_chunks):,}")

    if not all_chunks:
        print("[pipeline] WARNING — no chunks produced.  "
              "Check that .txt files are not empty.")
        return

    # --- 3. Embed all chunks ---
    print(f"\n[pipeline] Embedding {len(all_chunks):,} chunks with BAAI/bge-m3...")
    texts   = [c["text"] for c in all_chunks]
    vectors = encode_chunks(texts)         # shape (n, 1024), L2-normalised
    print(f"[pipeline] Embedding complete. Shape: {vectors.shape}")

    # --- 4. Save metadata ---
    save_metadata(all_chunks, _META_PATH if index_dir == _INDEX_DIR
                  else index_dir / "metadata.json")

    # --- 5. Build and save FAISS index ---
    index = _build_faiss(vectors)
    save_index(index, _INDEX_PATH if index_dir == _INDEX_DIR
               else index_dir / "legal.index")

    # Invalidate the search singleton so the new index is picked up immediately
    invalidate_cache()

    print(f"\n[pipeline] [OK] Index built successfully.")
    print(f"[pipeline]   {_INDEX_PATH}")
    print(f"[pipeline]   {_META_PATH}")
    print(f"[pipeline] {'=' * 55}\n")


# ---------------------------------------------------------------------------
# Online: search_cases
# ---------------------------------------------------------------------------

def search_cases(
    query:  str,
    top_k:  int = 20,
    n:      int = 5,
) -> List[dict]:
    """
    Search for the most semantically relevant cases for a user query.

    This is the **online** phase.  Runs on every user request.
    The index is loaded once and cached for the lifetime of the process.

    Pipeline:
        1. Embed the query using BAAI/bge-m3 (with instruction prefix)
        2. Retrieve top_k most similar chunks from FAISS
        3. Aggregate chunk scores into case-level scores (sum strategy)
        4. Return top-n cases, sorted by score

    Parameters
    ----------
    query  : raw user query string (no pre-processing needed)
    top_k  : number of chunk-level FAISS hits to retrieve (default 20)
    n      : number of cases to return after ranking (default 5)

    Returns
    -------
    list of dict (at most ``n`` entries), each:
        {
            "case_id":        str,    # e.g. "2022_1_1_17_EN"
            "score":          float,  # summed cosine similarity
            "matched_chunks": int,    # chunks that matched
            "top_chunk_text": str,    # text of the best-matching chunk
            "top_score":      float   # best single-chunk score
        }
    """
    chunk_hits   = _faiss_search(query, top_k=top_k)
    ranked       = rank_cases(chunk_hits)
    return ranked[:n]


def get_top_case_ids(
    query:  str,
    top_k:  int = 20,
    n:      int = 5,
) -> List[str]:
    """
    Convenience wrapper — returns only the top-n case_id strings.

    This is what gets handed to the Neo4j query layer:

        ids = get_top_case_ids("someone hit my parked bike")
        # → ["2022_1_1_17_EN", "2022_1_28_104_EN"]

    Parameters and returns: see ``search_cases``.
    """
    return top_case_ids(search_cases(query, top_k=top_k, n=n))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--build" in args:
        # Offline: build the index
        build_index()

    elif "--query" in args:
        # Online: run a query
        try:
            q_idx = args.index("--query")
            query_str = " ".join(args[q_idx + 1:])
        except (ValueError, IndexError):
            print("Usage: python -m semantic_retrieval.pipeline --query <your query>")
            sys.exit(1)

        if not query_str.strip():
            print("[pipeline] Empty query.")
            sys.exit(1)

        print(f"\n[pipeline] Query: '{query_str}'")
        print("=" * 60)

        results = search_cases(query_str, top_k=20, n=5)

        if not results:
            print("[pipeline] No results found.  "
                  "Have you run --build yet?")
        else:
            for i, r in enumerate(results, 1):
                print(f"\n  Rank #{i}")
                print(f"  Case ID       : {r['case_id']}")
                print(f"  Score         : {r['score']:.4f}")
                print(f"  Chunks matched: {r['matched_chunks']}")
                print(f"  Best chunk    : {r['top_chunk_text'][:200]}...")

        print("\n[pipeline] Case IDs for Neo4j:")
        print("  ", get_top_case_ids(query_str))

    else:
        print("Usage:")
        print("  python -m semantic_retrieval.pipeline --build")
        print("  python -m semantic_retrieval.pipeline --query <your query>")
