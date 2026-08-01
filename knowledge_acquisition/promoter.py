"""
knowledge_acquisition/promoter.py — Habeas Corpus
===================================================
Promotes a validated staging case into the permanent KG + FAISS.

Reuses the EXISTING pipeline modules:
  extractor.clean_text        → clean the retrieved text
  extractor.entity_extractor  → extract legal entities
  graph.graph_builder         → build graph structure
  graph.neo4j_loader          → persist to Neo4j (LIVE mode)
  semantic_retrieval.pipeline → append to FAISS index

This module adds NO new logic — it simply wires together what already exists.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from knowledge_acquisition.staging_pool import (
    update_status, STATUS_PROMOTED, STATUS_REJECTED
)


def promote(staging_hash: str, case: dict) -> bool:
    """
    Promote a validated staging case to the permanent KG + FAISS.

    Parameters
    ----------
    staging_hash : staging pool hash identifier
    case         : the staged case dict (from staging_pool.load_case)

    Returns
    -------
    bool — True if promotion succeeded, False otherwise.
    """
    title   = case.get("title", staging_hash)
    text    = case.get("text_excerpt", case.get("headline", ""))

    if not text or len(text.strip()) < 100:
        reason = f"Text too short for indexing ({len(text)} chars)"
        print(f"[promoter] SKIP '{title}': {reason}")
        update_status(staging_hash, STATUS_REJECTED, reason)
        return False

    print(f"\n[promoter] Promoting: '{title[:60]}'")

    try:
        # --- Stage 1: Clean text ---
        from extractor.clean_text import clean_document
        cleaned = clean_document(text)

        # --- Stage 2: Extract entities ---
        from extractor.entity_extractor import extract_legal_graph
        legal_graph = extract_legal_graph(cleaned)

        # Override case name with Indian Kanoon title if extractor missed it
        if not legal_graph["nodes"]["case"]["name"]:
            legal_graph["nodes"]["case"]["name"] = title[:200]

        # --- Stage 3: Build graph ---
        from graph.graph_builder import build_graph
        loader_graph = build_graph(legal_graph)

        # --- Stage 4: Load into Neo4j (LIVE) ---
        from graph.neo4j_loader import load_graph
        load_graph(loader_graph, dry_run=False)

        # --- Stage 5: Write temp .txt and rebuild FAISS ---
        case_id = _make_case_id(title)
        _append_to_faiss(case_id, cleaned, case.get("url", ""))

        update_status(staging_hash, STATUS_PROMOTED)
        print(f"[promoter] [OK] Promoted '{title[:60]}' -> Neo4j + FAISS")
        return True

    except Exception as exc:
        reason = f"Promotion error: {exc}"
        print(f"[promoter] ERROR: {reason}")
        update_status(staging_hash, STATUS_REJECTED, reason)
        return False


def _make_case_id(title: str) -> str:
    """Convert a case title into a safe file-system identifier."""
    import re
    clean = re.sub(r"[^\w\s]", "", title)
    tokens = clean.split()[:6]
    return "_".join(t.lower() for t in tokens) or "web_case"


def _append_to_faiss(case_id: str, cleaned_text: str, source_url: str) -> None:
    """
    Write the cleaned text to output/ and rebuild the FAISS index.

    We write a .txt file so the existing pipeline's glob("*.txt") picks it up.
    """
    from pathlib import Path
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    txt_path = output_dir / f"{case_id}_web.txt"
    txt_path.write_text(
        f"[Source: {source_url}]\n\n{cleaned_text}",
        encoding="utf-8",
    )
    print(f"[promoter] Wrote {txt_path.name} to output/")

    # Full index rebuild (always safe for current corpus size)
    from semantic_retrieval.pipeline import build_index
    build_index()
    print("[promoter] FAISS index rebuilt.")
