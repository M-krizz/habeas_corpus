"""
evidence/aggregator.py — Habeas Corpus
=======================================
Assembles CaseEvidence objects by joining:
  - FAISS semantic metadata (chunk texts, per-case scores)
  - Neo4j structured metadata (judges, acts, sections, parties)

This is the "Evidence Aggregator" stage described in the architecture.
Instead of passing raw chunks to the LLM, we give it complete, structured
case objects — so the LLM never has to guess what Act or Judge applied.
"""

from __future__ import annotations

from semantic_retrieval.metadata_store import load_metadata
from pathlib import Path

from evidence.schema import CaseEvidence
from retrieval.graph_retriever import fetch_case_metadata

# Path to FAISS metadata — used to look up chunk texts by case_id
_PROJECT_ROOT   = Path(__file__).parent.parent
_META_PATH      = _PROJECT_ROOT / "semantic_index" / "metadata.json"

_MAX_CHUNKS_PER_CASE = 3   # max semantic passages to include per case


def _load_chunk_index() -> dict[str, list[str]]:
    """
    Build a {case_id: [chunk_text, ...]} index from the FAISS metadata file.
    Cached per-process via module-level variable (lazy load).
    """
    global _CHUNK_INDEX
    if _CHUNK_INDEX is not None:
        return _CHUNK_INDEX
    try:
        chunks = load_metadata(_META_PATH)
    except FileNotFoundError:
        print("[aggregator] WARNING — FAISS metadata not found. "
              "Run 'build_index()' first. Chunk texts will be empty.")
        _CHUNK_INDEX = {}
        return _CHUNK_INDEX

    index: dict[str, list[str]] = {}
    for chunk in chunks:
        cid  = chunk.get("case_id") or chunk.get("source") or "unknown"
        text = chunk.get("text", "")
        if text:
            index.setdefault(cid, []).append(text)
    _CHUNK_INDEX = index
    return _CHUNK_INDEX


_CHUNK_INDEX: dict[str, list[str]] | None = None


def aggregate_evidence(
    ranked_cases: list[dict],
    top_n: int = 5,
) -> list[CaseEvidence]:
    """
    Assemble full CaseEvidence objects for the top-N ranked cases.

    For each case the aggregator:
    1. Fetches structured metadata from Neo4j (judges, acts, sections, …)
    2. Retrieves the top FAISS chunk texts for that case
    3. Combines everything into a CaseEvidence object

    Parameters
    ----------
    ranked_cases : output of hybrid_ranker.hybrid_rank()
    top_n        : how many cases to assemble evidence for (default 5)

    Returns
    -------
    list[CaseEvidence] — one per case, ordered by final_score descending
    """
    chunk_index = _load_chunk_index()
    evidence_list: list[CaseEvidence] = []

    for rank_entry in ranked_cases[:top_n]:
        cid   = rank_entry["case_id"]

        # --- Graph metadata ---
        meta  = fetch_case_metadata(cid)

        # --- Semantic chunk texts ---
        all_chunks = chunk_index.get(cid, [])
        top_chunks = all_chunks[:_MAX_CHUNKS_PER_CASE]

        ev = CaseEvidence(
            case_id         = cid,
            case_name       = meta.get("case_name", cid),
            decision_date   = meta.get("decision_date", ""),
            court           = (meta.get("courts") or [""])[0],
            judges          = meta.get("judges", []),
            acts            = meta.get("acts", []),
            sections        = meta.get("sections", []),
            petitioners     = meta.get("petitioners", []),
            respondents     = meta.get("respondents", []),
            semantic_score  = rank_entry.get("semantic_score", 0.0),
            graph_score     = rank_entry.get("graph_score", 0.0),
            final_score     = rank_entry.get("final_score", 0.0),
            relevant_chunks = top_chunks,
            top_chunk_text  = rank_entry.get("top_chunk_text",
                                              top_chunks[0] if top_chunks else ""),
            source          = "permanent_kg",
        )
        evidence_list.append(ev)

    print(f"[aggregator] Assembled {len(evidence_list)} CaseEvidence objects.")
    return evidence_list


def aggregate_from_web(web_cases: list[dict]) -> list[CaseEvidence]:
    """
    Build lightweight CaseEvidence objects from web-retrieved case dicts.
    Used when the pipeline falls back to Indian Kanoon.

    web_cases format (from knowledge_acquisition.web_retriever):
        [{title, doc_id, headline, url, text_excerpt, acts, sections}]
    """
    result: list[CaseEvidence] = []
    for wc in web_cases:
        ev = CaseEvidence(
            case_id         = wc.get("doc_id", "web_case"),
            case_name       = wc.get("title", ""),
            decision_date   = wc.get("date", ""),
            court           = wc.get("court", ""),
            judges          = [],
            acts            = wc.get("acts", []),
            sections        = wc.get("sections", []),
            petitioners     = [],
            respondents     = [],
            semantic_score  = wc.get("relevance_score", 0.5),
            graph_score     = 0.0,
            final_score     = wc.get("relevance_score", 0.5),
            relevant_chunks = [wc.get("text_excerpt", "")],
            top_chunk_text  = wc.get("text_excerpt", ""),
            source          = "web_retrieved",
            source_url      = wc.get("url", ""),
        )
        result.append(ev)
    return result
