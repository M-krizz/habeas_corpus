"""
ranker.py — Habeas Corpus / Semantic Retrieval Engine
======================================================
Module: semantic_retrieval/ranker.py

Responsibility:
    Aggregate chunk-level similarity scores into case-level rankings.

    FAISS returns individual chunk hits.  Multiple chunks can belong to
    the same case.  The ranker groups them by ``case_id`` and computes a
    single case score, then sorts cases descending by that score.

    This is the critical step that converts:
        "20 paragraph-level hits" → "5 ranked cases"

    so that downstream Neo4j queries receive Case IDs, not paragraphs.

Scoring strategy:
    ``case_score = sum(chunk_scores for all chunks of that case)``

    Rationale:
    - A case with many highly-relevant chunks is more relevant than one
      with a single lucky hit.
    - Summation is simple, interpretable, and works well in practice for
      legal corpora where multiple sections of a judgment may all speak
      to the user's query.
    - Alternative strategies (max, mean) are provided as helper functions
      for experimentation.

Output:
    A list of case-level result dicts:

        {
            "case_id":         "2022_1_1_17_EN",
            "score":           2.743,        ← summed chunk similarity
            "matched_chunks":  3,            ← number of chunks that matched
            "top_chunk_text":  "The court held that...",   ← best chunk text
            "top_score":       0.9121        ← best single-chunk score
        }

Usage:
    from semantic_retrieval.ranker import rank_cases

    chunk_results = search("someone hit my parked bike", top_k=20)
    ranked_cases  = rank_cases(chunk_results)
    # → [{case_id, score, matched_chunks, top_chunk_text, top_score}, ...]
"""

from __future__ import annotations

from collections import defaultdict
from typing import List


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_cases(chunk_results: List[dict]) -> List[dict]:
    """
    Aggregate chunk-level search results into ranked case-level results.

    Parameters
    ----------
    chunk_results : list of chunk dicts (as returned by ``faiss_search.search``)
        Each dict must have keys: ``case_id``, ``score``, ``text``.

    Returns
    -------
    list of dict
        Sorted descending by aggregated score.  Each dict:

            case_id         (str)   — stable document identifier
            score           (float) — sum of per-chunk cosine similarities
            matched_chunks  (int)   — how many chunks contributed
            top_chunk_text  (str)   — text of the highest-scoring chunk
            top_score       (float) — highest individual chunk score
    """
    if not chunk_results:
        return []

    # --- Aggregate per case ---
    scores:        dict[str, float]     = defaultdict(float)
    counts:        dict[str, int]       = defaultdict(int)
    best_score:    dict[str, float]     = defaultdict(float)
    best_text:     dict[str, str]       = {}

    for chunk in chunk_results:
        cid   = chunk.get("case_id") or chunk.get("source") or "unknown"
        score = chunk.get("score", 0.0)

        scores[cid] += score
        counts[cid] += 1

        if score > best_score[cid]:
            best_score[cid] = score
            best_text[cid]  = chunk.get("text", "")

    # --- Build result list ---
    ranked = [
        {
            "case_id":        cid,
            "score":          round(scores[cid], 6),
            "matched_chunks": counts[cid],
            "top_chunk_text": best_text.get(cid, ""),
            "top_score":      round(best_score[cid], 6),
        }
        for cid in scores
    ]

    return sorted(ranked, key=lambda x: x["score"], reverse=True)


def top_case_ids(ranked_cases: List[dict], n: int = 5) -> List[str]:
    """
    Extract the top-n ``case_id`` strings from a ranked list.

    This is what gets handed to the Neo4j query layer:

        case_ids = top_case_ids(rank_cases(search(query)))
        # → ["2022_1_1_17_EN", "2022_1_28_104_EN", ...]

    Parameters
    ----------
    ranked_cases : output of ``rank_cases``
    n            : how many case IDs to return (default 5)

    Returns
    -------
    list of str — top-n case_id values, best first
    """
    return [r["case_id"] for r in ranked_cases[:n]]


# ---------------------------------------------------------------------------
# Alternate scoring strategies (for experimentation)
# ---------------------------------------------------------------------------

def rank_cases_by_max(chunk_results: List[dict]) -> List[dict]:
    """
    Rank cases by the single highest chunk score (max strategy).

    Useful when you want the case that has *at least one* highly
    relevant passage, regardless of how many chunks matched.
    """
    if not chunk_results:
        return []

    best: dict[str, dict] = {}
    for chunk in chunk_results:
        cid   = chunk["case_id"]
        score = chunk.get("score", 0.0)
        if cid not in best or score > best[cid]["score"]:
            best[cid] = chunk

    ranked = [
        {
            "case_id":        cid,
            "score":          round(data["score"], 6),
            "matched_chunks": sum(1 for c in chunk_results if c["case_id"] == cid),
            "top_chunk_text": data.get("text", ""),
            "top_score":      round(data["score"], 6),
        }
        for cid, data in best.items()
    ]
    return sorted(ranked, key=lambda x: x["score"], reverse=True)


def rank_cases_by_mean(chunk_results: List[dict]) -> List[dict]:
    """
    Rank cases by the mean chunk score (mean strategy).

    Useful when you want to penalise cases that matched only via
    many low-scoring chunks.
    """
    if not chunk_results:
        return []

    scores: dict[str, list] = defaultdict(list)
    texts:  dict[str, str]  = {}
    for chunk in chunk_results:
        cid   = chunk["case_id"]
        score = chunk.get("score", 0.0)
        scores[cid].append(score)
        if cid not in texts or score > max(scores[cid][:-1], default=0):
            texts[cid] = chunk.get("text", "")

    ranked = [
        {
            "case_id":        cid,
            "score":          round(sum(s) / len(s), 6),
            "matched_chunks": len(s),
            "top_chunk_text": texts.get(cid, ""),
            "top_score":      round(max(s), 6),
        }
        for cid, s in scores.items()
    ]
    return sorted(ranked, key=lambda x: x["score"], reverse=True)
