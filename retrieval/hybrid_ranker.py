"""
retrieval/hybrid_ranker.py — Habeas Corpus
==========================================
Fuses semantic (FAISS) scores and graph (Neo4j) scores into a single
ranked list of candidate cases.

Fusion formula:
    final_score = α × semantic_score_normalised + β × graph_score
    α = 0.65  (semantic preferred — captures meaning)
    β = 0.35  (graph confirms legal domain)

Both inputs must already be normalised to [0, 1].  FAISS returns cosine
similarities which are already in this range for normalised vectors.
Graph scores are normalised in graph_retriever.py.
"""

from __future__ import annotations


# Fusion weights — adjust these without touching any other module
_ALPHA = 0.65   # semantic weight
_BETA  = 0.35   # graph weight


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hybrid_rank(
    semantic_hits: list[dict],
    graph_hits:    list[dict],
    top_k:         int = 10,
) -> list[dict]:
    """
    Merge and re-rank FAISS semantic hits with Neo4j graph hits.

    Parameters
    ----------
    semantic_hits : from semantic_retrieval.ranker.rank_cases()
        Each dict must have: case_id, score (0–1), matched_chunks,
        top_chunk_text, top_score.
    graph_hits : from retrieval.graph_retriever.graph_search()
        Each dict must have: case_id, graph_score (0–1),
        matched_acts, matched_sections, matched_keywords.
    top_k : number of top merged results to return

    Returns
    -------
    list of dict (sorted desc by final_score), each:
        {
          case_id, final_score, semantic_score, graph_score,
          matched_chunks, matched_acts, matched_sections,
          top_chunk_text
        }
    """
    # Index both lists by case_id
    sem_by_id: dict[str, dict] = {h["case_id"]: h for h in semantic_hits}
    gph_by_id: dict[str, dict] = {h["case_id"]: h for h in graph_hits}

    # Normalise semantic scores (sum strategy → can exceed 1.0 for multi-chunk cases)
    if sem_by_id:
        max_sem = max(h["score"] for h in sem_by_id.values()) or 1.0
    else:
        max_sem = 1.0

    # Union of all case IDs from both sources
    all_case_ids = set(sem_by_id) | set(gph_by_id)

    merged: list[dict] = []
    for cid in all_case_ids:
        sem  = sem_by_id.get(cid, {})
        gph  = gph_by_id.get(cid, {})

        sem_score = (sem.get("score", 0.0) / max_sem)   # normalised to [0,1]
        gph_score = gph.get("graph_score", 0.0)

        final = round(_ALPHA * sem_score + _BETA * gph_score, 6)

        merged.append({
            "case_id":          cid,
            "final_score":      final,
            "semantic_score":   round(sem_score, 6),
            "graph_score":      round(gph_score, 6),
            "matched_chunks":   sem.get("matched_chunks", 0),
            "matched_acts":     gph.get("matched_acts", 0),
            "matched_sections": gph.get("matched_sections", 0),
            "top_chunk_text":   sem.get("top_chunk_text", ""),
        })

    ranked = sorted(merged, key=lambda x: x["final_score"], reverse=True)
    result = ranked[:top_k]

    print(f"[hybrid_ranker] {len(all_case_ids)} unique cases -> "
          f"top {len(result)} after hybrid ranking.")
    for r in result[:3]:
        print(f"  {r['case_id']:40s}  "
              f"final={r['final_score']:.3f}  "
              f"sem={r['semantic_score']:.3f}  "
              f"gph={r['graph_score']:.3f}")

    return result
