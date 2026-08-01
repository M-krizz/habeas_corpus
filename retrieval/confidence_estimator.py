"""
retrieval/confidence_estimator.py — Habeas Corpus
==================================================
Estimates how confident the system is in its current retrieval results.

If confidence falls below the threshold, the pipeline triggers the
Adaptive Knowledge Acquisition module to search Indian Kanoon.

The confidence score is a function of:
    1. Top hybrid score       — how strong is the best match?
    2. Score gap              — is the top result clearly better than #2?
    3. Coverage               — how many cases matched at all?
    4. Graph confirmation     — did the graph search also find the case?

Threshold: 0.60
    Below this → trigger web fallback.
    Above this → use permanent KG directly.
"""

from __future__ import annotations

# Confidence threshold below which adaptive acquisition fires
CONFIDENCE_THRESHOLD = 0.72


def estimate_confidence(ranked_results: list[dict]) -> float:
    """
    Compute a retrieval confidence score in [0, 1].

    Parameters
    ----------
    ranked_results : output of hybrid_ranker.hybrid_rank()
        Each dict must have: final_score, graph_score, semantic_score.

    Returns
    -------
    float in [0, 1]  — higher is more confident.
    """
    if not ranked_results:
        return 0.0

    top_score = ranked_results[0]["final_score"]

    # Factor 1: absolute top score (0–1). If top score < 0.55, heavily discount.
    f1 = top_score if top_score > 0.55 else top_score * 0.5

    # Factor 2: score gap (top vs second — rewards clear winners when top is high)
    if len(ranked_results) >= 2:
        gap = top_score - ranked_results[1]["final_score"]
        f2  = min(gap * 2, 1.0) if top_score > 0.60 else 0.2
    else:
        f2 = 0.5   # only one result -> lack of corroborating evidence

    # Factor 3: coverage (count ONLY solid matches with score > 0.55)
    strong_matches = sum(1 for r in ranked_results if r["final_score"] > 0.55)
    f3 = min(strong_matches / 3, 1.0)

    # Factor 4: graph confirmation (did structural graph search confirm relevance?)
    top_gph = ranked_results[0].get("graph_score", 0.0)
    f4 = 1.0 if top_gph > 0.15 else 0.0

    # Weighted combination: 45% top match, 20% gap, 20% quality coverage, 15% graph synergy
    confidence = 0.45 * f1 + 0.20 * f2 + 0.20 * f3 + 0.15 * f4
    confidence = round(min(confidence, 1.0), 4)

    print(f"[confidence_estimator] Score={confidence:.3f}  "
          f"(top={f1:.3f}, gap={f2:.3f}, cov={f3:.3f}, graph={f4:.1f})")

    return confidence


def is_confident(ranked_results: list[dict]) -> bool:
    """
    Returns True if retrieval confidence is above the threshold.
    Use this as the decision gate in the pipeline.
    """
    return estimate_confidence(ranked_results) >= CONFIDENCE_THRESHOLD
