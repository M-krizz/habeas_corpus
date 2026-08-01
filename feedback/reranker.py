"""
feedback/reranker.py — Habeas Corpus
======================================
Adjusts case scores based on historical user feedback.

This is Learning-to-Rank without retraining any model.
The embedding model and FAISS index stay untouched.
We only shift the final_score of each case by a small boost/penalty
derived from historical 👍 / 👎 signals.

Boost cap  : +0.12  (a helpful case can't jump too many positions)
Penalty cap: -0.10  (an unhelpful case is penalised but not buried)
"""

from __future__ import annotations

from feedback.store import get_case_ratings

_BOOST_PER_VOTE   = 0.03
_MAX_BOOST        = 0.12
_PENALTY_PER_VOTE = 0.025
_MAX_PENALTY      = 0.10


def apply_feedback_boost(ranked: list[dict]) -> list[dict]:
    """
    Adjust final_score of each case based on accumulated user feedback.

    Parameters
    ----------
    ranked : output of hybrid_ranker.hybrid_rank()

    Returns
    -------
    Re-sorted list with feedback_boost field added to each entry.
    """
    ratings = get_case_ratings()   # {case_id: net_vote_count}

    if not ratings:
        return ranked   # no feedback yet — return unchanged

    adjusted = []
    for entry in ranked:
        cid  = entry["case_id"]
        net  = ratings.get(cid, 0.0)

        if net > 0:
            boost = min(net * _BOOST_PER_VOTE, _MAX_BOOST)
        elif net < 0:
            boost = max(net * _PENALTY_PER_VOTE, -_MAX_PENALTY)
        else:
            boost = 0.0

        new_entry = {
            **entry,
            "final_score":    round(min(entry["final_score"] + boost, 1.0), 6),
            "feedback_boost": round(boost, 4),
            "feedback_votes": int(net),
        }
        adjusted.append(new_entry)

    return sorted(adjusted, key=lambda x: x["final_score"], reverse=True)
