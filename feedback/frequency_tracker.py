"""
feedback/frequency_tracker.py — Habeas Corpus
===============================================
Tracks how often each staging case is retrieved across different queries.
When a case is retrieved >= 3 times, it is eligible for auto-promotion.

The retrieval count is already tracked in the staging pool's index.json
(_retrieval_count field). This module provides the convenience check.
"""

from __future__ import annotations

AUTO_PROMOTE_THRESHOLD = 3


def should_auto_promote(retrieval_count: int) -> bool:
    """Return True if the retrieval count meets the auto-promotion threshold."""
    return retrieval_count >= AUTO_PROMOTE_THRESHOLD


def increment_retrieval_count(staging_hash: str) -> int:
    """
    Increment the retrieval count for a staging case and return the new count.
    Used when the same web case is surfaced for a different query.
    """
    from knowledge_acquisition.staging_pool import _load_index, _save_index
    index = _load_index()
    if staging_hash in index:
        count = index[staging_hash].get("retrieval_count", 1) + 1
        index[staging_hash]["retrieval_count"] = count
        _save_index(index)

        # Also update the case file
        from knowledge_acquisition.staging_pool import load_case, _STAGING_DIR
        import json
        case_file = _STAGING_DIR / f"{staging_hash}.json"
        if case_file.exists():
            try:
                data = json.loads(case_file.read_text(encoding="utf-8"))
                data["_retrieval_count"] = count
                case_file.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
        return count
    return 1
