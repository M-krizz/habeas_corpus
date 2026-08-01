"""
feedback/store.py — Habeas Corpus
===================================
Append-only feedback log stored as JSONL.
Simple, no DB required. One line per feedback event.
"""

from __future__ import annotations

import json
from pathlib import Path

from feedback.schema import FeedbackRecord

_FEEDBACK_FILE = Path(__file__).parent / "feedback_log.jsonl"


def record_feedback(fb: FeedbackRecord) -> None:
    """Append a feedback record to the JSONL log."""
    _FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(fb.model_dump(), ensure_ascii=False) + "\n")
    print(f"[feedback_store] Recorded: case={fb.case_id} rating={fb.rating:+d}")


def load_all_feedback() -> list[FeedbackRecord]:
    """Load all feedback records from the JSONL log."""
    if not _FEEDBACK_FILE.exists():
        return []
    records = []
    with open(_FEEDBACK_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(FeedbackRecord(**json.loads(line)))
                except Exception:
                    pass
    return records


def get_case_ratings() -> dict[str, float]:
    """
    Compute net rating per case_id across all feedback.
    Returns {case_id: net_score} where score is sum of +1 / -1 votes.
    """
    ratings: dict[str, float] = {}
    for fb in load_all_feedback():
        ratings[fb.case_id] = ratings.get(fb.case_id, 0.0) + fb.rating
    return ratings
