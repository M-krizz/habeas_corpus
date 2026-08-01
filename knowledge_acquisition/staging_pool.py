"""
knowledge_acquisition/staging_pool.py — Habeas Corpus
=======================================================
Temporary workspace for newly retrieved web cases.

Cases land here FIRST before any permanent storage.
Only after validation do they get promoted to Neo4j + FAISS.

Storage: knowledge_acquisition/staging/<hash>.json
Index:   knowledge_acquisition/staging/index.json

Status lifecycle:
    pending → validated → promoted
    pending → rejected
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

_STAGING_DIR = Path(__file__).parent / "staging"

# Status constants
STATUS_PENDING   = "pending"
STATUS_VALIDATED = "validated"
STATUS_PROMOTED  = "promoted"
STATUS_REJECTED  = "rejected"


def _ensure_dir() -> None:
    _STAGING_DIR.mkdir(parents=True, exist_ok=True)


def _index_path() -> Path:
    return _STAGING_DIR / "index.json"


def _load_index() -> dict:
    p = _index_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_index(index: dict) -> None:
    _ensure_dir()
    _index_path().write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _case_hash(doc_id: str, title: str) -> str:
    """Stable hash for deduplication."""
    key = f"{doc_id}:{title}".lower().strip()
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stage_case(web_case: dict) -> str:
    """
    Add a web-retrieved case to the staging pool.

    Parameters
    ----------
    web_case : dict from web_retriever.search_indian_kanoon()

    Returns
    -------
    str — the staging hash (used to look up / update status later)
    """
    _ensure_dir()
    h = _case_hash(web_case.get("doc_id", ""), web_case.get("title", ""))

    index = _load_index()
    if h in index and index[h]["status"] in (STATUS_PROMOTED, STATUS_VALIDATED):
        print(f"[staging_pool] Case {h} already staged/promoted — skipping.")
        return h

    record = {
        **web_case,
        "_staging_hash":     h,
        "_staged_at":        datetime.now(timezone.utc).isoformat(),
        "_status":           STATUS_PENDING,
        "_retrieval_count":  index.get(h, {}).get("_retrieval_count", 0) + 1,
    }

    # Write individual case file
    case_file = _STAGING_DIR / f"{h}.json"
    case_file.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Update index
    index[h] = {
        "doc_id":           web_case.get("doc_id", ""),
        "title":            web_case.get("title", "")[:120],
        "status":           STATUS_PENDING,
        "staged_at":        record["_staged_at"],
        "retrieval_count":  record["_retrieval_count"],
    }
    _save_index(index)
    print(f"[staging_pool] Staged case '{web_case.get('title','')[:60]}' -> {h}")
    return h


def update_status(staging_hash: str, status: str, reason: str = "") -> None:
    """Update the status of a staged case."""
    index = _load_index()
    if staging_hash not in index:
        return
    index[staging_hash]["status"] = status
    if reason:
        index[staging_hash]["rejection_reason"] = reason
    _save_index(index)

    # Also update the individual file
    case_file = _STAGING_DIR / f"{staging_hash}.json"
    if case_file.exists():
        try:
            data = json.loads(case_file.read_text(encoding="utf-8"))
            data["_status"] = status
            if reason:
                data["_rejection_reason"] = reason
            case_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass


def load_case(staging_hash: str) -> dict | None:
    """Load a staged case by its hash."""
    case_file = _STAGING_DIR / f"{staging_hash}.json"
    if not case_file.exists():
        return None
    try:
        return json.loads(case_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_pending_cases() -> list[dict]:
    """Return all cases currently in 'pending' status."""
    index = _load_index()
    pending_hashes = [h for h, v in index.items()
                      if v.get("status") == STATUS_PENDING]
    cases = []
    for h in pending_hashes:
        case = load_case(h)
        if case:
            cases.append(case)
    return cases


def get_staging_summary() -> list[dict]:
    """Return a summary of all staged cases (for the API /staging/status endpoint)."""
    index = _load_index()
    return [
        {
            "hash":             h,
            "title":            v.get("title", ""),
            "status":           v.get("status", ""),
            "staged_at":        v.get("staged_at", ""),
            "retrieval_count":  v.get("retrieval_count", 1),
        }
        for h, v in sorted(index.items(),
                            key=lambda x: x[1].get("staged_at", ""),
                            reverse=True)
    ]
