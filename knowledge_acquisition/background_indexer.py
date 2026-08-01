"""
knowledge_acquisition/background_indexer.py — Habeas Corpus
=============================================================
Background task that validates and promotes staging cases after
the user has already received their answer.

This is registered as a FastAPI BackgroundTask so it runs
asynchronously — the user never waits for it.

Flow:
    1. Get all pending staging cases
    2. Validate each one
    3. Promote passing cases → Neo4j + FAISS
    4. Reject failing cases with reason
"""

from __future__ import annotations

from knowledge_acquisition.staging_pool import get_pending_cases
from knowledge_acquisition.validator import validate
from knowledge_acquisition.promoter import promote


async def run_background_indexing(staging_hashes: list[str] | None = None) -> None:
    """
    Validate and promote pending staging cases.

    Parameters
    ----------
    staging_hashes : optional list of specific hashes to process.
                     If None, processes ALL pending cases.
    """
    if staging_hashes:
        from knowledge_acquisition.staging_pool import load_case
        pending = [c for h in staging_hashes
                   if (c := load_case(h)) is not None]
    else:
        pending = get_pending_cases()

    if not pending:
        print("[background_indexer] No pending staging cases.")
        return

    print(f"\n[background_indexer] Processing {len(pending)} pending case(s)...")
    promoted = 0
    rejected = 0

    for case in pending:
        h = case.get("_staging_hash", "")
        result = validate(case)

        if result.passed:
            success = promote(h, case)
            if success:
                promoted += 1
            else:
                rejected += 1
        else:
            from knowledge_acquisition.staging_pool import (
                update_status, STATUS_REJECTED
            )
            update_status(h, STATUS_REJECTED, result.reason)
            rejected += 1

    print(f"[background_indexer] Done — promoted={promoted}, rejected={rejected}")
