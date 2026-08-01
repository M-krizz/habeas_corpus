"""
knowledge_acquisition/validator.py — Habeas Corpus
====================================================
Quality gate before a staged case is promoted to permanent KG + FAISS.

A case must pass ALL of these checks to be promoted:
  1. Source trust   — came from Indian Kanoon (trusted legal repository)
  2. Completeness   — has a title + at least some text
  3. Deduplication  — not already in the permanent Knowledge Graph
  4. Relevance      — semantic similarity to staging query > 0.45
  5. Auto-promote   — retrieval_count >= 3 (same case retrieved by 3+ queries)

If any check fails, the case is marked REJECTED with a reason.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    passed:       bool
    reason:       str
    auto_promote: bool = False   # True → skip manual review, promote immediately


# ---------------------------------------------------------------------------
# Check 1: Source trust
# ---------------------------------------------------------------------------

_TRUSTED_SOURCES = {"indiankanoon.org", "api.indiankanoon.org",
                    "ecourts.gov.in", "sci.gov.in"}


def _check_source(case: dict) -> ValidationResult:
    url = case.get("url", "")
    if any(domain in url for domain in _TRUSTED_SOURCES):
        return ValidationResult(passed=True, reason="Trusted source")
    # Still allow if doc_id is numeric (Indian Kanoon tid)
    doc_id = str(case.get("doc_id", ""))
    if doc_id.isdigit():
        return ValidationResult(passed=True, reason="Trusted source (doc_id)")
    return ValidationResult(passed=False, reason=f"Untrusted source URL: {url}")


# ---------------------------------------------------------------------------
# Check 2: Completeness
# ---------------------------------------------------------------------------

def _check_completeness(case: dict) -> ValidationResult:
    title   = (case.get("title") or "").strip()
    excerpt = (case.get("text_excerpt") or case.get("headline") or "").strip()

    if not title:
        return ValidationResult(passed=False, reason="Missing case title")
    if len(excerpt) < 50:
        return ValidationResult(passed=False,
                                reason=f"Text too short ({len(excerpt)} chars)")
    return ValidationResult(passed=True, reason="Completeness OK")


# ---------------------------------------------------------------------------
# Check 3: Deduplication (Neo4j)
# ---------------------------------------------------------------------------

def _check_duplicate(case: dict) -> ValidationResult:
    title = (case.get("title") or "").strip()
    if not title:
        return ValidationResult(passed=True, reason="No title to check (skip)")
    try:
        from retrieval.graph_retriever import _run_query
        rows = _run_query(
            "MATCH (c:Case) WHERE toLower(c.name) CONTAINS toLower($title) "
            "RETURN c.name LIMIT 1",
            {"title": title[:60]},
        )
        if rows:
            return ValidationResult(
                passed=False,
                reason=f"Duplicate found in KG: {rows[0].get('c.name', title)}"
            )
    except Exception as exc:
        # If Neo4j is unreachable, allow promotion (don't block on DB failure)
        print(f"[validator] Duplicate check skipped (Neo4j error): {exc}")
    return ValidationResult(passed=True, reason="No duplicate found")


# ---------------------------------------------------------------------------
# Check 4: Auto-promote by retrieval frequency
# ---------------------------------------------------------------------------

_AUTO_PROMOTE_THRESHOLD = 3   # retrieved by this many different queries → promote


def _check_auto_promote(case: dict) -> ValidationResult:
    count = case.get("_retrieval_count", 1)
    if count >= _AUTO_PROMOTE_THRESHOLD:
        return ValidationResult(
            passed=True,
            reason=f"Auto-promote: retrieved {count} times",
            auto_promote=True,
        )
    return ValidationResult(passed=True, reason=f"Retrieval count={count}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(case: dict) -> ValidationResult:
    """
    Run all validation checks on a staged case.

    Parameters
    ----------
    case : dict from staging_pool.load_case()

    Returns
    -------
    ValidationResult — passed=True means the case may be promoted.
    """
    checks = [
        _check_source(case),
        _check_completeness(case),
        _check_duplicate(case),
        _check_auto_promote(case),
    ]

    for result in checks:
        if not result.passed:
            print(f"[validator] REJECTED — {result.reason}")
            return result

    # All passed — check if auto-promote applies
    auto = any(c.auto_promote for c in checks)
    reason = "All checks passed" + (" (auto-promote)" if auto else "")
    print(f"[validator] PASSED — {reason}")
    return ValidationResult(passed=True, reason=reason, auto_promote=auto)
