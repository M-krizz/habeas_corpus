"""
knowledge_acquisition/web_retriever.py — Habeas Corpus
========================================================
Retrieves judgments from Indian Kanoon API when the local KG
has insufficient evidence (confidence < 0.60).

API docs: https://api.indiankanoon.org/
Auth: Bearer token in Authorization header.

We retrieve a maximum of 5 judgments per query — not hundreds.
Quality over quantity prevents the staging pool from being polluted
with irrelevant cases.
"""

from __future__ import annotations

import os
import re

import httpx
from dotenv import load_dotenv

from query_understanding.schema import LegalQuery

load_dotenv()

_BASE_URL    = "https://api.indiankanoon.org"
_MAX_RESULTS = 5
_TIMEOUT     = 15.0   # seconds


def _get_headers() -> dict[str, str]:
    token = os.getenv("INDIANKANOON_API_KEY", "").strip()
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"
    return headers


def _build_search_query(legal_query: LegalQuery) -> str:
    """
    Build an Indian Kanoon search string from the LegalQuery.
    Combines acts, sections, incident type, and top keywords.
    """
    parts = []

    # Most specific → least specific
    for sec in legal_query.suggested_sections[:3]:
        parts.append(f"section {sec}")

    for act in legal_query.suggested_acts[:2]:
        # Indian Kanoon works best with short act names
        short = act.replace(" Act", "").strip()
        parts.append(short)

    parts.append(legal_query.incident_type)
    parts += legal_query.keywords[:4]

    return " ".join(parts)


def _clean_headline(text: str) -> str:
    """Strip HTML tags from Indian Kanoon headline text."""
    return re.sub(r"<[^>]+>", "", text or "").strip()


def search_indian_kanoon(
    legal_query: LegalQuery,
    max_results: int = _MAX_RESULTS,
) -> list[dict]:
    """
    Search Indian Kanoon API for relevant judgments.

    Parameters
    ----------
    legal_query : structured LegalQuery from the concept mapper
    max_results : cap on returned results (default 5)

    Returns
    -------
    list of dict, each:
        {doc_id, title, headline, url, court, date,
         text_excerpt, acts, sections, relevance_score}

    Returns empty list on any API error — pipeline must handle this.
    """
    search_q = _build_search_query(legal_query)
    print(f"[web_retriever] Searching Indian Kanoon: '{search_q}'")

    try:
        resp = httpx.post(
            f"{_BASE_URL}/search/",
            data={"formInput": search_q, "pagenum": 0},
            headers=_get_headers(),
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        print(f"[web_retriever] HTTP {exc.response.status_code}: {exc}")
        return []
    except Exception as exc:
        print(f"[web_retriever] Request failed: {exc}")
        return []

    docs = data.get("docs", [])[:max_results]
    results = []

    for doc in docs:
        doc_id   = str(doc.get("tid", ""))
        title    = doc.get("title", "")
        headline = _clean_headline(doc.get("headline", ""))
        url      = f"https://indiankanoon.org/doc/{doc_id}/"

        # Extract sections mentioned in headline
        mentioned_sections = re.findall(r"\bSection\s+(\d+[A-Za-z]?)\b",
                                         headline, re.IGNORECASE)

        results.append({
            "doc_id":          doc_id,
            "title":           title,
            "headline":        headline,
            "url":             url,
            "court":           doc.get("court", ""),
            "date":            doc.get("publishdate", ""),
            "text_excerpt":    headline[:600],
            "acts":            legal_query.suggested_acts,   # inferred from query
            "sections":        mentioned_sections or legal_query.suggested_sections,
            "relevance_score": 0.55,   # default web score (below permanent KG)
        })

    print(f"[web_retriever] Retrieved {len(results)} results from Indian Kanoon.")
    return results


def search_tavily(
    legal_query: LegalQuery,
    max_results: int = 5,
) -> list[dict]:
    """
    Search Tavily Web Search API for Indian legal precedents and court judgments.
    Used as an external retrieval engine when local KG confidence is low or when unlinked cases require external verification.
    """
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not tavily_key:
        print("[web_retriever] TAVILY_API_KEY not configured.")
        return []

    # Build search query for Tavily
    query_parts = ["Indian Supreme Court High Court legal precedent judgment"]
    if legal_query.incident_type:
        query_parts.append(legal_query.incident_type)
    for sec in legal_query.suggested_sections[:3]:
        query_parts.append(f"Section {sec}")
    for act in legal_query.suggested_acts[:2]:
        query_parts.append(act)
    if hasattr(legal_query, "search_text") and legal_query.search_text:
        query_parts.append(legal_query.search_text[:100])

    tavily_q = " ".join(query_parts)
    print(f"[web_retriever] Searching Tavily: '{tavily_q}'")

    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": tavily_key,
                "query": tavily_q,
                "max_results": max_results,
                "search_depth": "advanced",
                "include_answer": False,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[web_retriever] Tavily search failed: {exc}")
        return []

    results = []
    raw_results = data.get("results", [])
    for idx, item in enumerate(raw_results):
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")

        # Extract section numbers mentioned in content
        sec_matches = re.findall(r"\bSection\s+(\d+[A-Za-z]?)\b", content, re.IGNORECASE)

        # Infer court name from URL or title
        court = "Supreme Court of India"
        if "highcourt" in url.lower() or "hc" in url.lower() or "high court" in title.lower():
            court = "High Court"

        doc_id = f"tavily_{idx+1}_{abs(hash(url)) & 0xffffff}"

        results.append({
            "doc_id": doc_id,
            "title": title,
            "headline": content[:400],
            "url": url,
            "court": court,
            "date": "Recent",
            "text_excerpt": content[:800],
            "acts": legal_query.suggested_acts,
            "sections": sec_matches or legal_query.suggested_sections,
            "relevance_score": 0.65,
        })

    print(f"[web_retriever] Retrieved {len(results)} results from Tavily.")
    return results


def search_external_cases(
    legal_query: LegalQuery,
    max_results: int = 5,
) -> list[dict]:
    """
    Primary entry point for external case retrieval.
    Tries Tavily first (if configured), then Indian Kanoon as fallback/supplement.
    """
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    results = []

    if tavily_key:
        results = search_tavily(legal_query, max_results=max_results)

    if not results:
        results = search_indian_kanoon(legal_query, max_results=max_results)

    return results


def fetch_full_document(doc_id: str) -> str:
    """
    Fetch the full text of a single Indian Kanoon document.
    Used by the promoter when staging a case for permanent storage.

    Returns the plain text (HTML tags stripped), or empty string on failure.
    """
    try:
        resp = httpx.post(
            f"{_BASE_URL}/doc/{doc_id}/",
            headers=_get_headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        html_text = data.get("doc", "")
        # Strip HTML
        plain = re.sub(r"<[^>]+>", " ", html_text)
        plain = re.sub(r"\s+", " ", plain).strip()
        return plain
    except Exception as exc:
        print(f"[web_retriever] Full doc fetch failed for {doc_id}: {exc}")
        return ""
