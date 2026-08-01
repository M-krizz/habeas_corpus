"""
api/pipeline.py — Habeas Corpus
=================================
Master orchestrator. Called by the FastAPI /query endpoint.

Implements the full pipeline:

    User Query
        │
        ▼
    Legal Concept Mapper        (query_understanding)
        │
        ├──► FAISS Semantic Search   (semantic_retrieval)
        ├──► Neo4j Graph Search      (retrieval.graph_retriever)
        │
        ▼
    Hybrid Ranking              (retrieval.hybrid_ranker)
        │
        ▼
    Feedback Boost              (feedback.reranker)
        │
        ▼
    Confidence Estimation       (retrieval.confidence_estimator)
        │
    ┌───┴────────────────────────────┐
    │ High confidence (≥ 0.60)       │ Low confidence (< 0.60)
    ▼                                ▼
    Evidence Aggregator         Indian Kanoon Retrieval
    (permanent KG)              → Stage → Temp Evidence
        │                                │
        └──────────────┬─────────────────┘
                       ▼
                LLM Legal Reasoner     (reasoning)
                       │
                       ▼
                ReasoningResponse
                       │
                       ▼
        Background Indexer (async)  ← promotes staging cases later
"""

from __future__ import annotations

from query_understanding.legal_concept_mapper import map_legal_concepts
from query_understanding.schema import ReasoningResponse
from semantic_retrieval.pipeline import search_cases as faiss_search_cases
from retrieval.graph_retriever import graph_search
from retrieval.hybrid_ranker import hybrid_rank
from retrieval.confidence_estimator import estimate_confidence, CONFIDENCE_THRESHOLD
from feedback.reranker import apply_feedback_boost
from evidence.aggregator import aggregate_evidence, aggregate_from_web
from reasoning.legal_reasoner import reason


async def run_query(raw_query: str) -> tuple[ReasoningResponse, list[str]]:
    """
    Run the complete Habeas Corpus pipeline for a user query.

    Parameters
    ----------
    raw_query : plain-English user query string

    Returns
    -------
    tuple of:
        ReasoningResponse  — the full structured answer
        list[str]          — staging hashes of new cases added (may be empty)
                             used by the caller to schedule background indexing
    """
    print(f"\n{'='*60}")
    print(f"[pipeline] New query: '{raw_query}'")
    print(f"{'='*60}")

    # ── Step 1: Legal Concept Mapping ───────────────────────────────
    legal_query = map_legal_concepts(raw_query)

    # ── Step 2: Parallel Retrieval ──────────────────────────────────
    # FAISS: retrieve top 20 semantic chunks → case-level scores
    faiss_results = faiss_search_cases(
        query  = legal_query.search_text,
        top_k  = 20,
        n      = 15,
    )

    # Neo4j: search by acts, sections, keywords
    graph_results = graph_search(legal_query, top_k=15)

    # ── Step 3: Hybrid Ranking ──────────────────────────────────────
    ranked = hybrid_rank(faiss_results, graph_results, top_k=10)

    # ── Step 4: Feedback Boost (Learning-to-Rank) ───────────────────
    ranked = apply_feedback_boost(ranked)

    # ── Step 5: Confidence Estimation ──────────────────────────────
    confidence = estimate_confidence(ranked)
    print(f"[pipeline] Confidence: {confidence:.3f} "
          f"(threshold={CONFIDENCE_THRESHOLD})")

    # ── Step 6: Evidence Assembly ───────────────────────────────────
    new_staging_hashes: list[str] = []

    if confidence >= CONFIDENCE_THRESHOLD:
        # High confidence → use permanent KG
        evidence = aggregate_evidence(ranked, top_n=5)
        knowledge_source = "permanent_kg"

    else:
        # Low confidence → Indian Kanoon fallback
        print(f"[pipeline] Low confidence → triggering Indian Kanoon retrieval...")
        from knowledge_acquisition.web_retriever import search_indian_kanoon
        from knowledge_acquisition.staging_pool import stage_case
        from evidence.aggregator import aggregate_from_web

        web_results = search_indian_kanoon(legal_query, max_results=5)

        for wc in web_results:
            h = stage_case(wc)
            new_staging_hashes.append(h)

        # Combine: any existing KG evidence first, then web evidence
        kg_evidence  = aggregate_evidence(ranked[:2], top_n=2) if ranked else []
        web_evidence = aggregate_from_web(web_results)
        evidence     = kg_evidence + web_evidence
        knowledge_source = "hybrid" if kg_evidence else "temporary_pool"

    # ── Step 7: LLM Reasoning ───────────────────────────────────────
    response = reason(legal_query, evidence, confidence)
    response.knowledge_source = knowledge_source

    print(f"[pipeline] Done. Source={knowledge_source}, "
          f"Precedents={len(response.precedents)}, "
          f"New staging={len(new_staging_hashes)}")

    return response, new_staging_hashes
