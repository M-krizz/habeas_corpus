"""
api/pipeline.py — Nyaya-Setu Knowledge Brain + Reasoning Brain
================================================================
Master orchestrator. Called by both:
  - POST /chat  (multi-turn: receives LegalQuery from Conversation Brain)
  - POST /query (legacy single-shot: receives raw query string)

Architecture:

    Conversation Brain
         │
         ▼
    LegalQuery (rewritten from structured memory)
         │
    ┌────┴────────────────┐
    │                     │
    ▼                     ▼
  FAISS Search      Neo4j Graph Search
    │                     │
    └────────┬────────────┘
             ▼
    Hybrid Ranking → Feedback Boost → Confidence Estimation
             │
    ┌────────┴────────────────────────┐
    │ High confidence (>= 0.72)      │ Low confidence (< 0.72)
    ▼                                ▼
    Evidence Aggregator         Indian Kanoon Retrieval
    (permanent KG)              → Stage → Temp Evidence
         │                                │
         └──────────────┬─────────────────┘
                        ▼
                 Reasoning Brain (LLM)
                        │
                        ▼
                 ReasoningResponse
"""

from __future__ import annotations

from conversation.memory import ConversationMemory
from query_understanding.legal_concept_mapper import map_legal_concepts
from query_understanding.schema import LegalQuery, ReasoningResponse
from semantic_retrieval.pipeline import search_cases as faiss_search_cases
from retrieval.graph_retriever import graph_search
from retrieval.hybrid_ranker import hybrid_rank
from retrieval.confidence_estimator import estimate_confidence, CONFIDENCE_THRESHOLD
from feedback.reranker import apply_feedback_boost
from evidence.aggregator import aggregate_evidence, aggregate_from_web
from reasoning.legal_reasoner import reason


async def run_query_with_legal_query(
    legal_query: LegalQuery,
    memory: ConversationMemory | None = None,
) -> tuple[ReasoningResponse, list[str]]:
    """
    Run the Knowledge Brain + Reasoning Brain using a pre-built LegalQuery.

    Called by the Conversation Brain after slot-filling is complete.
    The LegalQuery has been rewritten from structured memory — not from
    the user's raw text.

    Parameters
    ----------
    legal_query : structured legal query (from rewriter or concept mapper)
    memory      : optional ConversationMemory for Reasoning Brain context

    Returns
    -------
    tuple of:
        ReasoningResponse  — the full structured answer
        list[str]          — staging hashes of new cases added (may be empty)
    """
    print(f"\n{'='*60}")
    print(f"[pipeline] Processing query: domain={legal_query.legal_domain}, "
          f"incident={legal_query.incident_type}")
    print(f"{'='*60}")

    # ── Step 1: Parallel Retrieval ──────────────────────────────────
    # FAISS: use the rewritten search text (not raw user message)
    faiss_results = faiss_search_cases(
        query  = legal_query.search_text,
        top_k  = 20,
        n      = 15,
    )

    # Neo4j: search by acts, sections, keywords
    graph_results = graph_search(legal_query, top_k=15)

    # ── Step 2: Hybrid Ranking ──────────────────────────────────────
    ranked = hybrid_rank(faiss_results, graph_results, top_k=10)

    # ── Step 3: Feedback Boost (Learning-to-Rank) ───────────────────
    ranked = apply_feedback_boost(ranked)

    # ── Step 4: Confidence Estimation ──────────────────────────────
    confidence = estimate_confidence(ranked)
    print(f"[pipeline] Confidence: {confidence:.3f} "
          f"(threshold={CONFIDENCE_THRESHOLD})")

    # ── Step 5: Evidence Assembly ───────────────────────────────────
    new_staging_hashes: list[str] = []

    if confidence >= CONFIDENCE_THRESHOLD:
        # High confidence → use permanent KG
        evidence = aggregate_evidence(ranked, top_n=5)
        knowledge_source = "permanent_kg"

    else:
        # Low confidence or missing linked case → External Web Retrieval (Tavily / Indian Kanoon)
        print(f"[pipeline] Low confidence -> triggering external web retrieval (Tavily / Indian Kanoon)...")
        from knowledge_acquisition.web_retriever import search_external_cases
        from knowledge_acquisition.staging_pool import stage_case

        web_results = search_external_cases(legal_query, max_results=5)

        for wc in web_results:
            h = stage_case(wc)
            new_staging_hashes.append(h)

        # Combine: any existing KG evidence first, then web evidence
        kg_evidence  = aggregate_evidence(ranked[:2], top_n=2) if ranked else []
        web_evidence = aggregate_from_web(web_results)
        evidence     = kg_evidence + web_evidence
        knowledge_source = "hybrid" if kg_evidence else "temporary_pool"

    # ── Step 6: LLM Reasoning ───────────────────────────────────────
    response = reason(legal_query, evidence, confidence, memory=memory)
    response.knowledge_source = knowledge_source

    print(f"[pipeline] Done. Source={knowledge_source}, "
          f"Precedents={len(response.precedents)}, "
          f"New staging={len(new_staging_hashes)}")

    return response, new_staging_hashes


async def run_query(raw_query: str) -> tuple[ReasoningResponse, list[str]]:
    """
    Legacy single-shot pipeline entry point.

    Runs the Legal Concept Mapper on the raw query string to produce
    a LegalQuery, then delegates to run_query_with_legal_query().

    Parameters
    ----------
    raw_query : plain-English user query string

    Returns
    -------
    tuple of (ReasoningResponse, list[str] staging hashes)
    """
    print(f"\n{'='*60}")
    print(f"[pipeline] New query: '{raw_query}'")
    print(f"{'='*60}")

    # Step 1: Legal Concept Mapping (single-shot, no conversation memory)
    legal_query = map_legal_concepts(raw_query)

    # Step 2: Run through Knowledge + Reasoning Brains
    return await run_query_with_legal_query(legal_query, memory=None)
