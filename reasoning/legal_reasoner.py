"""
reasoning/legal_reasoner.py — Habeas Corpus
=============================================
The top-level reasoning orchestrator.

Wires together: prompt_builder → llm_client → response parser
→ ReasoningResponse

The LLM only sees:
  - The user's original question
  - The structured LegalQuery (concepts, acts, sections)
  - The assembled CaseEvidence objects (text passages + metadata)

It NEVER sees:
  - The raw FAISS index
  - Raw Neo4j Cypher
  - Unverified web content

This is by design — every fact the LLM uses is already in your
verified Knowledge Graph.
"""

from __future__ import annotations

import json
import re

from evidence.schema import CaseEvidence
from query_understanding.schema import LegalQuery, ReasoningResponse
from reasoning.prompt_builder import build_prompt
from reasoning.llm_client import generate


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(raw: str, legal_query: LegalQuery,
                    evidence: list[CaseEvidence],
                    confidence: float) -> ReasoningResponse:
    """
    Parse the LLM's JSON response into a ReasoningResponse.
    Has multiple fallback strategies to avoid hard crashes.
    """
    # Strip accidental markdown fences
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from within the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    # Build the response — use LLM data where available, KG data as fallback
    all_acts     = data.get("applicable_acts", []) or \
                   list({a for ev in evidence for a in ev.acts})
    all_sections = data.get("applicable_sections", []) or \
                   list({s for ev in evidence for s in ev.sections})

    precedents = data.get("precedents", [])
    if not precedents:
        # Build precedents from CaseEvidence if LLM didn't produce them
        precedents = [
            {
                "case":      ev.case_name or ev.case_id,
                "court":     ev.court,
                "date":      ev.decision_date,
                "held":      ev.top_chunk_text[:300] if ev.top_chunk_text else "",
                "relevance": f"Score: {ev.final_score:.2f}",
            }
            for ev in evidence
        ]

    return ReasoningResponse(
        summary            = data.get("summary", "Insufficient evidence found."),
        applicable_acts    = all_acts,
        applicable_sections = all_sections,
        precedents         = precedents,
        confidence         = confidence,
        knowledge_source   = _determine_source(evidence),
        disclaimer         = data.get(
            "disclaimer",
            "This is AI-assisted legal research, not legal advice. "
            "Consult a qualified advocate."
        ),
        legal_query        = legal_query,
    )


def _determine_source(evidence: list[CaseEvidence]) -> str:
    sources = {ev.source for ev in evidence}
    if sources == {"permanent_kg"}:
        return "permanent_kg"
    if "web_retrieved" in sources and "permanent_kg" in sources:
        return "hybrid"
    if "web_retrieved" in sources:
        return "temporary_pool"
    return "permanent_kg"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reason(
    legal_query: LegalQuery,
    evidence:    list[CaseEvidence],
    confidence:  float = 1.0,
) -> ReasoningResponse:
    """
    Run the full LLM reasoning step.

    Parameters
    ----------
    legal_query : structured query from the concept mapper
    evidence    : assembled case evidence from the aggregator
    confidence  : retrieval confidence score (passed through to response)

    Returns
    -------
    ReasoningResponse — structured, LLM-generated answer
    """
    print(f"\n[legal_reasoner] Reasoning over {len(evidence)} case(s)...")

    # Build the fallback context (used if LLM is unavailable)
    fallback_ctx = {
        "n_cases":   len(evidence),
        "incident":  legal_query.incident_type,
        "acts":      list({a for ev in evidence for a in ev.acts}),
        "sections":  list({s for ev in evidence for s in ev.sections}),
        "precedents": [
            {"case": ev.case_name or ev.case_id, "court": ev.court,
             "date": ev.decision_date, "held": ev.top_chunk_text[:200],
             "relevance": f"Relevance score: {ev.final_score:.2f}"}
            for ev in evidence
        ],
    }

    # Build prompt
    prompt = build_prompt(legal_query, evidence)

    # Call LLM
    raw = generate(prompt, fallback_context=fallback_ctx)

    # Parse and return
    response = _parse_response(raw, legal_query, evidence, confidence)
    print(f"[legal_reasoner] Done. Summary length: {len(response.summary)} chars.")
    return response
