"""
conversation/rewriter.py — Nyaya-Setu Conversation Brain
=========================================================
Converts ConversationMemory into optimized search inputs
for the Knowledge Brain:

1. A LegalQuery object (for the existing pipeline)
2. A rewritten FAISS search string (retrieval-optimized, not the user's raw text)
3. Structured fields for precise Neo4j Cypher queries

The rewriter is the BRIDGE between the Conversation Brain and
the Knowledge Brain. It ensures that retrieval operates on
structured legal context, not raw conversational noise.
"""

from __future__ import annotations

from conversation.memory import ConversationMemory
from query_understanding.schema import LegalQuery


def rewrite_for_retrieval(memory: ConversationMemory) -> LegalQuery:
    """
    Convert conversation memory into a LegalQuery object
    ready for the Knowledge Brain pipeline.

    Parameters
    ----------
    memory : filled ConversationMemory from the Conversation Brain

    Returns
    -------
    LegalQuery — structured legal query for FAISS + Neo4j retrieval
    """
    # Build a rich, retrieval-optimized search sentence
    search_parts = []

    if memory.incident:
        search_parts.append(memory.incident)

    if memory.vehicle:
        search_parts.append(f"involving {memory.vehicle.lower()}")

    if memory.injury:
        search_parts.append(f"{memory.injury.lower()} injury")

    for sec in memory.sections:
        search_parts.append(f"Section {sec}")

    for act in memory.acts:
        search_parts.append(act)

    if memory.state:
        search_parts.append(f"in {memory.state}")

    if memory.current_goal:
        search_parts.append(memory.current_goal.lower())

    # Include any custom facts that might be search-relevant
    for key, val in memory.custom_facts.items():
        if isinstance(val, str) and len(val) > 2:
            search_parts.append(val)

    search_text = " ".join(search_parts) if search_parts else "legal dispute"

    # Build expanded concepts from the incident type
    concepts = _get_expanded_concepts(memory.incident, memory.legal_domain)

    # Build keywords from filled slots
    keywords = []
    if memory.incident:
        keywords.append(memory.incident.lower())
    if memory.vehicle:
        keywords.append(memory.vehicle.lower())
    if memory.injury:
        keywords.append(memory.injury.lower())
    if memory.fir_filed is not None:
        keywords.append("FIR" if memory.fir_filed else "no FIR")

    # Assemble the original user query from conversation history
    user_messages = [
        m["content"] for m in memory.messages if m["role"] == "user"
    ]
    original_query = " | ".join(user_messages[-3:]) if user_messages else search_text

    legal_query = LegalQuery(
        original_query=original_query,
        legal_domain=memory.legal_domain or "General",
        incident_type=memory.incident or "Legal Dispute",
        keywords=keywords,
        suggested_acts=memory.acts,
        suggested_sections=memory.sections,
        key_entities=[memory.vehicle] if memory.vehicle else [],
        expanded_concepts=concepts,
    )

    print(f"[rewriter] Rewritten search: '{search_text}'")
    print(f"[rewriter] LegalQuery: domain={legal_query.legal_domain}, "
          f"incident={legal_query.incident_type}, "
          f"sections={legal_query.suggested_sections}")

    return legal_query


def build_faiss_query(memory: ConversationMemory) -> str:
    """
    Build a plain-text query optimized for FAISS semantic search.
    This is NOT the user's raw message — it's a retrieval-optimized
    rewrite based on structured memory.

    Returns
    -------
    str — the query string to embed and search FAISS with
    """
    parts = []

    if memory.incident:
        parts.append(memory.incident)

    if memory.vehicle:
        parts.append(f"{memory.vehicle} accident")

    for sec in memory.sections:
        parts.append(f"Section {sec}")

    for act in memory.acts:
        parts.append(act)

    if memory.injury:
        parts.append(f"{memory.injury} injury")

    if memory.current_goal:
        parts.append(memory.current_goal)

    if memory.state:
        parts.append(memory.state)

    return " ".join(parts) if parts else "legal dispute India"


# ---------------------------------------------------------------------------
# Concept expansion (enriches FAISS retrieval quality)
# ---------------------------------------------------------------------------

_DOMAIN_CONCEPTS: dict[str, list[str]] = {
    "Motor Vehicles": [
        "tort liability", "negligence", "compensation",
        "contributory negligence", "motor accident claim",
        "rash and negligent driving", "third party insurance",
    ],
    "Criminal": [
        "cognizable offence", "mens rea", "actus reus",
        "burden of proof", "FIR", "bail", "anticipatory bail",
    ],
    "Criminal/Tort": [
        "strict liability", "nuisance", "negligence",
        "animal owner liability", "municipal corporation duty",
    ],
    "Property": [
        "wrongful eviction", "tenancy rights", "notice to quit",
        "mesne profits", "adverse possession", "title dispute",
    ],
    "Family": [
        "maintenance", "custody", "divorce petition",
        "domestic violence", "protection order",
    ],
    "Consumer": [
        "deficiency in service", "unfair trade practice",
        "product liability", "consumer rights", "compensation",
    ],
    "Labour": [
        "wrongful termination", "retrenchment",
        "reinstatement", "back wages", "industrial dispute",
    ],
}


def _get_expanded_concepts(
    incident: str | None, domain: str | None
) -> list[str]:
    """Get legal doctrine concepts for the given domain."""
    concepts = []
    if domain and domain in _DOMAIN_CONCEPTS:
        concepts.extend(_DOMAIN_CONCEPTS[domain])
    if incident:
        concepts.append(incident.lower())
    return concepts[:8]  # cap to avoid prompt bloat
