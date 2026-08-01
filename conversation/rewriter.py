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

import re

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
    # Assemble the original user query from conversation history
    user_messages = [
        m["content"] for m in memory.messages if m["role"] == "user"
    ]
    original_query = " | ".join(user_messages[-3:]) if user_messages else search_text

    # Default statutory acts and sections based on incident if empty
    acts = list(memory.acts)
    sections = list(memory.sections)

    if not acts or not sections:
        inc_lower = (memory.incident or "").lower()
        if "forge" in inc_lower or "fraud" in inc_lower or "sign" in original_query.lower():
            if not acts: acts = ["Indian Penal Code"]
            if not sections: sections = ["463", "465", "468", "471", "420"]
        elif "accident" in inc_lower or "vandi" in original_query.lower():
            if not acts: acts = ["Motor Vehicles Act", "Indian Penal Code"]
            if not sections: sections = ["134", "166", "279"]
        elif "cheque" in inc_lower or "bounce" in inc_lower:
            if not acts: acts = ["Negotiable Instruments Act"]
            if not sections: sections = ["138"]
        elif "property" in inc_lower or "evict" in inc_lower:
            if not acts: acts = ["Transfer of Property Act"]
            if not sections: sections = ["106", "111"]

    # Build a rich, retrieval-optimized search sentence
    search_parts = []
    if memory.incident: search_parts.append(memory.incident)
    if memory.vehicle: search_parts.append(f"involving {memory.vehicle.lower()}")
    if memory.injury: search_parts.append(f"{memory.injury.lower()} injury")
    for sec in sections: search_parts.append(f"Section {sec}")
    for act in acts: search_parts.append(act)
    if memory.state: search_parts.append(f"in {memory.state}")
    if memory.current_goal: search_parts.append(memory.current_goal.lower())
    search_parts.append(original_query)

    search_text = " ".join(search_parts) if search_parts else "legal dispute"

    # Build expanded concepts from the incident type
    concepts = _get_expanded_concepts(memory.incident, memory.legal_domain)

    # Build keywords from filled slots
    keywords = []
    if memory.incident: keywords.append(memory.incident.lower())
    if memory.vehicle: keywords.append(memory.vehicle.lower())
    if memory.injury: keywords.append(memory.injury.lower())
    if memory.fir_filed is not None: keywords.append("FIR" if memory.fir_filed else "no FIR")

    # Detect if user spoke in Tamil or Tanglish across conversation messages
    detected_lang = "en"
    all_user_text = " ".join([m["content"] for m in memory.messages if m["role"] == "user"])
    
    # Check for native Tamil script
    if re.search(r"[\u0B80-\u0BFF]", all_user_text):
        detected_lang = "ta"
    else:
        # Check for common Tanglish words/patterns
        tanglish_keywords = [
            "ena", "vandi", "idichu", "enaka", "iruku", "teriyum", "ungalukku",
            "puriyudha", "aachen", "sonnanga", "panren", "pannanum", "aachu",
            "poyi", "vanthu", "senthuten", "paathu", "mudiyum", "pannanga"
        ]
        text_words = set(re.findall(r"\b[a-zA-Z]+\b", all_user_text.lower()))
        if any(w in text_words for w in tanglish_keywords):
            detected_lang = "ta_roman"

    legal_query = LegalQuery(
        original_query=original_query,
        detected_language=detected_lang,
        legal_domain=memory.legal_domain or "General",
        incident_type=memory.incident or "Legal Dispute",
        keywords=keywords,
        suggested_acts=acts,
        suggested_sections=sections,
        key_entities=[memory.vehicle] if memory.vehicle else [],
        expanded_concepts=concepts,
    )

    print(f"[rewriter] Rewritten search: '{search_text}'")
    print(f"[rewriter] LegalQuery: domain={legal_query.legal_domain}, "
          f"incident={legal_query.incident_type}, "
          f"detected_language={detected_lang}, "
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
