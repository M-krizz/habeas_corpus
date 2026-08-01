"""
graph_builder.py — Habeas Corpus Legal Search Engine
======================================================
Module: graph/graph_builder.py

Responsibility:
    Accept the raw output of ``extract_legal_graph()`` and produce a
    validated, sanitised, loader-ready graph dict.

    This module knows about the *shape* of the graph — what constitutes a
    valid node, which relationships are permitted, how to clean strings —
    but it knows nothing about Neo4j or how to write Cypher.

Pipeline position:
    extractor.entity_extractor.extract_legal_graph()
        ↓
    graph.graph_builder.build_graph()          ← this file
        ↓
    graph.neo4j_loader.load_graph()

Phase 1 — pass-through with validation and light sanitisation.
Phase 2 TODO — resolve citation strings to actual Case nodes and add
               ("Case", "CITES", "Case") relationships.

Usage:
    from graph.graph_builder import build_graph

    legal_graph  = extract_legal_graph(raw_text)
    loader_ready = build_graph(legal_graph)
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Permitted relationship types — used to reject unexpected entries
# ---------------------------------------------------------------------------

_VALID_RELATIONSHIPS: set[tuple[str, str, str]] = {
    ("Case", "HEARD_IN",         "Court"),
    ("Case", "DECIDED_BY",       "Judge"),
    ("Case", "UNDER_ACT",        "Act"),
    ("Case", "INVOLVES_SECTION", "Section"),
    ("Case", "HAS_PETITIONER",   "Party"),
    ("Case", "HAS_RESPONDENT",   "Party"),
    ("Case", "INVOLVES_CONCEPT", "LegalConcept"),
}


# ---------------------------------------------------------------------------
# Low-level sanitisers
# ---------------------------------------------------------------------------

def _clean_str(value: object) -> str:
    """Return a whitespace-collapsed string, or '' if the value is not a str."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


# ---------------------------------------------------------------------------
# Per-node-type validators
# ---------------------------------------------------------------------------

def _validate_case(raw: dict) -> dict:
    return {
        "name":          _clean_str(raw.get("name")),
        "decision_date": _clean_str(raw.get("decision_date")),
        "language":      _clean_str(raw.get("language")) or "English",
    }


def _validate_court(raw: dict) -> dict:
    return {"name": _clean_str(raw.get("name"))}


def _validate_judge(raw: dict) -> dict | None:
    name = _clean_str(raw.get("name"))
    return {"name": name} if name else None


def _validate_act(raw: dict) -> dict | None:
    name = _clean_str(raw.get("name"))
    year = raw.get("year")
    if not name:
        return None
    if not isinstance(year, int) or not (1800 <= year <= 2100):
        return None
    return {"name": name, "year": year}


def _validate_section(raw: dict) -> dict | None:
    typ    = _clean_str(raw.get("type"))
    number = _clean_str(raw.get("number"))
    if not typ or not number:
        return None
    return {"type": typ, "number": number}


def _validate_party(raw: dict) -> dict | None:
    name = _clean_str(raw.get("name"))
    role = _clean_str(raw.get("role"))
    if not name or role not in ("Petitioner", "Respondent"):
        return None
    return {"name": name, "role": role}


def _validate_citation(raw: object) -> str | None:
    s = _clean_str(raw)
    return s if s else None


def _validate_concept(raw: dict) -> dict | None:
    cid    = _clean_str(raw.get("id"))
    name   = _clean_str(raw.get("name"))
    domain = _clean_str(raw.get("domain"))
    raw_aliases = raw.get("aliases", [])
    aliases = [_clean_str(a) for a in raw_aliases if _clean_str(a)]
    if not cid or not name:
        return None
    return {
        "id":      cid,
        "name":    name,
        "domain":  domain,
        "aliases": aliases,
    }


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_graph(legal_graph: dict) -> dict:
    """
    Validate and sanitise the output of ``extract_legal_graph()``.
    """
    raw_nodes = legal_graph.get("nodes", {})
    raw_rels   = legal_graph.get("relationships", [])

    # Validate and filter each node collection
    judges   = [v for j in raw_nodes.get("judges",   []) if (v := _validate_judge(j))]
    acts     = [v for a in raw_nodes.get("acts",     []) if (v := _validate_act(a))]
    sections = [v for s in raw_nodes.get("sections", []) if (v := _validate_section(s))]
    parties  = [v for p in raw_nodes.get("parties",  []) if (v := _validate_party(p))]
    citations = [v for c in raw_nodes.get("citations", []) if (v := _validate_citation(c))]
    concepts = [v for c in raw_nodes.get("concepts",  []) if (v := _validate_concept(c))]

    nodes = {
        "case":      _validate_case(raw_nodes.get("case", {})),
        "court":     _validate_court(raw_nodes.get("court", {})),
        "judges":    judges,
        "acts":      acts,
        "sections":  sections,
        "parties":   parties,
        "citations": citations,
        "concepts":  concepts,
    }

    # Keep only relationships that are in the permitted set
    valid_rels = [
        rel for rel in raw_rels
        if isinstance(rel, (tuple, list)) and tuple(rel) in _VALID_RELATIONSHIPS
    ]

    return {"nodes": nodes, "relationships": valid_rels}
