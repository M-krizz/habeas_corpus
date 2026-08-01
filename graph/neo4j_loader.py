"""
neo4j_loader.py — Habeas Corpus Legal Search Engine
=====================================================
Module: graph/neo4j_loader.py

Responsibility:
    Accept the output of ``graph_builder.build_graph()`` and persist every
    node and relationship into Neo4j using idempotent MERGE statements.

    This module has NO legal knowledge.  It simply loops over the nodes
    and relationships declared by the graph builder and drives MERGE for
    each.  Adding a new relationship type in the future requires only:
      1. A new Cypher template in ``_REL_TEMPLATES``
      2. A new branch in ``_merge_relationships``

Pipeline position:
    graph.graph_builder.build_graph()
        ↓
    graph.neo4j_loader.load_graph()    ← this file

Usage:
    from graph.neo4j_loader import load_graph

    load_graph(built_graph)                  # live write
    load_graph(built_graph, dry_run=True)    # log without writing (default)

Environment variables (via .env or shell):
    NEO4J_URI      — e.g.  neo4j+s://xxxx.databases.neo4j.io
    NEO4J_USER     — e.g.  neo4j
    NEO4J_PASSWORD — your AuraDB password
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


# ---------------------------------------------------------------------------
# Cypher templates — one MERGE per node label
# ---------------------------------------------------------------------------

_MERGE_CASE = """
MERGE (c:Case {name: $name})
SET   c.decision_date = $decision_date,
      c.language = COALESCE($language, 'English')
"""

_MERGE_COURT = """
MERGE (ct:Court {name: $name})
"""

_MERGE_JUDGE = """
MERGE (j:Judge {name: $name})
"""

_MERGE_ACT = """
MERGE (a:Act {name: $name})
SET   a.year = $year
"""

_MERGE_SECTION = """
MERGE (s:Section {type: $type, number: $number})
"""

_MERGE_PARTY = """
MERGE (p:Party {name: $name})
"""

_MERGE_LEGAL_CONCEPT = """
MERGE (lc:LegalConcept {id: $id})
SET   lc.name = $name,
      lc.domain = $domain,
      lc.aliases = $aliases
"""

# ---------------------------------------------------------------------------
# Cypher templates — one MERGE per relationship type
# ---------------------------------------------------------------------------

_REL_TEMPLATES: dict[str, str] = {
    "HEARD_IN": """
        MATCH (c:Case  {name: $case_name})
        MATCH (ct:Court {name: $node_name})
        MERGE (c)-[:HEARD_IN]->(ct)
    """,
    "DECIDED_BY": """
        MATCH (c:Case  {name: $case_name})
        MATCH (j:Judge {name: $node_name})
        MERGE (c)-[:DECIDED_BY]->(j)
    """,
    "UNDER_ACT": """
        MATCH (c:Case {name: $case_name})
        MATCH (a:Act  {name: $node_name})
        MERGE (c)-[:UNDER_ACT]->(a)
    """,
    "INVOLVES_SECTION": """
        MATCH (c:Case    {name: $case_name})
        MATCH (s:Section {type: $sec_type, number: $node_name})
        MERGE (c)-[:INVOLVES_SECTION]->(s)
    """,
    "HAS_PETITIONER": """
        MATCH (c:Case  {name: $case_name})
        MATCH (p:Party {name: $node_name})
        MERGE (c)-[:HAS_PETITIONER]->(p)
    """,
    "HAS_RESPONDENT": """
        MATCH (c:Case  {name: $case_name})
        MATCH (p:Party {name: $node_name})
        MERGE (c)-[:HAS_RESPONDENT]->(p)
    """,
    "INVOLVES_CONCEPT": """
        MATCH (c:Case  {name: $case_name})
        MATCH (lc:LegalConcept {id: $node_name})
        MERGE (c)-[:INVOLVES_CONCEPT]->(lc)
    """,
}


# ---------------------------------------------------------------------------
# Driver helper
# ---------------------------------------------------------------------------

def _get_driver():
    uri      = os.getenv("NEO4J_URI")
    user     = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri or not password:
        raise EnvironmentError(
            "NEO4J_URI and NEO4J_PASSWORD must be set in the environment "
            "or a .env file at the project root."
        )

    return GraphDatabase.driver(uri, auth=(user, password))


# ---------------------------------------------------------------------------
# Internal write helpers
# ---------------------------------------------------------------------------

def _run(tx, cypher: str, dry_run: bool, **params: Any) -> None:
    """Execute a single Cypher statement, or log it if dry_run is True."""
    if dry_run:
        short = " ".join(cypher.split())[:90]
        print(f"  [DRY] {short}  | params={params}")
    else:
        tx.run(cypher, **params)


def _merge_nodes(tx, nodes: dict, dry_run: bool) -> None:
    """MERGE all node types into Neo4j (or log if dry_run)."""
    case_data = dict(nodes["case"])
    if "language" not in case_data:
        case_data["language"] = "English"
    _run(tx, _MERGE_CASE, dry_run, **case_data)

    if nodes["court"].get("name"):
        _run(tx, _MERGE_COURT, dry_run, **nodes["court"])

    for judge in nodes["judges"]:
        _run(tx, _MERGE_JUDGE, dry_run, **judge)

    for act in nodes["acts"]:
        _run(tx, _MERGE_ACT, dry_run, **act)

    for sec in nodes["sections"]:
        _run(tx, _MERGE_SECTION, dry_run, **sec)

    for party in nodes["parties"]:
        _run(tx, _MERGE_PARTY, dry_run, name=party["name"])

    for concept in nodes.get("concepts", []):
        _run(tx, _MERGE_LEGAL_CONCEPT, dry_run, **concept)


def _merge_relationships(tx, nodes: dict, relationships: list, dry_run: bool) -> None:
    """
    Create relationships declared in the ``relationships`` list.
    """
    case_name = nodes["case"].get("name", "")

    for _subj, rel_type, _obj in relationships:
        cypher = _REL_TEMPLATES.get(rel_type)
        if not cypher:
            print(f"  [WARN] No Cypher template for relationship type: {rel_type}")
            continue

        if rel_type == "HEARD_IN":
            if nodes["court"].get("name"):
                _run(tx, cypher, dry_run,
                     case_name=case_name, node_name=nodes["court"]["name"])

        elif rel_type == "DECIDED_BY":
            for judge in nodes["judges"]:
                _run(tx, cypher, dry_run,
                     case_name=case_name, node_name=judge["name"])

        elif rel_type == "UNDER_ACT":
            for act in nodes["acts"]:
                _run(tx, cypher, dry_run,
                     case_name=case_name, node_name=act["name"])

        elif rel_type == "INVOLVES_SECTION":
            for sec in nodes["sections"]:
                _run(tx, cypher, dry_run,
                     case_name=case_name,
                     node_name=sec["number"],
                     sec_type=sec["type"])

        elif rel_type == "HAS_PETITIONER":
            for party in nodes["parties"]:
                if party["role"] == "Petitioner":
                    _run(tx, cypher, dry_run,
                         case_name=case_name, node_name=party["name"])

        elif rel_type == "HAS_RESPONDENT":
            for party in nodes["parties"]:
                if party["role"] == "Respondent":
                    _run(tx, cypher, dry_run,
                         case_name=case_name, node_name=party["name"])

        elif rel_type == "INVOLVES_CONCEPT":
            for concept in nodes.get("concepts", []):
                _run(tx, cypher, dry_run,
                     case_name=case_name, node_name=concept["id"])


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------

def load_graph(graph: dict, dry_run: bool = True) -> None:
    """
    Persist the graph produced by ``graph_builder.build_graph()`` into Neo4j.

    Parameters
    ----------
    graph : dict
        Output of ``build_graph()``.  Must have ``nodes`` and
        ``relationships`` keys.
    dry_run : bool
        If ``True`` (default), log what *would* be written without touching
        the database.  Set to ``False`` for a live write.
    """
    nodes         = graph.get("nodes", {})
    relationships = graph.get("relationships", [])
    case_name     = nodes.get("case", {}).get("name", "<unknown>")

    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"\n[neo4j_loader] {mode} — loading graph for: {case_name}")
    print("-" * 60)

    if dry_run:
        # No DB connection needed — log only
        _merge_nodes(None, nodes, dry_run=True)
        print()
        _merge_relationships(None, nodes, relationships, dry_run=True)
        print("-" * 60)
        print("[neo4j_loader] Dry run complete — no data written to Neo4j.")
        return

    driver = _get_driver()
    try:
        with driver.session() as session:
            session.execute_write(_merge_nodes,         nodes, dry_run)
            session.execute_write(_merge_relationships, nodes, relationships, dry_run)
        print(f"[neo4j_loader] Successfully loaded graph for: {case_name}")
    finally:
        driver.close()
