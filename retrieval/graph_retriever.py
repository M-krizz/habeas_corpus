"""
retrieval/graph_retriever.py — Habeas Corpus
=============================================
Queries the Neo4j Knowledge Graph using structured legal concepts
extracted by the Legal Concept Mapper.

Returns a list of case matches with graph-based relevance scores,
which are then fused with FAISS semantic scores by hybrid_ranker.py.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

from query_understanding.schema import LegalQuery

load_dotenv()


# ---------------------------------------------------------------------------
# Neo4j driver — singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_driver():
    from neo4j import GraphDatabase
    uri      = os.getenv("NEO4J_URI")
    user     = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not password:
        raise EnvironmentError("[graph_retriever] NEO4J_URI / NEO4J_PASSWORD not set.")
    return GraphDatabase.driver(uri, auth=(user, password))


_NEO4J_ONLINE = True


def _run_query(cypher: str, params: dict) -> list[dict]:
    global _NEO4J_ONLINE
    if not _NEO4J_ONLINE:
        raise ConnectionError("Neo4j circuit breaker open (database paused or offline).")
    try:
        driver = _get_driver()
        with driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]
    except Exception as exc:
        msg = str(exc).lower()
        if any(w in msg for w in ["routing", "connection", "timed out", "unavailable", "unreachable"]):
            _NEO4J_ONLINE = False
            print("[graph_retriever] WARNING — Neo4j appears paused/offline. Circuit breaker opened to prevent slow timeouts.")
        raise exc


# ---------------------------------------------------------------------------
# Individual graph queries
# ---------------------------------------------------------------------------

_QUERY_BY_ACT = """
MATCH (c:Case)-[:UNDER_ACT]->(a:Act)
WHERE any(act IN $act_names WHERE toLower(a.name) CONTAINS toLower(act))
WITH c, count(a) AS act_hits
RETURN c.name AS case_id, act_hits AS hits, 'act' AS match_type
"""

_QUERY_BY_SECTION = """
MATCH (c:Case)-[:INVOLVES_SECTION]->(s:Section)
WHERE s.number IN $section_numbers
WITH c, count(s) AS sec_hits
RETURN c.name AS case_id, sec_hits AS hits, 'section' AS match_type
"""

_QUERY_BY_KEYWORD = """
MATCH (c:Case)
WHERE any(kw IN $keywords WHERE toLower(c.name) CONTAINS toLower(kw))
RETURN c.name AS case_id, 1 AS hits, 'keyword' AS match_type
"""

_QUERY_BY_CONCEPT = """
MATCH (c:Case)-[:INVOLVES_CONCEPT]->(lc:LegalConcept)
WHERE any(alias IN lc.aliases WHERE any(kw IN $keywords WHERE toLower(alias) CONTAINS toLower(kw)))
   OR toLower(lc.name) CONTAINS toLower($incident_type)
WITH c, count(lc) AS concept_hits
RETURN c.name AS case_id, concept_hits AS hits, 'concept' AS match_type
"""

_QUERY_CASE_METADATA = """
MATCH (c:Case {name: $case_name})
OPTIONAL MATCH (c)-[:HEARD_IN]->(ct:Court)
OPTIONAL MATCH (c)-[:DECIDED_BY]->(j:Judge)
OPTIONAL MATCH (c)-[:UNDER_ACT]->(a:Act)
OPTIONAL MATCH (c)-[:INVOLVES_SECTION]->(s:Section)
OPTIONAL MATCH (c)-[:HAS_PETITIONER]->(pet:Party)
OPTIONAL MATCH (c)-[:HAS_RESPONDENT]->(resp:Party)
RETURN
  c.name            AS case_name,
  c.decision_date   AS decision_date,
  collect(DISTINCT ct.name)   AS courts,
  collect(DISTINCT j.name)    AS judges,
  collect(DISTINCT a.name)    AS acts,
  collect(DISTINCT s.number)  AS sections,
  collect(DISTINCT pet.name)  AS petitioners,
  collect(DISTINCT resp.name) AS respondents
"""


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------

def _merge_hits(raw_hits: list[dict]) -> dict[str, dict]:
    """
    Merge results from multiple query types into a per-case score dict.

    Scoring weights:
        section match  → 3 points each   (most specific)
        concept match  → 3 points each   (canonical legal concept)
        act match      → 2 points each
        keyword match  → 1 point each
    """
    weights = {"section": 3, "concept": 3, "act": 2, "keyword": 1}
    cases: dict[str, dict] = {}

    for row in raw_hits:
        cid   = row["case_id"]
        wt    = weights.get(row["match_type"], 1)
        score = wt * row["hits"]

        if cid not in cases:
            cases[cid] = {"case_id": cid, "raw_score": 0,
                          "matched_acts": 0, "matched_sections": 0,
                          "matched_keywords": 0}

        cases[cid]["raw_score"] += score
        if row["match_type"] == "act":
            cases[cid]["matched_acts"] += row["hits"]
        elif row["match_type"] == "section":
            cases[cid]["matched_sections"] += row["hits"]
        else:
            cases[cid]["matched_keywords"] += row["hits"]

    return cases


def _normalise(cases: dict[str, dict]) -> list[dict]:
    """
    Normalise raw scores to [0, 1] and return sorted list.
    """
    if not cases:
        return []
    max_score = max(c["raw_score"] for c in cases.values()) or 1
    result = []
    for c in cases.values():
        c["graph_score"] = round(c["raw_score"] / max_score, 6)
        result.append(c)
    return sorted(result, key=lambda x: x["graph_score"], reverse=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def graph_search(legal_query: LegalQuery, top_k: int = 20) -> list[dict]:
    """
    Search Neo4j for cases matching the structured legal concepts.

    Parameters
    ----------
    legal_query : LegalQuery from the concept mapper
    top_k       : max cases to return

    Returns
    -------
    list of dict, each:
        {case_id, graph_score, matched_acts, matched_sections, matched_keywords}
    Sorted descending by graph_score.
    """
    all_hits: list[dict] = []

    # Query by canonical LegalConcepts
    search_keywords = legal_query.keywords + [legal_query.original_query]
    try:
        rows = _run_query(_QUERY_BY_CONCEPT, {
            "keywords": search_keywords,
            "incident_type": legal_query.incident_type
        })
        all_hits.extend(rows)
    except Exception as exc:
        print(f"[graph_retriever] Concept query failed: {exc}")

    # Query by suggested Acts
    if legal_query.suggested_acts:
        try:
            rows = _run_query(_QUERY_BY_ACT,
                              {"act_names": legal_query.suggested_acts})
            all_hits.extend(rows)
        except Exception as exc:
            print(f"[graph_retriever] Act query failed: {exc}")

    # Query by Sections
    if legal_query.suggested_sections:
        try:
            rows = _run_query(_QUERY_BY_SECTION,
                              {"section_numbers": legal_query.suggested_sections})
            all_hits.extend(rows)
        except Exception as exc:
            print(f"[graph_retriever] Section query failed: {exc}")

    # Query by keywords (fallback broadening)
    if legal_query.keywords:
        try:
            rows = _run_query(_QUERY_BY_KEYWORD,
                              {"keywords": legal_query.keywords[:10]})
            all_hits.extend(rows)
        except Exception as exc:
            print(f"[graph_retriever] Keyword query failed: {exc}")

    merged   = _merge_hits(all_hits)
    ranked   = _normalise(merged)
    result   = ranked[:top_k]

    print(f"[graph_retriever] Found {len(result)} cases via graph search.")
    return result


def fetch_case_metadata(case_id: str) -> dict:
    """
    Fetch full metadata for a single case from Neo4j.

    Used by the Evidence Aggregator to enrich chunk hits.

    Parameters
    ----------
    case_id : case name / identifier string (matches Case.name in Neo4j)

    Returns
    -------
    dict with keys: case_name, decision_date, courts, judges, acts,
                    sections, petitioners, respondents
    """
    try:
        rows = _run_query(_QUERY_CASE_METADATA, {"case_name": case_id})
        if rows:
            return rows[0]
    except Exception as exc:
        print(f"[graph_retriever] Metadata fetch failed for '{case_id}': {exc}")

    # Return empty scaffold so Evidence Aggregator never crashes
    return {
        "case_name": case_id,
        "decision_date": "",
        "courts": [],
        "judges": [],
        "acts": [],
        "sections": [],
        "petitioners": [],
        "respondents": [],
    }
