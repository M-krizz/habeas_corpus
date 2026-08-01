"""
test_pipeline.py — Habeas Corpus Manual Test Suite
====================================================
Run individual layers or the full end-to-end pipeline from the terminal.

Usage:
    # Test each layer independently
    python test_pipeline.py --layer concept
    python test_pipeline.py --layer retrieval
    python test_pipeline.py --layer evidence
    python test_pipeline.py --layer reason
    python test_pipeline.py --layer acquisition   # Indian Kanoon fallback

    # Full end-to-end pipeline (no server needed)
    python test_pipeline.py --full

    # Test the live HTTP API (server must be running)
    python test_pipeline.py --api

    # Custom query
    python test_pipeline.py --full --query "My landlord threw me out without notice"

    # Start the server (separate terminal)
    python test_pipeline.py --serve
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

import json
import argparse
import asyncio

# ── Default test query ──────────────────────────────────────────────────────
DEFAULT_QUERY = "I got into a road accident. The other party says Section 134 protects him. What should I do?"

SEP  = "=" * 65
SEP2 = "-" * 65


def header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def ok(msg: str) -> None:  print(f"  [OK]  {msg}")
def err(msg: str) -> None: print(f"  [ERR] {msg}")
def info(msg: str) -> None: print(f"  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Legal Concept Mapper
# ─────────────────────────────────────────────────────────────────────────────

def test_concept(query: str) -> object:
    header("LAYER 1 — Legal Concept Mapper")
    from query_understanding.legal_concept_mapper import map_legal_concepts

    lq = map_legal_concepts(query)
    ok(f"Domain      : {lq.legal_domain}")
    ok(f"Incident    : {lq.incident_type}")
    ok(f"Acts        : {lq.suggested_acts}")
    ok(f"Sections    : {lq.suggested_sections}")
    ok(f"Keywords    : {lq.keywords}")
    ok(f"Concepts    : {lq.expanded_concepts}")
    info(f"\nSearch text : {lq.search_text[:100]}...")
    return lq


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Hybrid Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def test_retrieval(lq) -> list:
    header("LAYER 2 — Hybrid Retrieval (FAISS + Neo4j)")

    # FAISS semantic
    print("\n  [FAISS]")
    from semantic_retrieval.pipeline import search_cases
    faiss_hits = search_cases(lq.search_text, top_k=20, n=10)
    for h in faiss_hits[:3]:
        ok(f"  case={h['case_id']}  score={h['score']:.4f}  chunks={h['matched_chunks']}")

    # Neo4j graph
    print("\n  [Neo4j Graph]")
    from retrieval.graph_retriever import graph_search
    graph_hits = graph_search(lq, top_k=10)
    for h in graph_hits[:3]:
        ok(f"  case={h['case_id']}  graph_score={h['graph_score']:.4f}  acts={h['matched_acts']}")

    # Hybrid merge
    print("\n  [Hybrid Ranking]")
    from retrieval.hybrid_ranker import hybrid_rank
    ranked = hybrid_rank(faiss_hits, graph_hits, top_k=10)

    print(f"\n  {'Case ID':<40}  {'Final':>7}  {'Sem':>7}  {'Graph':>7}")
    print(f"  {SEP2}")
    for r in ranked[:5]:
        print(f"  {r['case_id']:<40}  {r['final_score']:>7.4f}  "
              f"{r['semantic_score']:>7.4f}  {r['graph_score']:>7.4f}")

    # Confidence
    from retrieval.confidence_estimator import estimate_confidence, CONFIDENCE_THRESHOLD
    conf = estimate_confidence(ranked)
    status = "HIGH (permanent KG)" if conf >= CONFIDENCE_THRESHOLD else \
             f"LOW  (will trigger Indian Kanoon fallback)"
    ok(f"\nConfidence: {conf:.3f}  →  {status}")

    # Feedback boost
    from feedback.reranker import apply_feedback_boost
    ranked = apply_feedback_boost(ranked)

    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Evidence Aggregator
# ─────────────────────────────────────────────────────────────────────────────

def test_evidence(ranked: list) -> list:
    header("LAYER 3 — Evidence Aggregator")
    from evidence.aggregator import aggregate_evidence

    evidence = aggregate_evidence(ranked, top_n=5)

    for i, ev in enumerate(evidence, 1):
        print(f"\n  Case #{i}")
        ok(f"  ID       : {ev.case_id}")
        ok(f"  Name     : {ev.case_name}")
        ok(f"  Court    : {ev.court}")
        ok(f"  Judges   : {ev.judges}")
        ok(f"  Acts     : {ev.acts}")
        ok(f"  Sections : {ev.sections}")
        ok(f"  Score    : final={ev.final_score:.4f}  sem={ev.semantic_score:.4f}  graph={ev.graph_score:.4f}")
        ok(f"  Chunks   : {len(ev.relevant_chunks)} text passage(s)")
        if ev.relevant_chunks:
            info(f"  Passage 1: {ev.relevant_chunks[0][:120]}...")

    return evidence


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — LLM Reasoning
# ─────────────────────────────────────────────────────────────────────────────

def test_reasoning(lq, evidence: list, confidence: float = 0.8) -> object:
    header("LAYER 4 — LLM Legal Reasoner")
    from reasoning.legal_reasoner import reason

    response = reason(lq, evidence, confidence)

    print(f"\n  SUMMARY\n  {SEP2}")
    print(f"  {response.summary}\n")

    print(f"  APPLICABLE ACTS     : {response.applicable_acts}")
    print(f"  APPLICABLE SECTIONS : {response.applicable_sections}")
    print(f"  KNOWLEDGE SOURCE    : {response.knowledge_source}")
    print(f"  CONFIDENCE          : {response.confidence:.3f}")

    print(f"\n  PRECEDENTS ({len(response.precedents)} case(s))")
    for i, p in enumerate(response.precedents, 1):
        print(f"  [{i}] {p.get('case','?')} — {p.get('court','?')}")
        print(f"       Held: {str(p.get('held',''))[:120]}...")

    if response.legal_query and hasattr(response.legal_query, 'original_query'):
        pass  # Already printed

    print(f"\n  DISCLAIMER: {response.disclaimer[:80]}...")
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Layer 5 — Adaptive Acquisition (Indian Kanoon fallback)
# ─────────────────────────────────────────────────────────────────────────────

def test_acquisition(lq) -> None:
    header("LAYER 5 — Adaptive Knowledge Acquisition (Indian Kanoon)")

    from knowledge_acquisition.web_retriever import search_indian_kanoon
    results = search_indian_kanoon(lq, max_results=3)

    if not results:
        err("No results returned. Check INDIANKANOON_API_KEY in .env")
        return

    for i, r in enumerate(results, 1):
        print(f"\n  Result #{i}")
        ok(f"  Title   : {r['title'][:70]}")
        ok(f"  Court   : {r.get('court','?')}")
        ok(f"  Date    : {r.get('date','?')}")
        ok(f"  URL     : {r['url']}")
        ok(f"  Excerpt : {r['text_excerpt'][:120]}...")

    # Stage them
    from knowledge_acquisition.staging_pool import stage_case, get_staging_summary
    for r in results:
        h = stage_case(r)
        ok(f"  Staged  : hash={h}")

    print("\n  Staging pool summary:")
    for s in get_staging_summary()[:5]:
        info(f"  [{s['status']:10}] {s['title'][:55]}  (retrieved {s['retrieval_count']}x)")


# ─────────────────────────────────────────────────────────────────────────────
# Full end-to-end pipeline (no server)
# ─────────────────────────────────────────────────────────────────────────────

async def test_full(query: str) -> None:
    header(f"FULL PIPELINE — End-to-End")
    info(f"Query: \"{query}\"\n")

    from api.pipeline import run_query
    response, new_hashes = await run_query(query)

    print(f"\n{'='*65}")
    print("  FINAL RESPONSE")
    print(f"{'='*65}")
    print(f"\n  Summary:\n  {response.summary}")
    print(f"\n  Acts     : {response.applicable_acts}")
    print(f"  Sections : {response.applicable_sections}")
    print(f"  Source   : {response.knowledge_source}")
    print(f"  Conf     : {response.confidence:.3f}")
    print(f"\n  Precedents ({len(response.precedents)}):")
    for i, p in enumerate(response.precedents, 1):
        print(f"  [{i}] {p.get('case','?')}")
        print(f"       {str(p.get('held',''))[:100]}...")

    if new_hashes:
        info(f"\n  New staging hashes: {new_hashes}")
        info("  (Background indexer would promote these after validation)")

    print(f"\n{'='*65}")


# ─────────────────────────────────────────────────────────────────────────────
# API test (live HTTP calls — server must be running)
# ─────────────────────────────────────────────────────────────────────────────

def test_api(query: str, base_url: str = "http://127.0.0.1:8000") -> None:
    header(f"API TEST — {base_url}")

    try:
        import httpx
    except ImportError:
        err("httpx not installed. Run: pip install httpx")
        return

    client = httpx.Client(timeout=120)

    # Health check
    print("\n  GET /health")
    try:
        r = client.get(f"{base_url}/health")
        data = r.json()
        for k, v in data.items():
            status = "OK" if "ok" in str(v).lower() or "configured" in str(v).lower() else "!!"
            print(f"  [{status}] {k:15}: {v}")
    except Exception as e:
        err(f"Health check failed: {e}")
        return

    # Query
    print(f"\n  POST /query")
    info(f"  Query: \"{query[:60]}...\"")
    try:
        r = client.post(f"{base_url}/query", json={"query": query})
        if r.status_code != 200:
            err(f"HTTP {r.status_code}: {r.text[:200]}")
            return
        data = r.json()
        ok(f"Summary  : {data.get('summary','')[:100]}...")
        ok(f"Acts     : {data.get('applicable_acts', [])}")
        ok(f"Sections : {data.get('applicable_sections', [])}")
        ok(f"Source   : {data.get('knowledge_source','?')}")
        ok(f"Conf     : {data.get('confidence', 0):.3f}")
        ok(f"Cases    : {len(data.get('precedents',[]))} precedent(s)")
    except Exception as e:
        err(f"Query failed: {e}")
        return

    # Feedback
    print(f"\n  POST /feedback")
    first_case = data.get('precedents', [{}])[0].get('case', 'test_case')
    try:
        r = client.post(f"{base_url}/feedback", json={
            "query":   query,
            "case_id": first_case,
            "rating":  1,
        })
        ok(f"Feedback recorded: {r.json()}")
    except Exception as e:
        err(f"Feedback failed: {e}")

    # Staging status
    print(f"\n  GET /staging/status")
    try:
        r = client.get(f"{base_url}/staging/status")
        staging = r.json().get("staging", [])
        ok(f"{len(staging)} case(s) in staging pool")
        for s in staging[:3]:
            info(f"  [{s['status']:10}] {s['title'][:55]}")
    except Exception as e:
        err(f"Staging status failed: {e}")

    client.close()
    print(f"\n  Open in browser: {base_url}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Habeas Corpus — Manual Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--query",       default=DEFAULT_QUERY, help="Legal query to test")
    parser.add_argument("--layer",       choices=["concept","retrieval","evidence","reason","acquisition"],
                        help="Test a single pipeline layer")
    parser.add_argument("--full",        action="store_true", help="Full end-to-end pipeline test")
    parser.add_argument("--api",         action="store_true", help="Test live HTTP API")
    parser.add_argument("--serve",       action="store_true", help="Start the web server")
    parser.add_argument("--url",         default="http://127.0.0.1:8000", help="API base URL for --api")
    args = parser.parse_args()

    query = args.query
    print(f"\nHabeas Corpus — Test Suite")
    print(f"Query: \"{query[:70]}\"")

    if args.serve:
        import uvicorn
        print("\nStarting server at http://127.0.0.1:8000 ...")
        uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)
        return

    if args.api:
        test_api(query, args.url)
        return

    if args.full:
        asyncio.run(test_full(query))
        return

    if args.layer == "concept":
        test_concept(query)

    elif args.layer == "retrieval":
        lq = test_concept(query)
        test_retrieval(lq)

    elif args.layer == "evidence":
        lq = test_concept(query)
        ranked = test_retrieval(lq)
        test_evidence(ranked)

    elif args.layer == "reason":
        lq = test_concept(query)
        ranked = test_retrieval(lq)
        evidence = test_evidence(ranked)
        from retrieval.confidence_estimator import estimate_confidence
        conf = estimate_confidence(ranked)
        test_reasoning(lq, evidence, conf)

    elif args.layer == "acquisition":
        lq = test_concept(query)
        test_acquisition(lq)

    else:
        # Default: run all layers in sequence
        lq      = test_concept(query)
        ranked  = test_retrieval(lq)
        evidence = test_evidence(ranked)
        from retrieval.confidence_estimator import estimate_confidence
        conf    = estimate_confidence(ranked)
        test_reasoning(lq, evidence, conf)

    print(f"\n{SEP}")
    print("  Done.")
    print(SEP)


if __name__ == "__main__":
    main()
