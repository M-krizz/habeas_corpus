"""
api/app.py — Habeas Corpus
============================
FastAPI application — the HTTP interface for the Habeas Corpus AI engine.

Endpoints:
  POST /query           — main query endpoint
  POST /feedback        — record user feedback (👍/👎)
  GET  /health          — system health check
  GET  /staging/status  — view staging pool contents
  GET  /               — serve the UI (index.html)

Run:
    uv run uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title       = "Habeas Corpus — Adaptive Judicial Knowledge Engine",
    description = "AI-powered Indian legal research with Knowledge Graph + Semantic Search",
    version     = "2.0.0",
    docs_url    = "/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

_UI_PATH = Path(__file__).parent.parent / "ui" / "index.html"


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str

class FeedbackRequest(BaseModel):
    query:      str
    case_id:    str
    rating:     int    # 1 or -1
    session_id: str = "anonymous"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve the single-page UI."""
    if _UI_PATH.exists():
        return HTMLResponse(content=_UI_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h2>UI not found. Run the full build to generate ui/index.html.</h2>",
        status_code=404,
    )


@app.post("/query")
async def query_endpoint(req: QueryRequest, background_tasks: BackgroundTasks):
    """
    Main query endpoint.

    Runs the full pipeline and returns a structured JSON answer.
    Any new web-retrieved cases are staged and indexed in the background
    AFTER the response is returned — so the user never waits.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        from api.pipeline import run_query
        response, new_hashes = await run_query(req.query.strip())

        # Schedule background indexing (non-blocking)
        if new_hashes:
            from knowledge_acquisition.background_indexer import run_background_indexing
            background_tasks.add_task(run_background_indexing, new_hashes)

        return response.model_dump()

    except Exception as exc:
        print(f"[app] Pipeline error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    """Record user feedback (👍 = 1, 👎 = -1)."""
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="Rating must be 1 or -1.")

    from feedback.store import record_feedback
    from feedback.schema import FeedbackRecord

    fb = FeedbackRecord(
        query      = req.query,
        case_id    = req.case_id,
        rating     = req.rating,
        session_id = req.session_id,
    )
    record_feedback(fb)
    return {"status": "recorded", "case_id": req.case_id, "rating": req.rating}


@app.get("/health")
async def health():
    """Check connectivity to Neo4j, FAISS, and the LLM."""
    status = {}

    # Neo4j
    try:
        from retrieval.graph_retriever import _run_query
        _run_query("RETURN 1 AS ok", {})
        status["neo4j"] = "ok"
    except Exception as exc:
        status["neo4j"] = f"error: {exc}"

    # FAISS
    try:
        from pathlib import Path
        idx = Path(__file__).parent.parent / "semantic_index" / "legal.index"
        status["faiss"] = "ok" if idx.exists() else "index not built"
    except Exception as exc:
        status["faiss"] = f"error: {exc}"

    # Gemini
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    status["gemini"] = (
        "configured" if gemini_key and gemini_key != "YOUR_GEMINI_KEY_HERE"
        else "NOT CONFIGURED — add GEMINI_API_KEY to .env"
    )

    # Indian Kanoon
    ik_key = os.getenv("INDIANKANOON_API_KEY", "")
    status["indian_kanoon"] = "configured" if ik_key else "not configured"

    return status


@app.get("/staging/status")
async def staging_status():
    """Show the current state of the staging pool."""
    from knowledge_acquisition.staging_pool import get_staging_summary
    return {"staging": get_staging_summary()}
