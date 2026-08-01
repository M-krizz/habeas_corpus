"""
api/app.py — Nyaya-Setu API Server
=====================================
FastAPI application — the HTTP interface for the Nyaya-Setu AI engine.

Endpoints:
  POST /chat              — multi-turn conversation (Three-Brain Architecture)
  POST /query             — legacy single-shot query
  GET  /conversation/{id} — retrieve conversation state
  POST /feedback          — record user feedback
  GET  /health            — system health check
  GET  /staging/status    — view staging pool contents
  GET  /                  — serve the UI (index.html)

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
    title       = "Nyaya-Setu — Adaptive Judicial Knowledge Engine",
    description = "AI-powered Indian legal research with Three-Brain Architecture: "
                  "Conversation Brain + Knowledge Brain + Reasoning Brain",
    version     = "3.0.0",
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

class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str

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


@app.post("/chat")
async def chat_endpoint(req: ChatRequest, background_tasks: BackgroundTasks):
    """
    Multi-turn conversation endpoint (Three-Brain Architecture).

    The Conversation Brain processes the message:
    - If more information is needed → returns a follow-up question
    - If enough slots are filled → triggers Knowledge + Reasoning Brains
      and returns a full legal analysis

    Parameters
    ----------
    req.conversation_id : session ID (null to start new conversation)
    req.message         : the user's message text
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        from conversation.conversation import process_message
        result = await process_message(req.conversation_id, req.message.strip())

        # Schedule background indexing if new cases were staged
        if result.full_response and result.full_response.get("_staging_hashes"):
            from knowledge_acquisition.background_indexer import run_background_indexing
            hashes = result.full_response.pop("_staging_hashes", [])
            background_tasks.add_task(run_background_indexing, hashes)

        return {
            "type":             result.type,
            "conversation_id":  result.conversation_id,
            "message":          result.message,
            "memory":           result.memory_snapshot,
            "slot_status":      result.slot_status,
            "full_response":    result.full_response,
        }

    except Exception as exc:
        print(f"[app] Chat error: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Retrieve the current state of a conversation."""
    from conversation.conversation import get_session
    session = get_session(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {
        "conversation_id": session.conversation_id,
        "memory":          session.filled_slots(),
        "messages":        session.messages,
        "turn_count":      session.turn_count,
        "ready":           session.ready_for_retrieval,
    }


@app.post("/query")
async def query_endpoint(req: QueryRequest, background_tasks: BackgroundTasks):
    """
    Legacy single-shot query endpoint.

    Runs the full pipeline without multi-turn conversation.
    Use POST /chat for the Three-Brain Architecture.
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
    """Record user feedback (thumbs up = 1, thumbs down = -1)."""
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
