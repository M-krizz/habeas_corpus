"""
api/app.py — Habeas Corpus API Server
=====================================
FastAPI application — the HTTP interface for the Habeas Corpus AI engine.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title       = "Habeas Corpus — Adaptive Judicial Knowledge Engine",
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


_UPLOADS_DIR = Path(__file__).parent.parent / "uploads"

@app.post("/upload")
async def upload_document_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conversation_id: str | None = Form(None),
):
    """
    Case Document Upload & Precedent Matching Endpoint.

    Accepts uploaded PDFs, images, or TXT documents.
    1. Extracts clean text via OCR pipeline or PyMuPDF/text reader.
    2. Extracts legal concepts and statutory sections.
    3. Runs hybrid retrieval (FAISS + Neo4j) to find related precedent judgments.
    4. Generates structured legal analysis using the Reasoning Brain.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = _UPLOADS_DIR / file.filename

    try:
        content = await file.read()
        temp_path.write_bytes(content)

        extracted_text = ""
        document_meta = {
            "filename": file.filename,
            "size_bytes": len(content),
            "mode": "selectable",
            "page_count": 1,
            "avg_confidence": None
        }

        ext = temp_path.suffix.lower()
        if ext in (".txt", ".md", ".json"):
            extracted_text = content.decode("utf-8", errors="ignore")
        else:
            from ocr.pipeline import process_document
            doc_result = process_document(str(temp_path))
            extracted_text = doc_result.text
            document_meta["mode"] = doc_result.mode
            document_meta["page_count"] = len(doc_result.pages)
            document_meta["avg_confidence"] = doc_result.avg_ocr_confidence

        if not extracted_text.strip():
            raise HTTPException(status_code=422, detail="Could not extract readable text from uploaded document.")

        from query_understanding.legal_concept_mapper import map_legal_concepts
        from api.pipeline import run_query_with_legal_query

        sample_text = extracted_text[:3000]
        legal_query = map_legal_concepts(sample_text)

        response, new_hashes = await run_query_with_legal_query(legal_query)

        if new_hashes:
            from knowledge_acquisition.background_indexer import run_background_indexing
            background_tasks.add_task(run_background_indexing, new_hashes)

        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

        return {
            "document": document_meta,
            "extracted_text_preview": sample_text[:500] + "..." if len(sample_text) > 500 else sample_text,
            "legal_query": {
                "domain": legal_query.legal_domain,
                "incident_type": legal_query.incident_type,
                "acts": legal_query.suggested_acts,
                "sections": legal_query.suggested_sections,
                "search_text": legal_query.search_text,
            },
            "analysis": response.model_dump(),
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"[app] Upload document error: {exc}")
        import traceback
        traceback.print_exc()
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
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

    # LLM (OpenRouter / Gemini)
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if openrouter_key:
        status["llm"] = "configured (OpenRouter)"
    elif gemini_key and gemini_key != "YOUR_GEMINI_KEY_HERE":
        status["llm"] = "configured (Gemini AI Studio)"
    else:
        status["llm"] = "NOT CONFIGURED — add OPENROUTER_API_KEY to .env"

    # Indian Kanoon
    ik_key = os.getenv("INDIANKANOON_API_KEY", "")
    status["indian_kanoon"] = "configured" if ik_key else "not configured"

    return status


@app.get("/staging/status")
async def staging_status():
    """Show the current state of the staging pool."""
    from knowledge_acquisition.staging_pool import get_staging_summary
    return {"staging": get_staging_summary()}


_PROJECT_ROOT_DIR = Path(__file__).parent.parent
_SC_ENGLISH_DIR = _PROJECT_ROOT_DIR / "sc_English"
_SC_TAMIL_DIR   = _PROJECT_ROOT_DIR / "sc_tamil"
_OUTPUT_TEXT_DIR = _PROJECT_ROOT_DIR / "output"

def _find_case_file(case_id: str) -> tuple[Path, str] | None:
    """Locate the original judgment PDF or extracted text file for a given case_id."""
    clean_id = case_id.strip()
    if clean_id.endswith(".pdf") or clean_id.endswith(".txt"):
        clean_id = Path(clean_id).stem

    # Direct PDF check
    pdf_eng = _SC_ENGLISH_DIR / f"{clean_id}.pdf"
    if pdf_eng.exists():
        return (pdf_eng, "application/pdf")

    pdf_tam = _SC_TAMIL_DIR / f"{clean_id}.pdf"
    if pdf_tam.exists():
        return (pdf_tam, "application/pdf")

    # Direct TXT check
    txt_out = _OUTPUT_TEXT_DIR / f"{clean_id}.txt"
    if txt_out.exists():
        return (txt_out, "text/plain")

    # Partial/stem search
    for folder, ext, mime in [(_SC_ENGLISH_DIR, "*.pdf", "application/pdf"), (_SC_TAMIL_DIR, "*.pdf", "application/pdf"), (_OUTPUT_TEXT_DIR, "*.txt", "text/plain")]:
        if folder.exists():
            for f in folder.glob(ext):
                if clean_id.lower() in f.stem.lower() or f.stem.lower() in clean_id.lower():
                    return (f, mime)

    return None


@app.get("/download/case/{case_id}")
async def download_case_file(case_id: str):
    """Download the original judgment PDF or text file for a precedent case."""
    found = _find_case_file(case_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Case file for '{case_id}' not found on server.")

    filepath, mime_type = found
    return FileResponse(
        path=filepath,
        media_type=mime_type,
        filename=filepath.name,
        headers={"Content-Disposition": f'attachment; filename="{filepath.name}"'}
    )
