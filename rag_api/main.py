"""
FastAPI entrypoint for the RAG Chatbot.

Endpoints:
  GET  /              — serve the chat UI
  POST /upload        — upload a document file; returns {session_id, filename}
  POST /chat          — ask a question; returns {answer, sources}
  GET  /sessions      — list all ingested documents
  POST /clear/{id}    — clear chat history for a session
"""

import os
import shutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_api.rag_engine import chat, clear_session_history, ingest_document, list_sessions

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_file = STATIC_DIR / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a PDF, TXT, or DOCX file and ingest it into the vector store."""
    allowed = {".pdf", ".txt", ".docx", ".doc"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}",
        )

    # Save file temporarily
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        session_id = ingest_document(str(dest), file.filename)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return {"session_id": session_id, "filename": file.filename, "status": "ready"}


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """Ask a question about the ingested document."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = chat(req.session_id, req.question)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")
    return result


@app.get("/sessions")
async def sessions_endpoint():
    return {"sessions": list_sessions()}


@app.post("/clear/{session_id}")
async def clear_history(session_id: str):
    clear_session_history(session_id)
    return {"status": "cleared", "session_id": session_id}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
