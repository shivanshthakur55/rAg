"""
FastAPI entrypoint for the Dual-Mode RAG Application.

PDF/Document RAG Mode (universal — all formats):
  GET  /              — serve the dual-mode chat UI
  POST /upload        — upload & ingest any supported document
  POST /chat          — Q&A against a specific document session
  POST /chat/all      — Q&A across ALL uploaded documents (no selection needed)
  GET  /sessions      — list all ingested document sessions
  POST /clear/{id}    — clear chat history for a session
  POST /clear-all     — clear the cross-document search history

Invoice RAG Mode:
  POST /invoice/upload    — upload & ingest an invoice PDF (OCR-capable)
  POST /invoice/query     — query ALL invoices (semantic + metadata); no selection needed
  POST /invoice/compare   — compare invoices; auto-finds best pair from query
  GET  /invoice/list      — list all stored invoices
  GET  /invoice/{id}      — fetch a single invoice by ID
  DELETE /invoice/{id}    — delete an invoice
"""

import shutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from rag_api import invoice_db
from rag_api.invoice_engine import (
    answer_invoice_query,
    compare_invoices,
    ingest_invoice,
    hybrid_invoice_search,
    _record_to_dict,
)
from rag_api.models import InvoiceCompareRequest, InvoiceQueryRequest
from rag_api.query_parser import parse_query
from rag_api.rag_engine import (
    chat,
    chat_all,
    clear_all_history,
    clear_session_history,
    ingest_document,
    list_sessions,
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Dual-Mode RAG API",
    version="2.1.0",
    description=(
        "Universal Document Q&A (PDF/Excel/Word/CSV/PPTX/Images) "
        "+ Invoice Analysis in one application."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Directory setup ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

UPLOAD_PDF_DIR = BASE_DIR / "uploads" / "pdf"
UPLOAD_PDF_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_INVOICE_DIR = BASE_DIR / "uploads" / "invoices"
UPLOAD_INVOICE_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Supported document extensions
ALLOWED_DOC_EXTENSIONS = {
    ".pdf",   # digital PDFs
    ".txt",   # plain text
    ".docx",  # Word
    ".doc",   # Word (legacy)
    ".xlsx",  # Excel
    ".xls",   # Excel (legacy)
    ".csv",   # Comma-separated values
    ".pptx",  # PowerPoint
    ".ppt",   # PowerPoint (legacy)
    ".jpg",   # JPEG image (OCR)
    ".jpeg",
    ".png",   # PNG image (OCR)
    ".tiff",  # TIFF image (OCR)
    ".bmp",   # BMP image (OCR)
    ".webp",  # WebP image (OCR)
}

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Initialize the SQLite database on server start."""
    invoice_db.init_db()


# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatAllRequest(BaseModel):
    question: str


# ═══════════════════════════════════════════════════════════════════════════════
# PDF / DOCUMENT MODE
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the dual-mode chat UI."""
    html_file = STATIC_DIR / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload any supported document and ingest it into the vector store.

    Supported: PDF (text + scanned/OCR), DOCX, TXT, XLSX, XLS, CSV, PPTX, PPT,
               JPG, PNG, TIFF, BMP, WEBP
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_DOC_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_DOC_EXTENSIONS))}"
            ),
        )

    dest = UPLOAD_PDF_DIR / file.filename
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
    """Ask a question about a specific ingested document session."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = chat(req.session_id, req.question)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")
    return result


@app.post("/chat/all")
async def chat_all_endpoint(req: ChatAllRequest):
    """
    Ask a question across ALL uploaded documents — no session selection needed.
    The system searches every document in the vector store and returns a unified answer.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = chat_all(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")
    return result


@app.get("/sessions")
async def sessions_endpoint():
    """List all ingested document sessions."""
    return {"sessions": list_sessions()}


@app.post("/clear/{session_id}")
async def clear_history(session_id: str):
    """Clear chat history for a specific document session."""
    clear_session_history(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/clear-all")
async def clear_all_endpoint():
    """Clear the cross-document search history."""
    clear_all_history()
    return {"status": "cleared", "mode": "all_documents"}


# ═══════════════════════════════════════════════════════════════════════════════
# INVOICE MODE
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/invoice/upload")
async def invoice_upload(file: UploadFile = File(...)):
    """
    Upload and ingest an invoice PDF.
    Supports both digitally-created PDFs and scanned/image-based PDFs (via OCR).
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for invoice ingestion.")

    dest = UPLOAD_INVOICE_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        record = ingest_invoice(str(dest), file.filename)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Invoice ingestion failed: {e}")

    return {
        "invoice_id": record.id,
        "vendor_name": record.vendor_name,
        "invoice_date": record.invoice_date,
        "total_amount": record.total_amount,
        "line_items_count": len(record.line_items),
        "status": "ready",
        "message": f"Invoice from '{record.vendor_name}' ingested successfully.",
    }


@app.post("/invoice/query")
async def invoice_query(req: InvoiceQueryRequest):
    """
    Query invoices using hybrid retrieval (metadata filters + semantic search).
    When no filters are provided, ALL stored invoices are searched automatically.

    Examples:
    - "Show me all invoices"
    - "Show invoices above $1000"
    - "Find invoices from ABC Corp in March 2024"
    - "Which vendor has the highest total?"
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        parsed = parse_query(req.query)

        # Override parsed filters with explicitly provided ones
        if req.vendor_name:
            parsed.vendor_hint = req.vendor_name
        if req.date_from:
            parsed.date_hint = req.date_from
        if req.amount_min is not None:
            parsed.amount_min = req.amount_min
        if req.amount_max is not None:
            parsed.amount_max = req.amount_max

        result = answer_invoice_query(req.query, parsed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return result


@app.post("/invoice/compare")
async def invoice_compare(req: InvoiceCompareRequest):
    """
    Compare two invoices with structured diff and LLM explanation.

    Modes:
      1. Provide invoice_id_a + invoice_id_b → direct comparison
      2. Provide invoice_id_a + query → auto-find the second invoice
      3. Provide only a query → auto-find the best two matching invoices
    """
    invoice_a = None
    invoice_b = None

    # Mode 3: pure query — find both invoices automatically
    if not req.invoice_id_a and req.query:
        parsed = parse_query(req.query)
        candidates = hybrid_invoice_search(req.query, parsed, k=10)

        # If no semantic results, fall back to all invoices
        if len(candidates) < 2:
            candidates = invoice_db.list_invoices()

        if len(candidates) < 2:
            raise HTTPException(
                status_code=404,
                detail="Need at least 2 invoices to compare. Please upload more invoices.",
            )
        invoice_a = candidates[0]
        invoice_b = candidates[1]

    else:
        # Modes 1 & 2: invoice_id_a is required
        if not req.invoice_id_a:
            raise HTTPException(
                status_code=400,
                detail="Provide either invoice_id_a or a query to identify the invoices.",
            )
        invoice_a = invoice_db.get_invoice(req.invoice_id_a)
        if not invoice_a:
            raise HTTPException(status_code=404, detail=f"Invoice '{req.invoice_id_a}' not found.")

        if req.invoice_id_b:
            invoice_b = invoice_db.get_invoice(req.invoice_id_b)
            if not invoice_b:
                raise HTTPException(status_code=404, detail=f"Invoice '{req.invoice_id_b}' not found.")
        elif req.query:
            parsed = parse_query(req.query)
            candidates = hybrid_invoice_search(req.query, parsed, k=5)
            candidates = [c for c in candidates if c.id != req.invoice_id_a]
            if not candidates:
                # Fallback: pick the second invoice from the full list
                all_inv = invoice_db.list_invoices()
                candidates = [i for i in all_inv if i.id != req.invoice_id_a]
            if not candidates:
                raise HTTPException(
                    status_code=404,
                    detail="Could not find a second invoice. Please upload more invoices.",
                )
            invoice_b = candidates[0]
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either invoice_id_b or a query to identify the second invoice.",
            )

    try:
        result = compare_invoices(
            invoice_a, invoice_b,
            user_query=req.query or f"Compare {invoice_a.vendor_name} with {invoice_b.vendor_name}."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

    return result.model_dump()


@app.get("/invoice/list")
async def invoice_list():
    """List all stored invoices with their metadata."""
    records = invoice_db.list_invoices()
    return {
        "total": len(records),
        "invoices": [_record_to_dict(r) for r in records],
    }


@app.get("/invoice/{invoice_id}")
async def invoice_get(invoice_id: str):
    """Fetch a single invoice by its ID."""
    record = invoice_db.get_invoice(invoice_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Invoice '{invoice_id}' not found.")
    return _record_to_dict(record)


@app.delete("/invoice/{invoice_id}")
async def invoice_delete(invoice_id: str):
    """Delete an invoice from structured storage."""
    deleted = invoice_db.delete_invoice(invoice_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Invoice '{invoice_id}' not found.")
    return {"status": "deleted", "invoice_id": invoice_id}

@app.post("/invoice/clear-all")
async def invoice_clear_all():
    """Clear all invoices from DB."""
    invoice_db.clear_all_invoices()
    return {"status": "cleared", "mode": "all_invoices"}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("rag_api.main:app", host="0.0.0.0", port=8001, reload=True)
