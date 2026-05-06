"""
Invoice Engine — Full pipeline for invoice ingestion, retrieval, and comparison.

Pipeline:
  1. PDF text extraction (PyPDF with OCR fallback via Tesseract for image-only PDFs)
  2. Structured field extraction via Groq LLM
  3. Storage: SQLite (structured) + Chroma (semantic embeddings)
  4. Hybrid retrieval: metadata filters (SQLite) + semantic search (Chroma)
     → When no filters are active, ALL stored invoices are included automatically
  5. Invoice comparison: structured diff + LLM explanation
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytesseract
from dotenv import load_dotenv

# Force Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_api import invoice_db
from rag_api.models import ComparisonResult, InvoiceRecord, LineItem
from rag_api.query_parser import ParsedQuery

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
INVOICE_VECTOR_DIR = Path(__file__).parent.parent / "Vector" / "invoices"
INVOICE_VECTOR_DIR.mkdir(parents=True, exist_ok=True)

INVOICE_COLLECTION = "invoices_collection"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ── Singletons ────────────────────────────────────────────────────────────────
_embedding_model: Optional[HuggingFaceEmbeddings] = None
_llm: Optional[ChatGroq] = None
_vector_store: Optional[Chroma] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    return _embedding_model


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,   # deterministic for field extraction
            max_tokens=2048,
        )
    return _llm


def _get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=INVOICE_COLLECTION,
            embedding_function=get_embedding_model(),
            persist_directory=str(INVOICE_VECTOR_DIR),
        )
    return _vector_store


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract raw text from an invoice PDF.
    Strategy:
      1. Try PyPDF (fast, works for digitally-created PDFs).
      2. If extracted text is too short (<50 chars), fall back to Tesseract OCR
         via pdf2image — handles scanned/image-only PDFs.
    """
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text = "\n".join(d.page_content for d in docs).strip()
        if len(text) >= 50:
            return text
        # Fall through to OCR below
    except Exception:
        pass

    # OCR fallback using PyMuPDF (no Poppler needed)
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image

        # Force Tesseract path for Windows
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        doc = fitz.open(file_path)
        pages = []
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                mat = fitz.Matrix(2.0, 2.0)  # 2x scale for better OCR accuracy
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img, lang="eng").strip()
                if page_text:
                    pages.append(page_text)
        finally:
            doc.close()
        
        text = "\n".join(pages).strip()
        if text:
            return text
        raise RuntimeError("OCR produced no text from this PDF.")
    except ImportError as e:
        raise RuntimeError(
            f"PyMuPDF or pytesseract not available: {e}. "
            "Run: uv add pymupdf pytesseract"
        ) from None
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}") from e


# ── LLM-based field extraction ────────────────────────────────────────────────

_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an information extraction system. Extract structured data from invoice text.

Return ONLY a valid JSON object. No explanation, no extra text.

Schema:
{{
  "invoice_id": string | null,
  "vendor_name": string | null,
  "issue_date": string (YYYY-MM-DD) | null,
  "due_date": string (YYYY-MM-DD) | null,
  "total_amount": number | null,
  "line_items": [
    {{
      "description": string | null,
      "quantity": number | null,
      "unit_price": number | null,
      "total_price": number | null
    }}
  ]
}}

Rules:

- invoice_id:
  Extract from "Invoice Number", "Invoice No", "Reference Number", or "Bill No".
  Prefer explicitly labeled invoice identifiers.

- vendor_name:
  The company issuing the invoice (seller), NOT the buyer.
  Ignore "Bill To", "Ship To", or customer details.

- issue_date:
  Extract "Invoice Date" or "Issue Date".
  Convert to YYYY-MM-DD format.
  If multiple dates exist, choose the earliest valid one.

- due_date:
  Extract "Due Date" or "Payment Due".
  Convert to YYYY-MM-DD format.
  Must be after issue_date if both exist.

- total_amount:
  Extract the final payable amount.
  Prefer "Total", "Grand Total", or "Amount Due".
  Return numeric value only (no currency symbols).
  If multiple values exist, choose the largest final payable amount.

- line_items:
  Extract ALL line items if present.
  Each item should include description, quantity, unit_price, total_price if available.
  If no items found, return an empty list [].

General Rules:
- Use null for missing fields (DO NOT use empty strings)
- Do NOT hallucinate values
- Ensure valid JSON output only
""",
    ),
    ("human", "Invoice text:\n\n{text}"),
])


def extract_invoice_fields(raw_text: str) -> dict:
    """
    Use Groq LLM to extract structured fields from raw invoice text.
    Returns a dict with keys: invoice_id, vendor_name, invoice_date, total_amount, line_items
    """
    truncated = raw_text[:6000]  # stay within token limits

    chain = _EXTRACTION_PROMPT | get_llm() | StrOutputParser()
    result = chain.invoke({"text": truncated})

    # Strip markdown fences if the model wrapped it
    result = result.strip()
    if result.startswith("```"):
        result = re.sub(r"^```(?:json)?\n?", "", result)
        result = re.sub(r"\n?```$", "", result)

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        # Attempt to salvage partial JSON
        match = re.search(r"\{.*\}", result, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_invoice(file_path: str, filename: str) -> InvoiceRecord:
    """
    Full invoice ingestion pipeline:
      extract text → extract fields → build InvoiceRecord →
      save to SQLite → embed in Chroma
    Returns the stored InvoiceRecord.
    """
    # 1. Extract text
    raw_text = extract_text_from_pdf(file_path)
    if not raw_text:
        raise ValueError("No text could be extracted from this PDF.")

    # 2. Extract structured fields
    fields = extract_invoice_fields(raw_text)

    # 3. Build line items
    line_items = []
    for item in fields.get("line_items") or []:
        if isinstance(item, dict):
            line_items.append(LineItem(
                description=str(item.get("description") or ""),
                quantity=_to_float(item.get("quantity")),
                unit_price=_to_float(item.get("unit_price")),
                total=_to_float(item.get("total")),
            ))

    # 4. Build InvoiceRecord
    invoice_id = str(fields.get("invoice_id") or uuid.uuid4())
    record = InvoiceRecord(
        id=invoice_id,
        vendor_name=str(fields.get("vendor_name") or ""),
        invoice_date=str(fields.get("invoice_date") or ""),
        total_amount=_to_float(fields.get("total_amount")) or 0.0,
        line_items=line_items,
        raw_text=raw_text,
        file_path=file_path,
        original_filename=filename,
        uploaded_at=datetime.utcnow().isoformat(),
    )

    # 5. Save to SQLite
    invoice_db.save_invoice(record)

    # 6. Embed in Chroma
    _embed_invoice(record)

    return record


def _embed_invoice(record: InvoiceRecord) -> None:
    """Chunk invoice text and upsert into the Chroma invoices collection."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(record.raw_text)

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(Document(
            page_content=chunk,
            metadata={
                "invoice_id": record.id,
                "vendor_name": record.vendor_name,
                "invoice_date": record.invoice_date,
                "total_amount": record.total_amount,
                "original_filename": record.original_filename,
                "chunk_index": i,
            },
        ))

    if docs:
        vs = _get_vector_store()
        vs.add_documents(docs)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def query_invoices_semantic(query: str, k: int = 5) -> list[InvoiceRecord]:
    """Pure semantic search over Chroma invoice collection."""
    vs = _get_vector_store()
    results = vs.similarity_search(query, k=k * 3)  # over-fetch, dedupe

    # Collect unique invoice IDs preserving order
    seen: set[str] = set()
    records: list[InvoiceRecord] = []
    for doc in results:
        iid = doc.metadata.get("invoice_id")
        if iid and iid not in seen:
            seen.add(iid)
            record = invoice_db.get_invoice(iid)
            if record:
                records.append(record)
        if len(records) >= k:
            break
    return records


def query_invoices_structured(parsed: ParsedQuery, limit: int = 10) -> list[InvoiceRecord]:
    """Metadata-filtered query via SQLite."""
    date_from = None
    date_to = None
    if parsed.date_hint:
        # If only year-month given, build a range covering that month
        if re.match(r"^\d{4}-\d{2}$", parsed.date_hint):
            date_from = parsed.date_hint + "-01"
            year, month = parsed.date_hint.split("-")
            # last day approximation: just set to -31 and SQLite string compare handles it
            date_to = parsed.date_hint + "-31"
        else:
            date_from = date_to = parsed.date_hint

    return invoice_db.query_invoices(
        vendor_name=parsed.vendor_hint,
        date_from=date_from,
        date_to=date_to,
        amount_min=parsed.amount_min,
        amount_max=parsed.amount_max,
        limit=limit,
    )


def hybrid_invoice_search(query: str, parsed: ParsedQuery, k: int = 5) -> list[InvoiceRecord]:
    """
    Combine structured metadata filtering with semantic search,
    deduplicate, and return up to k results.
    """
    structured = query_invoices_structured(parsed, limit=k * 2)
    semantic = []

    # Only run semantic search when no strong metadata signals
    if not structured or parsed.similar_intent:
        semantic = query_invoices_semantic(query, k=k)

    # Merge: structured results first (they matched exact filters)
    seen: set[str] = set()
    merged: list[InvoiceRecord] = []
    for rec in structured + semantic:
        if rec.id not in seen:
            seen.add(rec.id)
            merged.append(rec)
        if len(merged) >= k:
            break

    return merged


# ── LLM Q&A over invoice context ─────────────────────────────────────────────

_INVOICE_QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert invoice analyst. Answer the user's question using ONLY the invoice data provided.\n"
        "Be precise with numbers and dates. If the answer isn't in the data, say so clearly.\n\n"
        "Invoice Data:\n{context}",
    ),
    ("human", "{question}"),
])


def answer_invoice_query(question: str, parsed: ParsedQuery) -> dict:
    """
    Find relevant invoices using hybrid search, then answer the question via LLM.

    Search strategy:
      - If the query has no specific metadata filters (vendor/date/amount) AND
        no similarity intent → fetch ALL invoices so nothing is missed.
      - Otherwise use hybrid retrieval (structured filters + semantic search).

    Returns {answer, invoices: [...], total_searched: N}
    """
    has_filters = any([
        parsed.vendor_hint,
        parsed.date_hint,
        parsed.amount_min is not None,
        parsed.amount_max is not None,
    ])

    if not has_filters and not parsed.similar_intent:
        # No specific filters → use ALL stored invoices for full coverage
        all_records = invoice_db.list_invoices()
        records = all_records
        search_mode = "all"
    else:
        # Filtered / similarity search
        records = hybrid_invoice_search(question, parsed, k=50)  # generous cap
        search_mode = "filtered"

    if not records:
        return {
            "answer": "No invoices found. Try uploading some invoice PDFs first.",
            "invoices": [],
            "total_searched": 0,
        }

    # Build context from records
    context_parts = []
    for r in records:
        li_text = ""
        if r.line_items:
            li_text = "\n  Line Items:\n" + "\n".join(
                f"    - {li.description}"
                + (f" | Qty: {li.quantity}" if li.quantity else "")
                + (f" | Unit: ${li.unit_price}" if li.unit_price else "")
                + (f" | Total: ${li.total}" if li.total else "")
                for li in r.line_items
            )
        context_parts.append(
            f"Invoice ID: {r.id}\n"
            f"  File: {r.original_filename}\n"
            f"  Vendor: {r.vendor_name}\n"
            f"  Date: {r.invoice_date}\n"
            f"  Total Amount: ${r.total_amount:.2f}"
            + li_text
        )

    context = "\n\n---\n\n".join(context_parts)
    chain = _INVOICE_QA_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "invoices": [_record_to_dict(r) for r in records],
        "total_searched": len(records),
        "search_mode": search_mode,
    }


# ── Comparison Engine ─────────────────────────────────────────────────────────

_COMPARISON_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert financial analyst. A user wants to understand the difference between two invoices.\n"
        "Based on the structured diff below, write a clear, insightful natural-language explanation.\n"
        "Highlight the most significant differences. Use specific numbers and percentages where possible.\n"
        "Keep it under 150 words.\n\n"
        "Invoice A: {invoice_a_summary}\n"
        "Invoice B: {invoice_b_summary}\n\n"
        "Structured Diff:\n{diff}",
    ),
    ("human", "{user_query}"),
])


def compare_invoices(
    invoice_a: InvoiceRecord,
    invoice_b: InvoiceRecord,
    user_query: str = "Compare these two invoices.",
) -> ComparisonResult:
    """
    Perform a structured diff between two invoices and generate an LLM explanation.
    """
    diff: dict = {}

    # Amount comparison
    if invoice_a.total_amount and invoice_b.total_amount:
        diff_amount = invoice_a.total_amount - invoice_b.total_amount
        pct = (diff_amount / invoice_b.total_amount * 100) if invoice_b.total_amount else 0.0
        diff["total_amount"] = {
            "invoice_a": invoice_a.total_amount,
            "invoice_b": invoice_b.total_amount,
            "difference": round(diff_amount, 2),
            "percent_change": round(pct, 1),
        }

    # Vendor
    diff["vendor_name"] = {
        "invoice_a": invoice_a.vendor_name,
        "invoice_b": invoice_b.vendor_name,
        "same": invoice_a.vendor_name.lower() == invoice_b.vendor_name.lower(),
    }

    # Date
    diff["invoice_date"] = {
        "invoice_a": invoice_a.invoice_date,
        "invoice_b": invoice_b.invoice_date,
    }

    # Line items count
    diff["line_items_count"] = {
        "invoice_a": len(invoice_a.line_items),
        "invoice_b": len(invoice_b.line_items),
    }

    # Line item matching (by description similarity)
    if invoice_a.line_items and invoice_b.line_items:
        items_diff = _compare_line_items(invoice_a.line_items, invoice_b.line_items)
        if items_diff:
            diff["line_items_detail"] = items_diff

    # LLM explanation
    chain = _COMPARISON_PROMPT | get_llm() | StrOutputParser()
    explanation = chain.invoke({
        "invoice_a_summary": f"Vendor={invoice_a.vendor_name}, Date={invoice_a.invoice_date}, Total=${invoice_a.total_amount:.2f}",
        "invoice_b_summary": f"Vendor={invoice_b.vendor_name}, Date={invoice_b.invoice_date}, Total=${invoice_b.total_amount:.2f}",
        "diff": json.dumps(diff, indent=2),
        "user_query": user_query,
    })

    return ComparisonResult(
        invoice_a=_record_to_dict(invoice_a),
        invoice_b=_record_to_dict(invoice_b),
        diff=diff,
        explanation=explanation,
    )


def _compare_line_items(items_a: list[LineItem], items_b: list[LineItem]) -> list[dict]:
    """Match line items by description and compute differences."""
    results = []
    b_map = {item.description.lower(): item for item in items_b if item.description}

    for item_a in items_a:
        key = item_a.description.lower()
        item_b = b_map.get(key)
        entry: dict = {"description": item_a.description}
        if item_b:
            entry["total_a"] = item_a.total
            entry["total_b"] = item_b.total
            if item_a.total is not None and item_b.total is not None:
                entry["difference"] = round(item_a.total - item_b.total, 2)
        else:
            entry["status"] = "only_in_invoice_a"
        results.append(entry)

    a_keys = {item.description.lower() for item in items_a if item.description}
    for item_b in items_b:
        if item_b.description.lower() not in a_keys:
            results.append({"description": item_b.description, "status": "only_in_invoice_b"})

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        if isinstance(val, str):
            val = val.replace(",", "").replace("$", "").strip()
        return float(val)
    except (ValueError, TypeError):
        return None


def _record_to_dict(record: InvoiceRecord) -> dict:
    return {
        "id": record.id,
        "vendor_name": record.vendor_name,
        "invoice_date": record.invoice_date,
        "total_amount": record.total_amount,
        "line_items_count": len(record.line_items),
        "line_items": [li.model_dump() for li in record.line_items],
        "original_filename": record.original_filename,
        "uploaded_at": record.uploaded_at,
    }
