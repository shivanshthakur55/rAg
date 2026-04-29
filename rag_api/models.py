"""
Pydantic data models for the Dual-Mode RAG API.

Covers:
  - PDF/Document RAG mode (ChatRequest already in main.py, kept for backward compat)
  - Invoice RAG mode  (upload, query, compare)
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── PDF / Document Mode ──────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Ask a question about an ingested document session."""
    session_id: str
    question: str


# ── Invoice Mode ─────────────────────────────────────────────────────────────

class LineItem(BaseModel):
    """A single line-item on an invoice."""
    description: str = ""
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None


class InvoiceRecord(BaseModel):
    """Full invoice schema — stored in SQLite and Chroma."""
    id: str = Field(..., description="Unique invoice ID (UUID or extracted invoice number)")
    vendor_name: str = ""
    invoice_date: str = ""          # ISO format preferred: YYYY-MM-DD
    total_amount: float = 0.0
    line_items: list[LineItem] = Field(default_factory=list)
    raw_text: str = ""
    file_path: str = ""
    original_filename: str = ""
    uploaded_at: str = ""           # ISO datetime string


class InvoiceUploadResponse(BaseModel):
    """Returned after a successful invoice upload."""
    invoice_id: str
    vendor_name: str
    invoice_date: str
    total_amount: float
    line_items_count: int
    status: str = "ready"
    message: str = ""


class InvoiceQueryRequest(BaseModel):
    """Query the invoice collection with optional metadata filters."""
    query: str
    vendor_name: Optional[str] = None
    date_from: Optional[str] = None     # YYYY-MM-DD
    date_to: Optional[str] = None       # YYYY-MM-DD
    amount_min: Optional[float] = None
    amount_max: Optional[float] = None
    limit: int = 5


class InvoiceCompareRequest(BaseModel):
    """
    Compare two invoices.
    - invoice_id_a: always required
    - invoice_id_b: optional; if omitted, use `query` to find the second invoice
    - query: e.g. 'Compare with last invoice from ABC Corp'
    """
    invoice_id_a: str
    invoice_id_b: Optional[str] = None
    query: Optional[str] = None


class ComparisonResult(BaseModel):
    """Structured comparison output between two invoices."""
    invoice_a: dict
    invoice_b: dict
    diff: dict                       # field-by-field diffs
    explanation: str                 # LLM natural-language explanation
