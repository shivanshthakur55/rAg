"""
Invoice Database Layer — SQLite-backed persistent storage for structured invoice data.

Tables:
  invoices — one row per uploaded invoice with all extracted structured fields

All functions are synchronous (SQLite doesn't need async for small workloads).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from rag_api.models import InvoiceRecord, LineItem

# ── Path setup ────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent.parent / "data" / "invoices.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Schema ────────────────────────────────────────────────────────────────────
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS invoices (
    id              TEXT PRIMARY KEY,
    vendor_name     TEXT NOT NULL DEFAULT '',
    invoice_date    TEXT NOT NULL DEFAULT '',
    total_amount    REAL NOT NULL DEFAULT 0.0,
    line_items      TEXT NOT NULL DEFAULT '[]',   -- JSON array
    raw_text        TEXT NOT NULL DEFAULT '',
    file_path       TEXT NOT NULL DEFAULT '',
    original_filename TEXT NOT NULL DEFAULT '',
    uploaded_at     TEXT NOT NULL DEFAULT ''
);
"""


# ── Connection helper ─────────────────────────────────────────────────────────

@contextmanager
def _conn():
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the invoices table if it doesn't exist. Call once on startup."""
    with _conn() as con:
        con.execute(_CREATE_TABLE)


# ── CRUD ──────────────────────────────────────────────────────────────────────

def save_invoice(record: InvoiceRecord) -> None:
    """Insert or replace an invoice record."""
    line_items_json = json.dumps(
        [item.model_dump() for item in record.line_items]
    )
    uploaded_at = record.uploaded_at or datetime.utcnow().isoformat()

    with _conn() as con:
        con.execute(
            """
            INSERT OR REPLACE INTO invoices
              (id, vendor_name, invoice_date, total_amount, line_items,
               raw_text, file_path, original_filename, uploaded_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                record.id,
                record.vendor_name,
                record.invoice_date,
                record.total_amount,
                line_items_json,
                record.raw_text,
                record.file_path,
                record.original_filename,
                uploaded_at,
            ),
        )


def get_invoice(invoice_id: str) -> Optional[InvoiceRecord]:
    """Fetch a single invoice by its ID. Returns None if not found."""
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM invoices WHERE id = ?", (invoice_id,)
        ).fetchone()
    return _row_to_record(row) if row else None


def list_invoices() -> list[InvoiceRecord]:
    """Return all stored invoices ordered by upload date."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM invoices ORDER BY uploaded_at DESC"
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def query_invoices(
    vendor_name: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    amount_min: Optional[float] = None,
    amount_max: Optional[float] = None,
    limit: int = 20,
) -> list[InvoiceRecord]:
    """
    Filter invoices by one or more metadata dimensions.
    All parameters are optional; unset params are ignored.
    """
    clauses: list[str] = []
    params: list = []

    if vendor_name:
        clauses.append("LOWER(vendor_name) LIKE ?")
        params.append(f"%{vendor_name.lower()}%")

    if date_from:
        clauses.append("invoice_date >= ?")
        params.append(date_from)

    if date_to:
        clauses.append("invoice_date <= ?")
        params.append(date_to)

    if amount_min is not None:
        clauses.append("total_amount >= ?")
        params.append(amount_min)

    if amount_max is not None:
        clauses.append("total_amount <= ?")
        params.append(amount_max)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    with _conn() as con:
        rows = con.execute(
            f"SELECT * FROM invoices {where} ORDER BY uploaded_at DESC LIMIT ?",
            params,
        ).fetchall()

    return [_row_to_record(r) for r in rows]


def delete_invoice(invoice_id: str) -> bool:
    """Delete an invoice. Returns True if a row was deleted."""
    with _conn() as con:
        cur = con.execute("DELETE FROM invoices WHERE id = ?", (invoice_id,))
    return cur.rowcount > 0


# ── Helper ────────────────────────────────────────────────────────────────────

def _row_to_record(row: sqlite3.Row) -> InvoiceRecord:
    raw_items = json.loads(row["line_items"] or "[]")
    line_items = [LineItem(**item) for item in raw_items]
    return InvoiceRecord(
        id=row["id"],
        vendor_name=row["vendor_name"],
        invoice_date=row["invoice_date"],
        total_amount=row["total_amount"],
        line_items=line_items,
        raw_text=row["raw_text"],
        file_path=row["file_path"],
        original_filename=row["original_filename"],
        uploaded_at=row["uploaded_at"],
    )
