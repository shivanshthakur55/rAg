"""
Query Parser — extracts structured intent from free-text invoice queries.

Detects:
  - vendor names (keyword match + heuristic)
  - dates / date ranges
  - amount filters (above / below / between)
  - comparison intent
  - "this invoice" vs stored-invoice references
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Regex patterns ────────────────────────────────────────────────────────────

_COMPARE_KEYWORDS = re.compile(
    r"\b(compare|vs\.?|versus|differ|difference|higher|lower|more|less|than)\b",
    re.IGNORECASE,
)

_AMOUNT_ABOVE = re.compile(
    r"\b(?:above|over|more\s+than|greater\s+than|exceeds?)\s+\$?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)

_AMOUNT_BELOW = re.compile(
    r"\b(?:below|under|less\s+than|fewer\s+than)\s+\$?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)

_SIMILAR = re.compile(r"\bsimilar\b", re.IGNORECASE)
_THIS_INVOICE = re.compile(r"\bthis\s+invoice\b", re.IGNORECASE)

# Simple date patterns: "March 2024", "2024-03", "03/2024", "last month"
_DATE_MONTH_YEAR = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{4})\b",
    re.IGNORECASE,
)
_DATE_ISO = re.compile(r"\b(\d{4}[-/]\d{2}(?:[-/]\d{2})?)\b")
_DATE_RELATIVE = re.compile(r"\b(last\s+(?:month|year|week)|this\s+(?:month|year))\b", re.IGNORECASE)

MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ParsedQuery:
    original: str
    intent: str = "general"          # "general" | "compare" | "filter" | "similar"
    vendor_hint: Optional[str] = None
    date_hint: Optional[str] = None  # YYYY-MM or YYYY-MM-DD
    amount_min: Optional[float] = None
    amount_max: Optional[float] = None
    references_this_invoice: bool = False
    comparison_intent: bool = False
    similar_intent: bool = False
    extracted_dates: list[str] = field(default_factory=list)


# ── Core parser ───────────────────────────────────────────────────────────────

def parse_query(query: str) -> ParsedQuery:
    """
    Parse a raw user query and extract structured filters / intent signals.
    """
    pq = ParsedQuery(original=query)

    # ── Comparison intent ─────────────────────────────────────────────────
    if _COMPARE_KEYWORDS.search(query):
        pq.comparison_intent = True
        pq.intent = "compare"

    # ── Similar invoice intent ────────────────────────────────────────────
    if _SIMILAR.search(query):
        pq.similar_intent = True
        if pq.intent == "general":
            pq.intent = "similar"

    # ── "This invoice" reference ──────────────────────────────────────────
    if _THIS_INVOICE.search(query):
        pq.references_this_invoice = True

    # ── Amount filters ────────────────────────────────────────────────────
    above_match = _AMOUNT_ABOVE.search(query)
    if above_match:
        pq.amount_min = float(above_match.group(1).replace(",", ""))
        pq.intent = "filter"

    below_match = _AMOUNT_BELOW.search(query)
    if below_match:
        pq.amount_max = float(below_match.group(1).replace(",", ""))
        pq.intent = "filter"

    # ── Date extraction ───────────────────────────────────────────────────
    month_year = _DATE_MONTH_YEAR.findall(query)
    for month_name, year in month_year:
        month_num = MONTH_MAP.get(month_name.lower(), "01")
        date_str = f"{year}-{month_num}"
        pq.extracted_dates.append(date_str)

    iso_dates = _DATE_ISO.findall(query)
    for d in iso_dates:
        pq.extracted_dates.append(d.replace("/", "-"))

    if pq.extracted_dates:
        pq.date_hint = pq.extracted_dates[0]
        if pq.intent == "general":
            pq.intent = "filter"

    # ── Vendor hint heuristic ─────────────────────────────────────────────
    # Look for patterns like "from XYZ" / "by XYZ" / "vendor XYZ"
    vendor_pattern = re.search(
        r"\b(?:from|by|vendor|supplier|company|invoice\s+from)\s+([A-Z][A-Za-z0-9&\s\-\.]{1,40}?)(?:\s*(?:vendor|invoice|above|below|compare|vs|with|$|\.))",
        query,
    )
    if vendor_pattern:
        pq.vendor_hint = vendor_pattern.group(1).strip()

    # Fallback: quoted name
    quoted = re.search(r'"([^"]+)"', query)
    if quoted and not pq.vendor_hint:
        pq.vendor_hint = quoted.group(1).strip()

    return pq
