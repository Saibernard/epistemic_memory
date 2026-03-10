"""
Temporal Grounding for the Memory Layer.

Extracts, resolves, and scores temporal references in memory content.
Supports both document dates (when content was authored) and event dates
(when the described event occurred/will occur).

Inspired by Supermemory's dual temporal grounding and HydraDB's
t_commit / t_valid architecture.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class TemporalRef:
    """A single temporal reference extracted from text."""
    raw_text: str
    resolved_date: Optional[float] = None  # unix timestamp
    ref_type: str = "unknown"  # absolute, relative, duration, recurring
    description: str = ""
    confidence: float = 0.5


# Common relative date patterns and their resolution logic
_RELATIVE_PATTERNS: List[Tuple[str, Any]] = [
    (r"\byesterday\b", lambda now: now - timedelta(days=1)),
    (r"\btoday\b", lambda now: now),
    (r"\btomorrow\b", lambda now: now + timedelta(days=1)),
    (r"\blast\s+week\b", lambda now: now - timedelta(weeks=1)),
    (r"\bnext\s+week\b", lambda now: now + timedelta(weeks=1)),
    (r"\blast\s+month\b", lambda now: now.replace(day=1) - timedelta(days=1)),
    (r"\bnext\s+month\b", lambda now: (now.replace(day=28) + timedelta(days=4)).replace(day=1)),
    (r"\blast\s+year\b", lambda now: now.replace(year=now.year - 1)),
    (r"\bnext\s+year\b", lambda now: now.replace(year=now.year + 1)),
    (r"\b(\d+)\s+days?\s+ago\b", None),
    (r"\b(\d+)\s+weeks?\s+ago\b", None),
    (r"\b(\d+)\s+months?\s+ago\b", None),
    (r"\b(\d+)\s+years?\s+ago\b", None),
    (r"\bin\s+(\d+)\s+days?\b", None),
    (r"\bin\s+(\d+)\s+weeks?\b", None),
]

_ABSOLUTE_DATE_PATTERNS = [
    (r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", "iso"),
    (r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", "mdy"),
    (r"\b(\d{1,2})/(\d{1,2})/(\d{2})\b", "mdy_short"),
    (r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})\b", "month_name"),
    (r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s*(\d{4})\b", "day_month_name"),
    (r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})\b", "month_abbr"),
]

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}


def extract_temporal_refs(
    text: str,
    anchor: Optional[datetime] = None,
) -> List[TemporalRef]:
    """
    Extract temporal references from text using regex heuristics.

    Args:
        text: The text to scan for temporal references.
        anchor: The reference date for resolving relative dates.
                Defaults to now.

    Returns:
        List of TemporalRef objects with resolved unix timestamps.
    """
    if not text:
        return []

    anchor = anchor or datetime.now()
    refs: List[TemporalRef] = []

    for pattern, fmt in _ABSOLUTE_DATE_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            try:
                dt = _parse_absolute_match(m, fmt)
                if dt:
                    refs.append(TemporalRef(
                        raw_text=m.group(0),
                        resolved_date=dt.timestamp(),
                        ref_type="absolute",
                        description=dt.strftime("%Y-%m-%d"),
                        confidence=0.9,
                    ))
            except (ValueError, OverflowError):
                pass

    for pattern, resolver in _RELATIVE_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            try:
                if resolver is not None:
                    dt = resolver(anchor)
                else:
                    dt = _resolve_relative_match(m, anchor)
                if dt:
                    refs.append(TemporalRef(
                        raw_text=m.group(0),
                        resolved_date=dt.timestamp(),
                        ref_type="relative",
                        description=dt.strftime("%Y-%m-%d"),
                        confidence=0.7,
                    ))
            except (ValueError, OverflowError):
                pass

    seen = set()
    deduped = []
    for r in refs:
        key = (r.raw_text.lower(), r.resolved_date)
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return deduped


def _parse_absolute_match(m: re.Match, fmt: str) -> Optional[datetime]:
    """Parse an absolute date regex match into a datetime."""
    if fmt == "iso":
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    elif fmt == "mdy":
        return datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)))
    elif fmt == "mdy_short":
        year = int(m.group(3))
        year = year + 2000 if year < 50 else year + 1900
        return datetime(year, int(m.group(1)), int(m.group(2)))
    elif fmt == "month_name":
        month = _MONTH_MAP.get(m.group(1).lower())
        if month:
            return datetime(int(m.group(3)), month, int(m.group(2)))
    elif fmt == "day_month_name":
        month = _MONTH_MAP.get(m.group(2).lower())
        if month:
            return datetime(int(m.group(3)), month, int(m.group(1)))
    elif fmt == "month_abbr":
        month = _MONTH_MAP.get(m.group(1).lower())
        if month:
            return datetime(int(m.group(3)), month, int(m.group(2)))
    return None


def _resolve_relative_match(m: re.Match, anchor: datetime) -> Optional[datetime]:
    """Resolve a relative date regex match."""
    text = m.group(0).lower()
    num = int(m.group(1))
    if "day" in text:
        delta = timedelta(days=num)
    elif "week" in text:
        delta = timedelta(weeks=num)
    elif "month" in text:
        delta = timedelta(days=num * 30)
    elif "year" in text:
        delta = timedelta(days=num * 365)
    else:
        return None

    if "ago" in text:
        return anchor - delta
    elif "in " in text:
        return anchor + delta
    return None


def resolve_relative_date(
    ref_text: str,
    anchor_date: Optional[float] = None,
) -> Optional[float]:
    """
    Resolve a single relative date string to a unix timestamp.

    Args:
        ref_text: Relative date expression (e.g., "yesterday", "3 days ago")
        anchor_date: Unix timestamp anchor. Defaults to now.

    Returns:
        Unix timestamp or None if not parseable.
    """
    anchor = datetime.fromtimestamp(anchor_date) if anchor_date else datetime.now()
    refs = extract_temporal_refs(ref_text, anchor)
    return refs[0].resolved_date if refs else None


def temporal_relevance(
    query_text: str,
    memory_event_dates: Optional[List[Dict[str, Any]]],
    memory_document_date: Optional[float] = None,
    query_anchor: Optional[float] = None,
) -> float:
    """
    Compute how temporally relevant a memory is to a query.

    Returns a score from 0.0 to 1.0 where:
    - 1.0 = exact date match
    - 0.7+ = same week
    - 0.4+ = same month
    - 0.1+ = same year
    - 0.0 = no temporal relevance or no dates to compare

    Args:
        query_text: The query string (temporal refs extracted from it).
        memory_event_dates: List of dicts with "date" (unix ts), "type", "description".
        memory_document_date: Unix timestamp of when the content was authored.
        query_anchor: Reference time for query's relative dates.
    """
    if not memory_event_dates and not memory_document_date:
        return 0.0

    query_refs = extract_temporal_refs(query_text)
    if not query_refs:
        return 0.0

    query_dates = [r.resolved_date for r in query_refs if r.resolved_date]
    if not query_dates:
        return 0.0

    memory_dates = []
    if memory_event_dates:
        for ed in memory_event_dates:
            d = ed.get("date")
            if isinstance(d, (int, float)):
                memory_dates.append(float(d))
    if memory_document_date:
        memory_dates.append(float(memory_document_date))

    if not memory_dates:
        return 0.0

    best_score = 0.0
    for qd in query_dates:
        for md in memory_dates:
            diff_seconds = abs(qd - md)
            diff_days = diff_seconds / 86400.0

            if diff_days < 1:
                score = 1.0
            elif diff_days < 7:
                score = 0.8 - (diff_days / 7) * 0.1
            elif diff_days < 30:
                score = 0.6 - (diff_days / 30) * 0.2
            elif diff_days < 365:
                score = 0.3 - (diff_days / 365) * 0.2
            else:
                score = max(0.0, 0.1 - (diff_days / 3650) * 0.1)

            best_score = max(best_score, score)

    return best_score


def temporal_sort(
    memories: List[Any],
    direction: str = "newest",
    date_key: str = "event_dates",
) -> List[Any]:
    """
    Sort memories by their event date, falling back to created_at.

    Args:
        memories: List of Memory objects.
        direction: "newest" (descending) or "oldest" (ascending).
        date_key: Metadata key for event dates.
    """
    def _sort_key(m):
        event_dates = m.metadata.get(date_key, [])
        if event_dates:
            dates = [ed.get("date", 0) for ed in event_dates if ed.get("date")]
            if dates:
                return max(dates) if direction == "newest" else min(dates)
        doc_date = m.metadata.get("document_date")
        if doc_date:
            return doc_date
        return m.created_at

    return sorted(memories, key=_sort_key, reverse=(direction == "newest"))


def has_temporal_intent(query: str) -> bool:
    """Check if a query has temporal intent."""
    return bool(re.search(
        r"\b(when|date|year|month|time|before|after|ago|last|recent|"
        r"yesterday|tomorrow|today|next|previous|history|schedule|"
        r"appointment|deadline|until|since|during)\b",
        query.lower(),
    ))


def has_historical_intent(query: str) -> bool:
    """Check if a query asks about historical/past versions of information."""
    return bool(re.search(
        r"\b(before|previous|previously|used\s+to|formerly|"
        r"old|older|earlier|history|original|first|initial|"
        r"changed\s+from|switched\s+from|moved\s+from)\b",
        query.lower(),
    ))
