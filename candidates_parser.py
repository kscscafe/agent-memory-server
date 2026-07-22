"""Parse <ams:candidates>JSON</ams:candidates> blocks and queue them.

Used by:
  - main.py slack-webhook handler  : parses incoming Slack messages
  - drive_sync._save_episodic      : parses agent .md notes from Drive

The protocol is documented in ~/.claude/CLAUDE.md ("保存候補プロトコル").
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable

DB_PATH = Path(__file__).resolve().parent / "memory.db"

CANDIDATE_RE = re.compile(
    r"<ams:candidates>\s*(?:```(?:json)?\s*)?(.*?)(?:\s*```)?\s*</ams:candidates>",
    re.DOTALL | re.IGNORECASE,
)

REQUIRED_FIELDS: tuple[str, ...] = ("key", "value", "category")


def extract_candidates(text: str) -> list[dict]:
    """Find all <ams:candidates> blocks in `text` and return parsed entries.

    Accepts either a JSON object or a JSON array inside each block. Each
    candidate must have non-empty key/value/category; `agent` is optional.
    """
    if not text:
        return []
    out: list[dict] = []
    for m in CANDIDATE_RE.finditer(text):
        raw = m.group(1).strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            cleaned: dict[str, str | None] = {}
            ok = True
            for field in REQUIRED_FIELDS:
                v = (item.get(field) or "").strip()
                if not v:
                    ok = False
                    break
                cleaned[field] = v
            if not ok:
                continue
            agent = (item.get("agent") or "").strip()
            cleaned["agent"] = agent or None
            out.append(cleaned)
    return out


def queue_candidates(
    candidates: Iterable[dict], source: str = "unknown"
) -> int:
    """Insert candidates into pending_decisions. Returns rows inserted."""
    inserted = 0
    con = sqlite3.connect(DB_PATH)
    try:
        for c in candidates:
            con.execute(
                "INSERT INTO pending_decisions"
                "(key, value, category, agent, source, status) "
                "VALUES (?, ?, ?, ?, ?, 'pending')",
                (c["key"], c["value"], c["category"], c.get("agent"), source),
            )
            inserted += 1
        con.commit()
    finally:
        con.close()
    return inserted


def extract_and_queue(text: str, source: str = "unknown") -> int:
    """Shorthand: parse text and queue everything found. Returns count."""
    return queue_candidates(extract_candidates(text), source=source)
