"""Auto-approve pending_decisions and promote them to semantic_memories.

Called every 10 minutes by scheduler.py. Eligibility:
  - category ∈ AUTO_APPROVE_CATEGORIES
  - key and value are non-empty strings
  - source != 'codex' (Codex-proposed candidates must be reviewed by the owner
    before promotion; Phase 1 policy)

For each approved row we:
  1. INSERT/UPSERT into semantic_memories, carrying source / source_reference
     / scope from the pending row so provenance and global visibility survive.
     (owner is intentionally NOT set here — that stays the existing behaviour
     of leaving new auto-approved rows with owner NULL; Phase 2 will address
     the pre-existing NULL-owner backlog separately.)
  2. Sync the row to semantic_vec (vector embedding)
  3. UPDATE pending_decisions.status = 'approved'

Then send a Slack DM to the owner summarising what was promoted.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import aiosqlite

from slack_client import send_slack_message
from vector_store import load_vec_extension, upsert_embedding

DB_PATH = Path(__file__).resolve().parent / "memory.db"
AUTO_APPROVE_CATEGORIES: set[str] = {
    "design_decision", "surface", "infra", "agent_policy",
    "task", "study", "app_status", "session", "reminder",
}


def _eligible(row: dict) -> bool:
    category = (row.get("category") or "").strip()
    key = (row.get("key") or "").strip()
    value = (row.get("value") or "").strip()
    source = (row.get("source") or "").strip()
    return (
        category in AUTO_APPROVE_CATEGORIES
        and bool(key)
        and bool(value)
        # Phase 1: Codex-proposed candidates must be reviewed by the owner.
        # They surface via notify_stale_pending()'s daily Slack digest.
        and source != "codex"
    )


async def approve_pending() -> list[str]:
    """Promote eligible pending rows. Returns list of approved keys."""
    approved_keys: list[str] = []
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await load_vec_extension(db)

        async with db.execute(
            "SELECT * FROM pending_decisions WHERE status='pending' "
            "ORDER BY created_at"
        ) as cur:
            pending = [dict(r) for r in await cur.fetchall()]

        for row in pending:
            if not _eligible(row):
                continue
            # Phase 1: forward source / source_reference / scope from the
            # pending row so provenance and global visibility survive
            # promotion. owner is intentionally omitted here (behaviour
            # unchanged from before Phase 1 — new rows get owner NULL, existing
            # rows keep their owner via ON CONFLICT).
            await db.execute(
                """
                INSERT INTO semantic_memories
                    (key, value, category, agent,
                     source, source_reference, scope)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    agent = excluded.agent,
                    source = excluded.source,
                    source_reference = excluded.source_reference,
                    scope = excluded.scope,
                    updated_at = datetime('now')
                """,
                (
                    row["key"], row["value"], row["category"], row["agent"],
                    row.get("source"),
                    row.get("source_reference"),
                    row.get("scope") or "agent",
                ),
            )
            async with db.execute(
                "SELECT id FROM semantic_memories WHERE key = ?",
                (row["key"],),
            ) as r:
                sem = await r.fetchone()
            try:
                await upsert_embedding(
                    db, sem["id"], row["key"], row["value"]
                )
            except Exception as e:
                print(
                    f"[pending-approver] embed failed for {row['key']}: {e}",
                    flush=True,
                )
            await db.execute(
                "UPDATE pending_decisions SET status='approved' WHERE id=?",
                (row["id"],),
            )
            approved_keys.append(row["key"])

        await db.commit()
    return approved_keys


async def approve_and_notify() -> int:
    """Run approvals + Slack notification. Returns approved count."""
    approved = await approve_pending()
    n = len(approved)
    if n == 0:
        return 0
    head = approved[0]
    tail = f" 他 {n - 1} 件" if n > 1 else ""
    msg = f"確定事項 {n} 件をAMSに登録しました：{head}{tail}"
    await send_slack_message(msg)
    print(f"[pending-approver] approved {n}: {approved}", flush=True)
    return n


STALE_THRESHOLD_HOURS = 24
RENOTIFY_COOLDOWN_HOURS = 24


async def list_stale_pending(
    min_age_hours: int = STALE_THRESHOLD_HOURS,
    skip_recently_notified: bool = True,
) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        sql = (
            "SELECT * FROM pending_decisions "
            "WHERE status='pending' "
            "AND datetime(created_at) <= datetime('now', ?) "
        )
        params: list = [f"-{min_age_hours} hours"]
        if skip_recently_notified:
            sql += (
                "AND (notified_at IS NULL "
                "OR datetime(notified_at) <= datetime('now', ?)) "
            )
            params.append(f"-{RENOTIFY_COOLDOWN_HOURS} hours")
        sql += "ORDER BY created_at"
        async with db.execute(sql, params) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def _mark_notified(ids: list[int]) -> None:
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE pending_decisions SET notified_at = datetime('now') "
            f"WHERE id IN ({placeholders})",
            ids,
        )
        await db.commit()


def _age_days(created_at: str) -> int:
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(created_at.replace(" ", "T"))
    except ValueError:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - dt).days)


async def notify_stale_pending() -> int:
    """Slack DM about pending candidates the auto-approver won't take.

    Why: AUTO_APPROVE_CATEGORIES is intentionally narrow, so rows with other
    categories sit in 'pending' forever with no surface. This daily nag makes
    them visible so the owner can manually approve/reject.
    """
    stale = await list_stale_pending()
    if not stale:
        return 0
    lines = [f"⏳ 手動レビュー待ちの候補 {len(stale)} 件 (≥{STALE_THRESHOLD_HOURS}h pending):"]
    for row in stale:
        agent = row.get("agent") or "—"
        lines.append(
            f"• id={row['id']} [{row['category']}] {row['key']} "
            f"(agent: {agent}, {_age_days(row['created_at'])}日経過)"
        )
    lines.append("")
    lines.append("approve: `curl -X PATCH -H \"X-API-Key: $AMS_API_KEY\" "
                 "-H 'Content-Type: application/json' -d '{\"action\":\"approve\"}' "
                 "http://localhost:8000/memory/pending/<id>`")
    lines.append("reject:  同上 で `\"action\":\"reject\"`")
    await send_slack_message("\n".join(lines))
    ids = [r["id"] for r in stale]
    await _mark_notified(ids)
    print(f"[pending-approver] stale notify {len(stale)} ids={ids}", flush=True)
    return len(stale)


if __name__ == "__main__":
    n = asyncio.run(approve_and_notify())
    sys.exit(0 if n >= 0 else 1)
