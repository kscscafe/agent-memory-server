"""Inbox: low-friction capture endpoint.

REDACTEDの「外部脳」入口。LINE webhook / Slack / iPhoneショートカット / MCP どこから
入っても、一言を未分類のまま `inbox` に放り込む。後段（operatorのバッチ）で分類・
昇格する。書式・カテゴリ・最小文字数などの要求は一切なし。content 非空のみ検査。

semantic_memories の owner-gate / category 検査は適用しない。
"""
from pathlib import Path
from typing import Optional

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel


DB_PATH = Path(__file__).resolve().parent / "memory.db"

VALID_INBOX_SOURCES: frozenset[str] = frozenset({
    "line", "slack", "mcp", "api", "shortcut",
})
VALID_INBOX_STATUSES: tuple[str, ...] = (
    "unprocessed", "promoted", "discarded",
)

router = APIRouter(prefix="/api/inbox", tags=["inbox"])


class InboxCreate(BaseModel):
    content: str
    source: str = "api"
    media_path: Optional[str] = None


class InboxPatch(BaseModel):
    status: Optional[str] = None
    promoted_to: Optional[int] = None


def _row_to_dict(row: aiosqlite.Row) -> dict:
    return {k: row[k] for k in row.keys()}


async def save_inbox_item(
    content: str,
    source: str = "line",
    media_path: Optional[str] = None,
) -> int:
    """Insert one raw inbox item; returns the new row id.

    Raises ValueError if content is empty after stripping. The only validation
    in the inbox path — keeps the input surface as wide as possible.
    """
    content = (content or "").strip()
    if not content:
        raise ValueError("content must be non-empty")
    src = source if source in VALID_INBOX_SOURCES else "api"
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO inbox (content, source, media_path) VALUES (?, ?, ?)",
            (content, src, media_path),
        )
        await db.commit()
        return cur.lastrowid


def build_router(verify_api_key) -> APIRouter:
    """Bind the project-wide X-API-Key dep and return the router."""

    @router.post("")
    async def create_inbox(
        payload: InboxCreate, api_key: str = Depends(verify_api_key)
    ):
        try:
            new_id = await save_inbox_item(
                payload.content, payload.source, payload.media_path
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM inbox WHERE id = ?", (new_id,)
            ) as cur:
                row = await cur.fetchone()
        return _row_to_dict(row)

    @router.get("")
    async def list_inbox(
        status: str = Query(
            "unprocessed",
            pattern="^(unprocessed|promoted|discarded|all)$",
        ),
        limit: int = Query(50, ge=1, le=500),
        api_key: str = Depends(verify_api_key),
    ):
        sql = "SELECT * FROM inbox"
        params: list = []
        if status != "all":
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql, params) as cur:
                rows = [_row_to_dict(r) for r in await cur.fetchall()]
        return {"status": status, "results": rows}

    @router.patch("/{inbox_id}")
    async def patch_inbox(
        inbox_id: int,
        payload: InboxPatch,
        api_key: str = Depends(verify_api_key),
    ):
        fields = {k: v for k, v in payload.dict().items() if v is not None}
        if not fields:
            raise HTTPException(status_code=400, detail="no fields to update")
        if "status" in fields and fields["status"] not in VALID_INBOX_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=(
                    "status must be one of: "
                    + ", ".join(VALID_INBOX_STATUSES)
                ),
            )
        sets = ", ".join(f"{k} = ?" for k in fields.keys())
        values = list(fields.values()) + [inbox_id]
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute(
                f"UPDATE inbox SET {sets} WHERE id = ?", values
            )
            await db.commit()
            if cur.rowcount == 0:
                raise HTTPException(
                    status_code=404, detail="inbox item not found"
                )
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM inbox WHERE id = ?", (inbox_id,)
            ) as q:
                row = await q.fetchone()
        return _row_to_dict(row)

    return router
