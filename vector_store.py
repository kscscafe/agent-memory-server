"""sqlite-vec integration for semantic_memories.

Maintains a parallel `semantic_vec` virtual table keyed by the same rowid
(=semantic_memories.id). FTS5 stays as-is; this is an additional index.
"""
from __future__ import annotations

from typing import Optional

import sqlite_vec

from embeddings import DIM, embed, embed_for_row


VEC_EXT_PATH = sqlite_vec.loadable_path()


async def load_vec_extension(db) -> None:
    """Load the sqlite-vec extension on an aiosqlite connection."""
    await db.enable_load_extension(True)
    await db.load_extension(VEC_EXT_PATH)
    await db.enable_load_extension(False)


async def init_vec_table(db) -> None:
    """Create the semantic_vec virtual table if missing.

    Extension must already be loaded on this connection.
    """
    await db.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS semantic_vec "
        f"USING vec0(embedding float[{DIM}])"
    )


async def upsert_embedding(db, rowid: int, key: str, value: str) -> None:
    e = embed_for_row(key, value)
    await db.execute("DELETE FROM semantic_vec WHERE rowid = ?", (rowid,))
    await db.execute(
        "INSERT INTO semantic_vec(rowid, embedding) VALUES (?, ?)",
        (rowid, e),
    )


async def delete_embedding(db, rowid: int) -> None:
    await db.execute("DELETE FROM semantic_vec WHERE rowid = ?", (rowid,))


async def search_vec(
    db, query: str, limit: int = 20, agent: Optional[str] = None
) -> list[dict]:
    """K-NN search via sqlite-vec, joined with semantic_memories metadata.

    Returns rows ordered by ascending distance (lower = more similar).
    Filtering by agent is applied after the KNN step (post-filter); for the
    current corpus size (<1k rows) this is fine.
    """
    q = embed(query)
    sql = (
        "SELECT s.id, s.key, s.value, s.category, s.agent, "
        "       s.created_at, s.updated_at, v.distance "
        "FROM semantic_vec v "
        "JOIN semantic_memories s ON s.id = v.rowid "
        "WHERE v.embedding MATCH ? AND k = ? "
    )
    params: list = [q, limit * 4 if agent else limit]
    if agent:
        sql += "AND s.agent = ? "
        params.append(agent)
    sql += "ORDER BY v.distance LIMIT ?"
    params.append(limit)

    db.row_factory = __import__("aiosqlite").Row
    async with db.execute(sql, params) as cur:
        rows = [dict(r) for r in await cur.fetchall()]
    return rows
