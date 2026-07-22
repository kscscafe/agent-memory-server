"""One-shot migration: create semantic_vec table and backfill embeddings.

Idempotent — running twice just re-embeds every row.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import aiosqlite

from vector_store import init_vec_table, load_vec_extension, upsert_embedding


DB_PATH = Path(__file__).resolve().parent / "memory.db"


async def main() -> int:
    t0 = time.time()
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await load_vec_extension(db)
        await init_vec_table(db)

        async with db.execute(
            "SELECT id, key, value FROM semantic_memories ORDER BY id"
        ) as cur:
            rows = await cur.fetchall()

        print(f"[migrate] backfilling {len(rows)} rows...")
        for i, (rid, key, value) in enumerate(rows, 1):
            await upsert_embedding(db, rid, key, value)
            if i % 25 == 0:
                print(f"  {i}/{len(rows)} done")
        await db.commit()

        async with db.execute(
            "SELECT COUNT(*) FROM semantic_vec"
        ) as cur:
            (n,) = await cur.fetchone()

    elapsed = time.time() - t0
    print(f"[migrate] semantic_vec row count = {n}, elapsed = {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
