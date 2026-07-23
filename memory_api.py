"""Hybrid memory API: semantic / procedural / episodic memories + FTS5 + vec search.

Mounted onto the main FastAPI app as a router under /memory. Auth reuses the
project-wide X-API-Key dependency from main.
"""
import os
from pathlib import Path
from typing import Optional

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, field_validator

from vector_store import (
    delete_embedding,
    load_vec_extension,
    search_vec,
    upsert_embedding,
)

DB_PATH = Path(__file__).resolve().parent / "memory.db"

router = APIRouter(prefix="/memory", tags=["memory"])


def _row_to_dict(row: aiosqlite.Row) -> dict:
    return {k: row[k] for k in row.keys()}


def _rrf(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across multiple ranked result lists."""
    scores: dict[int, float] = {}
    items: dict[int, dict] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            iid = item["id"]
            scores[iid] = scores.get(iid, 0.0) + 1.0 / (k + rank + 1)
            items[iid] = item
    sorted_ids = sorted(scores, key=lambda i: -scores[i])
    return [{**items[i], "rrf_score": scores[i]} for i in sorted_ids]


ALLOWED_CATEGORIES: frozenset[str] = frozenset({
    "surface",
    "design_decision",
    "infra",
    "agent_policy",
    "app_status",
    "task",
    "study",
})

# Thin-write threshold — values shorter than this trip a warning flag in the
# response (the write still succeeds). Used to surface low-effort memories
# during /api/session/checkout audits.
THIN_VALUE_MIN_LEN = 20

# Free-form markers that agents write into `value` to self-report a memory as
# stale / superseded. Used by the ?stale= filter on GET /memory/semantic to
# surface (or hide) rows that are self-reported as deletable. Patterns are
# intentionally left open (no trailing 】) so compound forms like
# 【陳腐化・削除可】 also match.
STALE_MARKERS: tuple[str, ...] = ("【削除", "【陳腐化", "【解消済み", "【廃止")

# Phase 1 (Codex integration): allowed scope values and source_reference bounds.
VALID_SCOPES: frozenset[str] = frozenset({"agent", "global"})
SOURCE_REFERENCE_MAX_LEN = 2000


def _require_nonempty(v: str, field: str) -> str:
    if v is None or not str(v).strip():
        raise ValueError(f"{field} must be non-empty")
    return str(v).strip()


# Optional allowlist of agent names. Set VALID_AGENTS in the environment to a
# comma-separated list (e.g. "alice,bob,carol") to reject writes whose agent
# field is outside the list. Leave the env var unset (or empty) to accept any
# non-empty agent name — this is the default for fresh installs.
_valid_agents_env = os.environ.get("VALID_AGENTS", "").strip()
VALID_AGENTS: Optional[frozenset[str]] = (
    frozenset(a.strip() for a in _valid_agents_env.split(",") if a.strip())
    if _valid_agents_env
    else None
)


def _require_valid_agent(v: str) -> str:
    s = _require_nonempty(v, "agent")
    if VALID_AGENTS is not None and s not in VALID_AGENTS:
        allowed = ", ".join(sorted(VALID_AGENTS))
        raise ValueError(f"agent must be one of: {allowed}")
    return s


def _require_category(v: str) -> str:
    s = _require_nonempty(v, "category")
    if s not in ALLOWED_CATEGORIES:
        allowed = ", ".join(sorted(ALLOWED_CATEGORIES))
        raise ValueError(f"category must be one of: {allowed}")
    return s


class SemanticMemoryIn(BaseModel):
    key: str
    value: str
    category: str
    agent: str

    @field_validator("agent")
    @classmethod
    def _agent_valid(cls, v: str) -> str:
        return _require_valid_agent(v)

    @field_validator("category")
    @classmethod
    def _category_allowed(cls, v: str) -> str:
        return _require_category(v)


class ProceduralMemoryIn(BaseModel):
    agent: str
    rule: str
    source: Optional[str] = None

    @field_validator("agent")
    @classmethod
    def _agent_valid(cls, v: str) -> str:
        return _require_valid_agent(v)


EPISODIC_SUMMARY_MIN_LEN = 50


class EpisodicMemoryIn(BaseModel):
    agent: str
    summary: str
    topics: Optional[str] = None
    session_date: Optional[str] = None

    @field_validator("agent")
    @classmethod
    def _agent_valid(cls, v: str) -> str:
        return _require_valid_agent(v)

    @field_validator("summary")
    @classmethod
    def _summary_min_length(cls, v: str) -> str:
        s = _require_nonempty(v, "summary")
        if len(s) < EPISODIC_SUMMARY_MIN_LEN:
            raise ValueError(
                f"episodic summary is too short "
                f"(min {EPISODIC_SUMMARY_MIN_LEN} chars)"
            )
        return s


class PendingDecisionIn(BaseModel):
    key: str
    value: str
    category: str
    agent: str
    source: Optional[str] = None
    # Phase 1 (Codex integration): scope controls whether the promoted row is
    # visible only to `agent` (default) or to every agent (`global`, used by
    # Codex proposals). source_reference is an optional free-form citation
    # (URL / justification / evidence) that survives promotion into
    # semantic_memories.
    scope: str = "agent"
    source_reference: Optional[str] = None

    @field_validator("agent")
    @classmethod
    def _agent_valid(cls, v: str) -> str:
        return _require_valid_agent(v)

    @field_validator("category")
    @classmethod
    def _category_allowed(cls, v: str) -> str:
        return _require_category(v)

    @field_validator("scope")
    @classmethod
    def _scope_valid(cls, v: str) -> str:
        if v not in VALID_SCOPES:
            allowed = ", ".join(sorted(VALID_SCOPES))
            raise ValueError(f"scope must be one of: {allowed}")
        return v

    @field_validator("source_reference")
    @classmethod
    def _source_reference_normalize(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        if len(s) > SOURCE_REFERENCE_MAX_LEN:
            raise ValueError(
                f"source_reference too long ({len(s)} chars, "
                f"max {SOURCE_REFERENCE_MAX_LEN})"
            )
        return s


class PendingActionIn(BaseModel):
    action: str  # 'approve' | 'reject'


def _quote_fts(query: str) -> str:
    """Wrap each whitespace-separated token in double quotes for FTS5 MATCH.
    Defangs operators / punctuation in user input and ANDs the tokens."""
    tokens = [t for t in query.strip().split() if t]
    if not tokens:
        return '""'
    return " ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def build_router(verify_api_key) -> APIRouter:
    """Bind the shared API-key dependency from main and return the router."""

    @router.get("/semantic")
    async def list_semantic_memories(
        agent: Optional[str] = Query(None),
        category: Optional[str] = Query(None),
        scope: Optional[str] = Query(None),
        source: Optional[str] = Query(None),
        stale: Optional[bool] = Query(
            None,
            description=(
                "True: only rows whose value contains a stale marker "
                "(【削除】/【解消済み】/【陳腐化】). False: exclude such rows. "
                "None: no filter."
            ),
        ),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        api_key: str = Depends(verify_api_key),
    ):
        """Return a paginated, filterable catalogue without full values."""
        if agent is not None and VALID_AGENTS is not None and agent not in VALID_AGENTS:
            raise HTTPException(status_code=400, detail="invalid agent")
        if category is not None and category not in ALLOWED_CATEGORIES:
            raise HTTPException(status_code=400, detail="invalid category")
        if scope is not None and scope not in VALID_SCOPES:
            raise HTTPException(status_code=400, detail="invalid scope")

        conditions: list[str] = []
        params: list = []
        for column, value in (
            ("agent", agent),
            ("category", category),
            ("scope", scope),
            ("source", source),
        ):
            if value is not None:
                conditions.append(f"{column} = ?")
                params.append(value)

        if stale is True:
            like_clauses = " OR ".join(["value LIKE ?"] * len(STALE_MARKERS))
            conditions.append(f"({like_clauses})")
            params.extend(f"%{marker}%" for marker in STALE_MARKERS)
        elif stale is False:
            for marker in STALE_MARKERS:
                conditions.append("value NOT LIKE ?")
                params.append(f"%{marker}%")

        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        count_sql = "SELECT COUNT(*) FROM semantic_memories" + where
        list_sql = (
            "SELECT id, key, category, agent, owner, scope, source, "
            "source_reference, created_at, updated_at, "
            "substr(value, 1, 200) AS value_preview "
            "FROM semantic_memories" + where +
            " ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?"
        )
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(count_sql, params) as cur:
                total = (await cur.fetchone())[0]
            async with db.execute(list_sql, [*params, limit, offset]) as cur:
                rows = [_row_to_dict(r) for r in await cur.fetchall()]
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(rows) < total,
            "results": rows,
        }

    @router.get("/semantic/{memory_key}")
    async def get_semantic_memory(
        memory_key: str,
        api_key: str = Depends(verify_api_key),
    ):
        """Return one semantic memory by its exact key."""
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM semantic_memories WHERE key = ?", (memory_key,)
            ) as cur:
                row = await cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return _row_to_dict(row)

    @router.delete("/semantic/{memory_key}")
    async def delete_semantic_memory(
        memory_key: str,
        agent: str = Query(
            ...,
            description="agent performing the delete (must match key's owner)",
        ),
        api_key: str = Depends(verify_api_key),
    ):
        """Physically delete a semantic memory by key.

        Owner-gated: if the row has an established `owner`, only that agent may
        delete it. FTS5 rows are cleaned by the AFTER DELETE trigger; the vector
        embedding is removed on a best-effort basis.
        """
        if VALID_AGENTS is not None and agent not in VALID_AGENTS:
            raise HTTPException(status_code=400, detail="invalid agent")
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT id, owner FROM semantic_memories WHERE key = ?",
                (memory_key,),
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="memory not found")
            if row["owner"] and row["owner"] != agent:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"key '{memory_key}' is owned by '{row['owner']}'. "
                        f"agent '{agent}' cannot delete it."
                    ),
                )
            try:
                await load_vec_extension(db)
                await delete_embedding(db, row["id"])
            except Exception:
                pass
            await db.execute(
                "DELETE FROM semantic_memories WHERE key = ?", (memory_key,)
            )
            await db.commit()
        return {"deleted": True, "key": memory_key, "id": row["id"]}

    @router.post("/semantic")
    async def upsert_semantic(
        payload: SemanticMemoryIn, api_key: str = Depends(verify_api_key)
    ):
        value_stripped = (payload.value or "").strip()
        thin_write = 0 < len(value_stripped) < THIN_VALUE_MIN_LEN
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            # Owner gate: an established owner blocks any other agent from
            # touching this key. Rows pre-migration may have NULL owner — in
            # that case the first writer claims it (COALESCE below).
            async with db.execute(
                "SELECT owner FROM semantic_memories WHERE key = ?",
                (payload.key,),
            ) as cur:
                existing = await cur.fetchone()
            if existing and existing["owner"] and existing["owner"] != payload.agent:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"key '{payload.key}' is owned by "
                        f"'{existing['owner']}'. agent '{payload.agent}' "
                        f"cannot overwrite it."
                    ),
                )
            await db.execute(
                """
                INSERT INTO semantic_memories (key, value, category, agent, owner)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    agent = excluded.agent,
                    owner = COALESCE(semantic_memories.owner, excluded.owner),
                    updated_at = datetime('now')
                """,
                (
                    payload.key,
                    payload.value,
                    payload.category,
                    payload.agent,
                    payload.agent,
                ),
            )
            await db.commit()
            async with db.execute(
                "SELECT * FROM semantic_memories WHERE key = ?", (payload.key,)
            ) as cur:
                row = await cur.fetchone()
            try:
                await load_vec_extension(db)
                await upsert_embedding(db, row["id"], row["key"], row["value"])
                await db.commit()
            except Exception:
                pass
        result = _row_to_dict(row)
        if thin_write:
            result["warning"] = "thin_write"
            result["warning_detail"] = (
                f"value is {len(value_stripped)} chars "
                f"(<{THIN_VALUE_MIN_LEN}). consider expanding."
            )
        return result

    @router.post("/procedural")
    async def add_procedural(
        payload: ProceduralMemoryIn, api_key: str = Depends(verify_api_key)
    ):
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                INSERT INTO procedural_memories (agent, rule, source)
                VALUES (?, ?, ?)
                """,
                (payload.agent, payload.rule, payload.source),
            )
            await db.commit()
            new_id = cur.lastrowid
            async with db.execute(
                "SELECT * FROM procedural_memories WHERE id = ?", (new_id,)
            ) as q:
                row = await q.fetchone()
        return _row_to_dict(row)

    @router.post("/episodic")
    async def add_episodic(
        payload: EpisodicMemoryIn, api_key: str = Depends(verify_api_key)
    ):
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                INSERT INTO episodic_memories (agent, summary, topics, session_date)
                VALUES (?, ?, ?, ?)
                """,
                (
                    payload.agent,
                    payload.summary,
                    payload.topics,
                    payload.session_date,
                ),
            )
            await db.commit()
            new_id = cur.lastrowid
            async with db.execute(
                "SELECT * FROM episodic_memories WHERE id = ?", (new_id,)
            ) as q:
                row = await q.fetchone()
        return _row_to_dict(row)

    @router.post("/pending")
    async def add_pending(
        payload: PendingDecisionIn, api_key: str = Depends(verify_api_key)
    ):
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                INSERT INTO pending_decisions
                    (key, value, category, agent, source,
                     scope, source_reference)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.key, payload.value, payload.category,
                    payload.agent, payload.source,
                    payload.scope, payload.source_reference,
                ),
            )
            await db.commit()
            new_id = cur.lastrowid
            async with db.execute(
                "SELECT * FROM pending_decisions WHERE id = ?", (new_id,)
            ) as q:
                row = await q.fetchone()
        return _row_to_dict(row)

    @router.patch("/pending/{pending_id}")
    async def patch_pending(
        pending_id: int,
        payload: PendingActionIn,
        api_key: str = Depends(verify_api_key),
    ):
        if payload.action not in ("approve", "reject"):
            raise HTTPException(
                status_code=400,
                detail="action must be 'approve' or 'reject'",
            )
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM pending_decisions WHERE id=?", (pending_id,)
            ) as cur:
                row = await cur.fetchone()
            if not row:
                raise HTTPException(
                    status_code=404, detail="pending decision not found"
                )
            if row["status"] != "pending":
                raise HTTPException(
                    status_code=409,
                    detail=f"already {row['status']}",
                )

            if payload.action == "reject":
                await db.execute(
                    "UPDATE pending_decisions SET status='rejected' WHERE id=?",
                    (pending_id,),
                )
                await db.commit()
                return {"id": pending_id, "status": "rejected"}

            # Owner gate also applies on promotion from pending_decisions:
            # an approved pending entry from agent X cannot overwrite a key
            # already owned by agent Y.
            async with db.execute(
                "SELECT owner FROM semantic_memories WHERE key = ?",
                (row["key"],),
            ) as cur:
                existing_sem = await cur.fetchone()
            if (
                existing_sem
                and existing_sem["owner"]
                and existing_sem["owner"] != row["agent"]
            ):
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"key '{row['key']}' is owned by "
                        f"'{existing_sem['owner']}'. pending decision from "
                        f"'{row['agent']}' cannot be approved."
                    ),
                )
            # approve → promote to semantic_memories with embedding.
            # Phase 1: also carry source / source_reference / scope from the
            # pending row so provenance and visibility survive promotion.
            await db.execute(
                """
                INSERT INTO semantic_memories
                    (key, value, category, agent, owner,
                     source, source_reference, scope)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    agent = excluded.agent,
                    owner = COALESCE(semantic_memories.owner, excluded.owner),
                    source = excluded.source,
                    source_reference = excluded.source_reference,
                    scope = excluded.scope,
                    updated_at = datetime('now')
                """,
                (
                    row["key"], row["value"], row["category"],
                    row["agent"], row["agent"],
                    row["source"], row["source_reference"], row["scope"],
                ),
            )
            async with db.execute(
                "SELECT id FROM semantic_memories WHERE key = ?",
                (row["key"],),
            ) as r:
                sem = await r.fetchone()
            try:
                await load_vec_extension(db)
                await upsert_embedding(
                    db, sem["id"], row["key"], row["value"]
                )
            except Exception:
                pass
            await db.execute(
                "UPDATE pending_decisions SET status='approved' WHERE id=?",
                (pending_id,),
            )
            await db.commit()
        return {
            "id": pending_id,
            "status": "approved",
            "semantic_id": sem["id"],
        }

    @router.get("/pending")
    async def list_pending(
        status: str = Query("pending", pattern="^(pending|approved|rejected|all)$"),
        source: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        api_key: str = Depends(verify_api_key),
    ):
        # Phase 1: support optional ?source= filter (e.g. 'codex') so callers can
        # scan for candidates from a specific origin. Combined with the existing
        # status filter via SQL AND. Both conditions use parameterised binds.
        conditions: list[str] = []
        params: list = []
        if status != "all":
            conditions.append("status = ?")
            params.append(status)
        if source is not None and source.strip():
            conditions.append("source = ?")
            params.append(source.strip())
        sql = "SELECT * FROM pending_decisions"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql, params) as cur:
                rows = [_row_to_dict(r) for r in await cur.fetchall()]
        return {"status": status, "source": source, "results": rows}

    @router.get("/search")
    async def search_semantic(
        q: str = Query(..., min_length=1),
        agent: Optional[str] = Query(None),
        limit: int = Query(20, ge=1, le=100),
        mode: str = Query("hybrid", pattern="^(fts|vec|hybrid)$"),
        api_key: str = Depends(verify_api_key),
    ):
        fts_results: list[dict] = []
        vec_results: list[dict] = []

        if mode in ("fts", "hybrid"):
            fts_q = _quote_fts(q)
            sql = (
                "SELECT s.id, s.key, s.value, s.category, s.agent, "
                "       s.created_at, s.updated_at, "
                "       bm25(memories_fts) AS score "
                "FROM memories_fts "
                "JOIN semantic_memories s ON s.id = memories_fts.rowid "
                "WHERE memories_fts MATCH ?"
            )
            params: list = [fts_q]
            if agent:
                sql += " AND s.agent = ?"
                params.append(agent)
            sql += " ORDER BY score LIMIT ?"
            params.append(limit)
            async with aiosqlite.connect(DB_PATH) as db:
                db.row_factory = aiosqlite.Row
                try:
                    async with db.execute(sql, params) as cur:
                        fts_results = [
                            _row_to_dict(r) for r in await cur.fetchall()
                        ]
                except aiosqlite.OperationalError as e:
                    if mode == "fts":
                        raise HTTPException(
                            status_code=400, detail=f"fts search failed: {e}"
                        )

        if mode in ("vec", "hybrid"):
            async with aiosqlite.connect(DB_PATH) as db:
                try:
                    await load_vec_extension(db)
                    vec_results = await search_vec(
                        db, q, limit=limit, agent=agent
                    )
                except Exception as e:
                    if mode == "vec":
                        raise HTTPException(
                            status_code=400, detail=f"vec search failed: {e}"
                        )

        if mode == "fts":
            return {"query": q, "mode": mode, "results": fts_results}
        if mode == "vec":
            return {"query": q, "mode": mode, "results": vec_results}
        merged = _rrf([fts_results, vec_results])[:limit]
        return {"query": q, "mode": mode, "results": merged}

    @router.get("/agent/{agent_name}")
    async def get_agent_rules(
        agent_name: str, api_key: str = Depends(verify_api_key)
    ):
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM procedural_memories WHERE agent = ? "
                "ORDER BY created_at",
                (agent_name,),
            ) as cur:
                rules = [_row_to_dict(r) for r in await cur.fetchall()]
        return {"agent": agent_name, "rules": rules}

    @router.get("/context")
    async def get_context(
        agent: str = Query(...), api_key: str = Depends(verify_api_key)
    ):
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            # agent-scoped: rows owned by or written by this agent (latest 40)
            async with db.execute(
                "SELECT * FROM semantic_memories "
                "WHERE agent = ? OR owner = ? "
                "ORDER BY updated_at DESC LIMIT 40",
                (agent, agent),
            ) as cur:
                agent_rows = [_row_to_dict(r) for r in await cur.fetchall()]
            # global: shared facts every agent should see (latest 20)
            async with db.execute(
                "SELECT * FROM semantic_memories "
                "WHERE scope = 'global' "
                "ORDER BY updated_at DESC LIMIT 20"
            ) as cur:
                global_rows = [_row_to_dict(r) for r in await cur.fetchall()]
            # agent-scoped first, then de-duplicated globals
            seen_ids: set[int] = set()
            semantic: list[dict] = []
            for row in agent_rows + global_rows:
                if row["id"] in seen_ids:
                    continue
                seen_ids.add(row["id"])
                semantic.append(row)
            async with db.execute(
                "SELECT * FROM procedural_memories WHERE agent = ? "
                "ORDER BY created_at",
                (agent,),
            ) as cur:
                procedural = [_row_to_dict(r) for r in await cur.fetchall()]
            async with db.execute(
                "SELECT * FROM episodic_memories WHERE agent = ? "
                "ORDER BY created_at DESC LIMIT 5",
                (agent,),
            ) as cur:
                episodic = [_row_to_dict(r) for r in await cur.fetchall()]
        return {
            "agent": agent,
            "semantic": semantic,
            "procedural": procedural,
            "episodic": episodic,
        }

    return router
