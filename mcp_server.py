"""MCP server exposing agent-memory-server (AMS) tools.

Runs on port 8001 (separate from AMS on 8000). Exposes MCP tools:
  - search_memory(query, mode, agent)
  - list_memories(agent, category, scope, source, limit, offset)
  - get_memory(key)
  - get_context(agent)
  - save_candidate(key, value, category, agent, source)
  - save_memory(key, value, category, agent)
  - inbox_add / inbox_list / inbox_resolve
  - save_codex_candidate(key, value, category, target_agent, evidence)  [Phase 1: Codex]
  - list_pending_candidates(source, limit)                              [Phase 1: read-only]

Authentication (both via Authorization: Bearer):
  - OAuth 2.1 JWT access token (for claude.ai web, issued by /authorize + /token)
  - Raw AMS_API_KEY as bearer (for CLI/Desktop one-shot use)
The /authorize endpoint redirects straight back to redirect_uri with ?code=&state=
(no consent / password prompt — single-user server).

Public access is via Tailscale Funnel path /mcp -> http://127.0.0.1:8001.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import sqlite3
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import jwt
import uvicorn
from dotenv import load_dotenv
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
)
from mcp.server.auth.settings import (
    AuthSettings,
    ClientRegistrationOptions,
)
from mcp.server.fastmcp import FastMCP
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyHttpUrl, AnyUrl
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

load_dotenv()

AMS_URL = os.environ.get("AMS_URL", "http://localhost:8000")
AMS_KEY = os.environ.get("API_KEY", "")
ISSUER_URL = os.environ.get(
    "MCP_ISSUER_URL", "http://localhost:8001"
)
ADMIN_PASS = os.environ.get("MCP_ADMIN_PASS", "")
JWT_SECRET = os.environ.get("MCP_JWT_SECRET", "")
PORT = int(os.environ.get("MCP_PORT", "8001"))

if not (AMS_KEY and ADMIN_PASS and JWT_SECRET):
    raise SystemExit(
        "Required env vars missing: API_KEY, MCP_ADMIN_PASS, MCP_JWT_SECRET. "
        "Check ~/Projects/agent-memory-server/.env"
    )

DB_PATH = Path(__file__).resolve().parent / "mcp_oauth.db"
JWT_AUD = "ams-mcp"
ACCESS_TOKEN_TTL = 3600
REFRESH_TOKEN_TTL = 30 * 86400
AUTH_CODE_TTL = 600
CONSENT_SESSION_TTL = 900  # 15 minutes for the user to enter password
DEFAULT_SCOPE = "mcp:full"


def _now() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())


def _init_db() -> None:
    con = sqlite3.connect(DB_PATH)
    # Lightweight migration: add expires_at if the table existed before
    try:
        con.execute(
            "ALTER TABLE oauth_sessions ADD COLUMN expires_at INTEGER NOT NULL DEFAULT 0"
        )
    except sqlite3.OperationalError:
        pass
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS oauth_clients (
            client_id TEXT PRIMARY KEY,
            client_data TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS oauth_sessions (
            session TEXT PRIMARY KEY,
            client_id TEXT NOT NULL,
            params TEXT NOT NULL,
            expires_at INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS oauth_codes (
            code TEXT PRIMARY KEY,
            client_id TEXT NOT NULL,
            redirect_uri TEXT NOT NULL,
            code_challenge TEXT NOT NULL,
            scopes TEXT NOT NULL,
            resource TEXT,
            expires_at INTEGER NOT NULL,
            used INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS oauth_refresh_tokens (
            token TEXT PRIMARY KEY,
            client_id TEXT NOT NULL,
            scopes TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            revoked INTEGER DEFAULT 0
        );
        """
    )
    con.commit()
    con.close()


_init_db()


# ---------------------------------------------------------------------------
# OAuth provider — sqlite-backed, single-user (password from MCP_ADMIN_PASS)
# ---------------------------------------------------------------------------


class SqliteOAuthProvider(OAuthAuthorizationServerProvider):
    async def register_client(
        self, client_info: OAuthClientInformationFull
    ) -> None:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT OR REPLACE INTO oauth_clients(client_id, client_data) "
            "VALUES (?, ?)",
            (client_info.client_id, client_info.model_dump_json()),
        )
        con.commit()
        con.close()

    async def get_client(
        self, client_id: str
    ) -> Optional[OAuthClientInformationFull]:
        con = sqlite3.connect(DB_PATH)
        row = con.execute(
            "SELECT client_data FROM oauth_clients WHERE client_id=?",
            (client_id,),
        ).fetchone()
        con.close()
        if not row:
            return None
        return OAuthClientInformationFull.model_validate_json(row[0])

    async def authorize(
        self,
        client: OAuthClientInformationFull,
        params: AuthorizationParams,
    ) -> str:
        # Single-user server: skip the /consent password gate and mint the
        # authorization code immediately, redirecting the browser straight
        # back to the client's redirect_uri with ?code=&state=.
        code = secrets.token_urlsafe(32)
        redirect_uri = str(params.redirect_uri)
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO oauth_codes"
            "(code, client_id, redirect_uri, code_challenge, scopes, resource, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                code,
                client.client_id,
                redirect_uri,
                params.code_challenge,
                json.dumps(params.scopes or []),
                params.resource,
                _now() + AUTH_CODE_TTL,
            ),
        )
        con.commit()
        con.close()
        query = {"code": code}
        if params.state:
            query["state"] = params.state
        sep = "&" if "?" in redirect_uri else "?"
        return f"{redirect_uri}{sep}{urllib.parse.urlencode(query)}"

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> Optional[AuthorizationCode]:
        con = sqlite3.connect(DB_PATH)
        row = con.execute(
            "SELECT client_id, redirect_uri, code_challenge, scopes, "
            "       resource, expires_at, used "
            "FROM oauth_codes WHERE code=?",
            (authorization_code,),
        ).fetchone()
        con.close()
        if not row:
            return None
        client_id, redirect_uri, code_challenge, scopes_json, resource, exp, used = row
        if used or exp < _now() or client_id != client.client_id:
            return None
        return AuthorizationCode(
            code=authorization_code,
            scopes=json.loads(scopes_json),
            expires_at=float(exp),
            client_id=client_id,
            code_challenge=code_challenge,
            redirect_uri=AnyUrl(redirect_uri),
            redirect_uri_provided_explicitly=True,
            resource=resource,
        )

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "UPDATE oauth_codes SET used=1 WHERE code=?",
            (authorization_code.code,),
        )
        now = _now()
        access_token = self._mint_access_token(
            client.client_id, authorization_code.scopes, now
        )
        refresh_token = secrets.token_urlsafe(48)
        con.execute(
            "INSERT INTO oauth_refresh_tokens"
            "(token, client_id, scopes, expires_at) VALUES (?, ?, ?, ?)",
            (
                refresh_token,
                client.client_id,
                json.dumps(authorization_code.scopes),
                now + REFRESH_TOKEN_TTL,
            ),
        )
        con.commit()
        con.close()
        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL,
            refresh_token=refresh_token,
            scope=" ".join(authorization_code.scopes),
        )

    async def load_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: str,
    ) -> Optional[RefreshToken]:
        con = sqlite3.connect(DB_PATH)
        row = con.execute(
            "SELECT client_id, scopes, expires_at, revoked "
            "FROM oauth_refresh_tokens WHERE token=?",
            (refresh_token,),
        ).fetchone()
        con.close()
        if not row:
            return None
        client_id, scopes_json, exp, revoked = row
        if revoked or exp < _now() or client_id != client.client_id:
            return None
        return RefreshToken(
            token=refresh_token,
            client_id=client_id,
            scopes=json.loads(scopes_json),
            expires_at=exp,
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        now = _now()
        access_token = self._mint_access_token(client.client_id, scopes, now)
        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL,
            refresh_token=refresh_token.token,
            scope=" ".join(scopes),
        )

    async def load_access_token(self, token: str) -> Optional[AccessToken]:
        if token == AMS_KEY:
            return AccessToken(
                token=token,
                client_id="ams-api-key",
                scopes=["mcp:full"],
                expires_at=_now() + ACCESS_TOKEN_TTL,
            )
        try:
            payload = jwt.decode(
                token,
                JWT_SECRET,
                algorithms=["HS256"],
                audience=JWT_AUD,
            )
        except jwt.InvalidTokenError:
            return None
        return AccessToken(
            token=token,
            client_id=payload["sub"],
            scopes=(payload.get("scope") or "").split(),
            expires_at=int(payload["exp"]),
        )

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "UPDATE oauth_refresh_tokens SET revoked=1 WHERE token=?",
            (token.token,),
        )
        con.commit()
        con.close()

    def _mint_access_token(
        self, client_id: str, scopes: list[str], now: int
    ) -> str:
        payload = {
            "iss": ISSUER_URL,
            "sub": client_id,
            "aud": JWT_AUD,
            "exp": now + ACCESS_TOKEN_TTL,
            "iat": now,
            "scope": " ".join(scopes),
        }
        return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


# ---------------------------------------------------------------------------
# Consent UI — minimal password gate
# ---------------------------------------------------------------------------


CONSENT_HTML = """<!doctype html>
<html lang="ja"><head>
<meta charset="utf-8"><title>AMS MCP authorization</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{font-family:-apple-system,system-ui,sans-serif;max-width:420px;margin:80px auto;padding:0 20px;color:#222}}
h1{{font-size:1.3rem;margin-bottom:4px}}
small{{color:#666}}
form{{margin-top:24px}}
input[type=password]{{width:100%;padding:10px;font-size:1rem;border:1px solid #ccc;border-radius:6px;box-sizing:border-box}}
button{{padding:10px 18px;font-size:1rem;border-radius:6px;border:0;cursor:pointer;margin-right:8px;margin-top:12px}}
button[value=approve]{{background:#1f883d;color:#fff}}
button[value=reject]{{background:#eee;color:#333}}
</style></head><body>
<h1>Authorize <code>{client}</code>?</h1>
<small>このアプリにagent-memory-serverへのアクセスを許可するか確認します。</small>
<form method="post" action="/consent">
  <input type="hidden" name="session" value="{session}">
  <label>Admin password:<br><input type="password" name="password" autofocus required></label>
  <div>
    <button type="submit" name="action" value="approve">Approve</button>
    <button type="submit" name="action" value="reject">Reject</button>
  </div>
</form>
</body></html>
"""


async def consent_get(request: Request) -> HTMLResponse:
    session = request.query_params.get("session", "")
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT client_id, expires_at FROM oauth_sessions WHERE session=?",
        (session,),
    ).fetchone()
    con.close()
    if not row:
        return HTMLResponse(
            "Authorization session not found. Restart the connection from the "
            "client (Claude.ai etc.) — sessions are one-shot and must not be "
            "bookmarked or refreshed.",
            status_code=400,
        )
    client_id, expires_at = row
    if expires_at and expires_at < _now():
        return HTMLResponse(
            "Authorization session expired (15 min limit). Restart the "
            "connection from the client.",
            status_code=400,
        )
    return HTMLResponse(
        CONSENT_HTML.format(session=session, client=client_id)
    )


async def consent_post(request: Request):
    form = await request.form()
    session = (form.get("session") or "").strip()
    password = form.get("password") or ""
    action = form.get("action") or ""

    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT client_id, params, expires_at FROM oauth_sessions WHERE session=?",
        (session,),
    ).fetchone()
    if not row:
        con.close()
        return HTMLResponse(
            "Authorization session not found (it may have already been used). "
            "Restart the connection from the client.",
            status_code=400,
        )
    client_id, params_json, expires_at = row
    if expires_at and expires_at < _now():
        con.execute("DELETE FROM oauth_sessions WHERE session=?", (session,))
        con.commit()
        con.close()
        return HTMLResponse(
            "Authorization session expired (15 min limit). Restart the "
            "connection from the client.",
            status_code=400,
        )
    params = AuthorizationParams.model_validate_json(params_json)
    redirect_uri = str(params.redirect_uri)
    state = params.state or ""

    def _redirect_with(query: dict[str, str]) -> RedirectResponse:
        sep = "&" if "?" in redirect_uri else "?"
        return RedirectResponse(
            f"{redirect_uri}{sep}{urllib.parse.urlencode(query)}",
            status_code=302,
        )

    if action != "approve" or password != ADMIN_PASS:
        con.execute("DELETE FROM oauth_sessions WHERE session=?", (session,))
        con.commit()
        con.close()
        return _redirect_with({"error": "access_denied", "state": state})

    code = secrets.token_urlsafe(32)
    con.execute(
        "INSERT INTO oauth_codes"
        "(code, client_id, redirect_uri, code_challenge, scopes, resource, expires_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            code,
            client_id,
            redirect_uri,
            params.code_challenge,
            json.dumps(params.scopes or []),
            params.resource,
            _now() + AUTH_CODE_TTL,
        ),
    )
    con.execute("DELETE FROM oauth_sessions WHERE session=?", (session,))
    con.commit()
    con.close()
    return _redirect_with({"code": code, "state": state})


# ---------------------------------------------------------------------------
# AMS HTTP client (used by the MCP tools)
# ---------------------------------------------------------------------------


def _ams_get(path: str, params: dict) -> dict:
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"{AMS_URL}{path}?{qs}",
        headers={"X-API-Key": AMS_KEY},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _ams_post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{AMS_URL}{path}",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"X-API-Key": AMS_KEY, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _ams_patch(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{AMS_URL}{path}",
        data=json.dumps(body).encode("utf-8"),
        method="PATCH",
        headers={"X-API-Key": AMS_KEY, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _ams_delete(path: str, params: dict) -> dict:
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"{AMS_URL}{path}?{qs}",
        method="DELETE",
        headers={"X-API-Key": AMS_KEY},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _refresh_agent_context_to_project(agent: str) -> None:
    """Regenerate <agent>_context.md and push it to the agent's Claude project.

    Best-effort. Any failure is logged but does not propagate — the caller's
    DB write already succeeded and shouldn't be reverted just because the
    outbound sync couldn't reach claude.ai.
    """
    try:
        from main import (
            AGENT_CONTEXT_REMOTE_NAME,
            _regenerate_agent_context_md,
        )
        from session_key import get_session_key
        from sync_status_to_project import AGENT_PROJECTS
        from upload_to_project import upload_file_to_project
    except Exception as e:  # noqa: BLE001
        print(f"[context_hook] import failed: {e}", flush=True)
        return

    project_id = AGENT_PROJECTS.get(agent)
    if not project_id:
        print(
            f"[context_hook] {agent} not in AGENT_PROJECTS; skip",
            flush=True,
        )
        return

    session_key = get_session_key()
    if not session_key:
        print(
            f"[context_hook] session key unavailable; skip {agent}",
            flush=True,
        )
        return

    try:
        context_path = asyncio.run(_regenerate_agent_context_md(agent))
    except Exception as e:  # noqa: BLE001
        print(f"[context_hook] regen failed for {agent}: {e}", flush=True)
        return

    try:
        upload_file_to_project(
            project_id=project_id,
            file_path=context_path,
            session_key=session_key,
            remote_name=AGENT_CONTEXT_REMOTE_NAME,
            verbose=False,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[context_hook] upload failed for {agent}: {e}", flush=True)
        return
    print(
        f"[context_hook] {agent} agent_context.md refreshed & uploaded",
        flush=True,
    )


# ---------------------------------------------------------------------------
# FastMCP server and tools
# ---------------------------------------------------------------------------


provider = SqliteOAuthProvider()

mcp = FastMCP(
    name="agent-memory-server",
    host="0.0.0.0",
    port=PORT,
    auth_server_provider=provider,
    auth=AuthSettings(
        issuer_url=AnyHttpUrl(ISSUER_URL),
        resource_server_url=AnyHttpUrl(f"{ISSUER_URL}/mcp"),
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=[DEFAULT_SCOPE],
            default_scopes=[DEFAULT_SCOPE],
        ),
    ),
)


@mcp.tool()
def search_memory(
    query: str, mode: str = "hybrid", agent: Optional[str] = None
) -> dict:
    """Search AMS semantic_memories (Japanese OK).

    Args:
      query: natural-language query
      mode: 'hybrid' (default, RRF of vec + FTS5), 'vec', or 'fts'
      agent: optional agent name filter (operator / agent_a / agent_b / agent_c / agent_d / agent_e / agent_h / agent_f)
    """
    params: dict[str, Any] = {"q": query, "mode": mode, "limit": 20}
    if agent:
        params["agent"] = agent
    return _ams_get("/memory/search", params)


@mcp.tool()
def list_memories(
    agent: Optional[str] = None,
    category: Optional[str] = None,
    scope: Optional[str] = None,
    source: Optional[str] = None,
    stale: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """List the AMS semantic-memory catalogue, newest first.

    Results contain keys, metadata, and a short value preview rather than the
    complete value. Use get_memory(key) after choosing an entry.

    Args:
      agent: optional agent filter
      category: optional category filter
      scope: optional 'agent' or 'global' filter
      source: optional provenance filter such as 'codex'
      stale: True to return only self-reported stale rows (value contains
        【削除】/【解消済み】/【陳腐化】), False to exclude them, None for no filter
      limit: page size (1-200, default 50)
      offset: zero-based page offset
    """
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    for name, value in (
        ("agent", agent),
        ("category", category),
        ("scope", scope),
        ("source", source),
    ):
        if value is not None:
            params[name] = value
    if stale is not None:
        params["stale"] = "true" if stale else "false"
    return _ams_get("/memory/semantic", params)


@mcp.tool()
def get_memory(key: str) -> dict:
    """Get one semantic memory by exact key, including its full value."""
    encoded_key = urllib.parse.quote(key, safe="")
    return _ams_get(f"/memory/semantic/{encoded_key}", {})


@mcp.tool()
def delete_memory(key: str, agent: str) -> dict:
    """Physically delete one semantic memory by its exact key.

    Owner-gated: if the row has an established owner, only that agent may
    delete it. FTS5 is cleaned by the AFTER DELETE trigger; the vector
    embedding is removed on a best-effort basis.

    Args:
      key: exact semantic_memories.key to remove (PRIMARY KEY)
      agent: agent performing the delete (must match the row's owner if set)
    """
    encoded_key = urllib.parse.quote(key, safe="")
    return _ams_delete(f"/memory/semantic/{encoded_key}", {"agent": agent})


@mcp.tool()
def get_context(agent: str) -> dict:
    """Get the full memory context (semantic + procedural + episodic) for an agent.

    Args:
      agent: agent name (operator / agent_a / agent_b / agent_c / agent_d / agent_e / agent_h / agent_f)
    """
    return _ams_get("/memory/context", {"agent": agent})


@mcp.tool()
def save_candidate(
    key: str,
    value: str,
    category: str,
    agent: str,
    source: str = "mcp",
) -> dict:
    """Queue a save candidate for operator's approval (does NOT write to semantic_memories directly).

    Args:
      key: snake_case identifier
      value: concise description
      category: 'surface' | 'design_decision' | 'infra' | 'agent_policy'
      agent: agent name
      source: who proposed it (default 'mcp')
    """
    result = _ams_post(
        "/memory/pending",
        {
            "key": key,
            "value": value,
            "category": category,
            "agent": agent,
            "source": source,
        },
    )
    _refresh_agent_context_to_project(agent)
    return result


@mcp.tool()
def save_memory(
    key: str,
    value: str,
    category: str,
    agent: str,
) -> dict:
    """Write directly to semantic_memories (no approval flow).

    Upserts on `key` (PRIMARY KEY): if the key exists it is updated, otherwise inserted.
    Embeddings are refreshed automatically.

    Args:
      key: snake_case identifier (PRIMARY KEY in semantic_memories)
      value: concise description (Japanese OK)
      category: 'surface' | 'design_decision' | 'infra' | 'agent_policy'
      agent: agent name (operator / agent_a / agent_b / agent_c / agent_d / agent_e / agent_h / agent_f)
    """
    result = _ams_post(
        "/memory/semantic",
        {
            "key": key,
            "value": value,
            "category": category,
            "agent": agent,
        },
    )
    _refresh_agent_context_to_project(agent)
    return result


@mcp.tool()
def inbox_add(content: str) -> dict:
    """Add a raw item to the AMS inbox.

    No classification, no key, no category — just save whatever the user wants
    captured. Use this when the user says『これメモしといて』in mid-conversation.
    """
    return _ams_post(
        "/api/inbox",
        {"content": content, "source": "mcp"},
    )


@mcp.tool()
def inbox_list(status: str = "unprocessed", limit: int = 50) -> dict:
    """List inbox items.

    Args:
      status: 'unprocessed' (default) | 'promoted' | 'discarded' | 'all'
      limit:  max rows (1-500)
    """
    return _ams_get(
        "/api/inbox",
        {"status": status, "limit": limit},
    )


@mcp.tool()
def inbox_resolve(
    inbox_id: int,
    action: str,
    promoted_to: Optional[int] = None,
) -> dict:
    """Mark an inbox item as processed.

    Args:
      inbox_id: target inbox row id
      action:   'promoted' (昇格済み) or 'discarded' (没)
      promoted_to: 昇格先 semantic_memories.id（action='promoted' のとき）
    """
    if action not in ("promoted", "discarded"):
        return {"ok": False, "error": "action must be 'promoted' or 'discarded'"}
    body: dict[str, Any] = {"status": action}
    if promoted_to is not None:
        body["promoted_to"] = promoted_to
    return _ams_patch(f"/api/inbox/{inbox_id}", body)


# ---------------------------------------------------------------------------
# Phase 1: Codex integration — write a Codex-proposed candidate to
# pending_decisions with source='codex' and scope='global' forced server-side.
# The auto-approver (pending_approver.py) skips source='codex' rows, so these
# candidates stay pending until the user manually approves via HTTP PATCH from the
# daily Slack stale-notification. Once approved, the promoted semantic row
# retains source='codex' and scope='global', making it visible to all 9 agents
# through /memory/context.
#
# NOTE: an approve_candidate MCP tool is intentionally NOT added in Phase 1 —
# the same MCP connection is also used by Codex, so exposing an approval tool
# would defeat the "manual review required" property. Approvals stay on the
# existing curl-PATCH surface until Phase 2 introduces a separate MCP endpoint
# with approver authentication.
# ---------------------------------------------------------------------------


@mcp.tool()
def save_codex_candidate(
    key: str,
    value: str,
    category: str,
    target_agent: str,
    evidence: Optional[str] = None,
) -> dict:
    """Codex proposes a shared fact for every agent to see.

    The candidate is written to pending_decisions with:
      - source='codex'       (server-side forced, ignored if caller sends it)
      - scope='global'       (server-side forced, ignored if caller sends it)

    Codex candidates are excluded from auto-approval. The user reviews them via
    the daily Slack stale-notification and approves/rejects with:
        curl -X PATCH -H "X-API-Key: $AMS_API_KEY" \\
             -H 'Content-Type: application/json' \\
             -d '{"action":"approve"}' \\
             http://localhost:8000/memory/pending/<id>

    Upon approval, the promoted semantic_memories row keeps source='codex' and
    scope='global', so every agent's /memory/context returns it.

    Args:
      key:          snake_case identifier
      value:        concise description (Japanese OK)
      category:     'surface' | 'design_decision' | 'infra' | 'agent_policy'
                    (also 'app_status' | 'task' | 'study' if applicable)
      target_agent: agent expected to become responsible after approval.
                    Recorded as pending_decisions.agent and enforced by the
                    owner-gate during promotion. Recommended: 'operator'.
      evidence:     free-form justification / URL / citation. Stored as
                    source_reference (empty → NULL, max 2000 chars).
    """
    body: dict[str, Any] = {
        "key": key,
        "value": value,
        "category": category,
        "agent": target_agent,
        "source": "codex",       # forced — do not accept caller override
        "scope": "global",       # forced — do not accept caller override
    }
    if evidence is not None:
        body["source_reference"] = evidence
    return _ams_post("/memory/pending", body)


@mcp.tool()
def list_pending_candidates(
    source: Optional[str] = None,
    limit: int = 50,
) -> dict:
    """List pending candidates awaiting manual approval.

    Read-only. Useful for spotting Codex proposals or reviewing what the
    auto-approver has deferred to human review.

    Args:
      source: optional filter (e.g. 'codex' to see only Codex proposals,
              'mcp' for the standard save_candidate path)
      limit:  max rows (1-500)
    """
    params: dict[str, Any] = {"status": "pending", "limit": limit}
    if source:
        params["source"] = source
    return _ams_get("/memory/pending", params)


STUDY_MAP_SUBJECTS = ("確率統計", "AIプログラミング", "Python", "UNIX")

STUDY_MAP_CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, X-API-Key",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
}


def _resolve_subject_short(subject: str) -> Optional[str]:
    for short in STUDY_MAP_SUBJECTS:
        if short in subject:
            return short
    return None


async def study_map_options(request: Request) -> Response:
    return Response(status_code=204, headers=STUDY_MAP_CORS)


def _persist_study_map(subject, session, data) -> tuple[bool, dict, int]:
    """Validate + upsert into semantic_memories. Returns (ok, body, status_code)."""
    if not (isinstance(subject, str) and isinstance(session, int) and isinstance(data, dict)):
        return False, {
            "ok": False,
            "error": "missing or invalid fields (subject:str, session:int, data:object)",
        }, 400
    subject_short = _resolve_subject_short(subject)
    if subject_short is None:
        return False, {"ok": False, "error": f"unknown subject: {subject}"}, 400
    key = f"cu_study_{subject_short}_{session:02d}"
    value = json.dumps(data, ensure_ascii=False)
    try:
        _ams_post(
            "/memory/semantic",
            {"key": key, "value": value, "category": "study", "agent": "agent_d"},
        )
    except Exception as e:  # noqa: BLE001
        return False, {"ok": False, "error": f"ams upsert failed: {e}"}, 502
    return True, {"ok": True}, 200


async def study_map_post(request: Request) -> JSONResponse:
    api_key = request.headers.get("X-API-Key", "")
    if not api_key or api_key != AMS_KEY:
        return JSONResponse(
            {"ok": False, "error": "unauthorized"},
            status_code=401, headers=STUDY_MAP_CORS,
        )
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"ok": False, "error": "invalid json"},
            status_code=400, headers=STUDY_MAP_CORS,
        )
    _, resp_body, status = _persist_study_map(
        body.get("subject"), body.get("session"), body.get("data")
    )
    return JSONResponse(resp_body, status_code=status, headers=STUDY_MAP_CORS)


async def study_map_generate_post(request: Request) -> JSONResponse:
    api_key = request.headers.get("X-API-Key", "")
    if not api_key or api_key != AMS_KEY:
        return JSONResponse(
            {"ok": False, "error": "unauthorized"},
            status_code=401, headers=STUDY_MAP_CORS,
        )
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"ok": False, "error": "invalid json"},
            status_code=400, headers=STUDY_MAP_CORS,
        )
    subject = body.get("subject")
    session = body.get("session")
    text = body.get("text")
    if not (isinstance(subject, str) and isinstance(session, int) and isinstance(text, str)):
        return JSONResponse(
            {"ok": False, "error": "missing or invalid fields (subject:str, session:int, text:str)"},
            status_code=400, headers=STUDY_MAP_CORS,
        )
    if _resolve_subject_short(subject) is None:
        return JSONResponse(
            {"ok": False, "error": f"unknown subject: {subject}"},
            status_code=400, headers=STUDY_MAP_CORS,
        )

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        return JSONResponse(
            {"ok": False, "error": "ANTHROPIC_API_KEY not set"},
            status_code=500, headers=STUDY_MAP_CORS,
        )

    system_prompt = (
        "あなたはREDACTEDの学習コーチです。\n"
        "ユーザーが貼り付けたNotebookLMのまとめから、以下のJSON形式のみを返してください（余分なテキスト不要）:\n"
        '{"title":"...","summary":"...","keywords":[...],"problems":[{"question":"...","answer":"...","explanation":"..."}]}\n'
        f"科目：{subject} 第{session}回\n"
        "問題例は2〜3問、試験に出そうな具体的な問題にしてください。"
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        msg = await asyncio.to_thread(
            client.messages.create,
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": text}],
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"ok": False, "error": f"claude call failed: {e}"},
            status_code=502, headers=STUDY_MAP_CORS,
        )

    raw = "".join(getattr(b, "text", "") for b in msg.content).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"ok": False, "error": f"invalid JSON from Claude: {e}", "raw": raw[:500]},
            status_code=502, headers=STUDY_MAP_CORS,
        )
    if not isinstance(data, dict):
        return JSONResponse(
            {"ok": False, "error": "Claude returned non-object"},
            status_code=502, headers=STUDY_MAP_CORS,
        )

    ok, resp_body, status = _persist_study_map(subject, session, data)
    if not ok:
        return JSONResponse(resp_body, status_code=status, headers=STUDY_MAP_CORS)
    return JSONResponse({"ok": True, "data": data}, headers=STUDY_MAP_CORS)


def build_app():
    app = mcp.streamable_http_app()
    app.routes.append(Route("/consent", consent_get, methods=["GET"]))
    app.routes.append(Route("/consent", consent_post, methods=["POST"]))
    app.routes.append(Route("/api/study-map", study_map_post, methods=["POST"]))
    app.routes.append(Route("/api/study-map", study_map_options, methods=["OPTIONS"]))
    app.routes.append(Route("/api/study-map/generate", study_map_generate_post, methods=["POST"]))
    app.routes.append(Route("/api/study-map/generate", study_map_options, methods=["OPTIONS"]))
    return app


if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=PORT)
