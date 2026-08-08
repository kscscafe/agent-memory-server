from dotenv import load_dotenv
load_dotenv()
import base64
import hashlib
import hmac
import json
import os
import re
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from starlette.responses import JSONResponse, Response

import asyncio

from auto_executor import scan_and_execute
from claude_executor import chat_reply, parse_status_updates
from slack_client import send_slack_message
from inbox_api import build_router as build_inbox_router, save_inbox_item
from memory_api import VALID_AGENTS, build_router as build_memory_router


async def _trigger_immediate_scan() -> None:
    """Fire scan_and_execute as a background task so the webhook can return fast."""
    try:
        await scan_and_execute()
    except Exception as e:  # noqa: BLE001
        print(f"[webhook] immediate scan_and_execute failed: {e}", flush=True)


CHAT_KEYWORDS = (
    "ありがとう", "了解", "わかった", "OK", "ok", "おk", "なるほど",
    "確認した", "見た", "受け取った",
)
CHAT_PREFIXES = ("まだ", "どうやって", "なんで", "なぜ", "いつ")
CHAT_ENDINGS = ("？", "?", "か", "の", "よ")
CHAT_BARE = {"OK", "ok", "あとで", "キャンセル"}


TASK_UPDATE_PREFIXES = ("タスク完了", "タスク削除", "タスク終了", "完了 ", "削除 ")

def _is_task_update_message(text: str) -> bool:
    """Return True if the message is a task completion/deletion request."""
    return any(text.startswith(p) for p in TASK_UPDATE_PREFIXES)


def _is_chat_message(text: str) -> bool:
    """Return True if the LINE text should be answered as a chat reply rather
    than saved as an instruction. OK:<id> / LATER:<id> are control replies and
    branch earlier in the webhook, so they normally never reach here.
    Status-update messages (e.g. 「<app-name> 公開中」) also branch earlier via
    `_looks_like_status_update`, so they will not be mis-classified here."""
    if text.startswith("OK:") and text[3:].isdigit():
        return False
    if text.startswith("LATER:") and text[6:].isdigit():
        return False
    if text.strip() in CHAT_BARE:
        return True
    if len(text) <= 20:
        if text.endswith(CHAT_ENDINGS):
            return True
        if any(kw in text for kw in CHAT_KEYWORDS):
            return True
        if any(text.startswith(p) for p in CHAT_PREFIXES):
            return True
    return False


async def _handle_chat(text: str, line_msg_id: str) -> None:
    """Reply to a chat-style LINE message without creating an instruction row."""
    try:
        reply = await chat_reply(text, line_msg_id)
    except Exception as e:  # noqa: BLE001
        print(f"[webhook] chat_reply failed: {e}", flush=True)
        reply = "ERROR: 応答生成に失敗しました"
    await send_slack_message(reply)


MARKET_SCOUT_DB = Path.home() / "Projects" / "market-scout" / "market.db"
MARKET_KEYWORDS = (
    "market", "market-scout", "マーケット", "ニッチ",
    "市場調査", "市場分析", "収集結果",
)


def _is_market_scout_query(text: str) -> bool:
    lower = text.lower()
    return any(kw.lower() in lower for kw in MARKET_KEYWORDS)


def _fetch_market_results_sync() -> list[dict] | str:
    """Return rows (today first, 3-day fallback) or an error-message string."""
    if not MARKET_SCOUT_DB.exists():
        return "market-scoutが設定されていません"
    cols = (
        "rank, niche, demand_score, competition_score, "
        "buildability_score, total_score, comment"
    )
    try:
        conn = sqlite3.connect(str(MARKET_SCOUT_DB))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                f"SELECT {cols} FROM analysis_results "
                "WHERE date(analyzed_at) = date('now', 'localtime') "
                "ORDER BY rank ASC LIMIT 10"
            )
            rows = [dict(r) for r in cur.fetchall()]
            if not rows:
                cur = conn.execute(
                    f"SELECT {cols} FROM analysis_results "
                    "WHERE analyzed_at >= datetime('now', '-3 days') "
                    "ORDER BY analyzed_at DESC, rank ASC LIMIT 10"
                )
                rows = [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        return f"market.db 読み込み失敗: {e}"
    if not rows:
        return "今日の分析結果はまだありません"
    return rows


def _summarize_market_sync(rows: list[dict]) -> str:
    data_text = "\n".join(
        f"{r.get('rank','?')}. {r.get('niche','?')} | "
        f"demand:{r.get('demand_score')} comp:{r.get('competition_score')} "
        f"build:{r.get('buildability_score')} total:{r.get('total_score')} | "
        f"{r.get('comment','') or ''}"
        for r in rows
    )
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return data_text
    prompt = (
        "以下はmarket-scoutが収集・分析したニッチ市場データです。"
        "上位ニッチを簡潔にまとめてください。箇条書き、日本語、300字以内。\n"
        f"{data_text}"
    )
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(b, "text", "") for b in msg.content).strip()
    except Exception as e:  # noqa: BLE001
        print(f"[market-scout] claude call failed: {e}", flush=True)
        return data_text


async def _handle_market_scout() -> None:
    result = await asyncio.to_thread(_fetch_market_results_sync)
    if isinstance(result, str):
        await send_slack_message(result)
        return
    summary = await asyncio.to_thread(_summarize_market_sync, result)
    await send_slack_message(f"📈 market-scout 結果\n{summary}")


MEMORY_EXTRACT_SYSTEM = (
    "あなたは LINE 受信メッセージから長期記憶として保存すべき情報を抽出する分類器です。"
    "JSON のみを返してください。説明文や Markdown コードフェンスは禁止。"
)

MEMORY_EXTRACT_USER_TEMPLATE = """以下のメッセージから記憶すべき情報を抽出してください。
メッセージ：「{message}」

以下のJSON形式のみで返してください：
{{
  "semantic": [{{"key": "...", "value": "...", "category": "...", "agent": "..."}}],
  "procedural": [{{"agent": "...", "rule": "..."}}],
  "episodic": []
}}

抽出できない場合は各配列を空にしてください。
category は app_status / design_decision / agent_rule / task のいずれか。
key は英数小文字とアンダースコアのみで一意になる短い識別子（例: app_<name>_<platform>_status）。
agent は VALID_AGENTS 環境変数で列挙された名前のいずれか
(未設定なら任意の非空文字列)。判別できない場合は DEFAULT_AGENT を指定してください。"""


def _extract_memory_sync(message: str) -> dict | None:
    """Call Claude to classify the message into semantic/procedural/episodic items.
    Returns parsed dict or None on any failure."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        import anthropic  # local import to keep startup light
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=MEMORY_EXTRACT_SYSTEM,
            messages=[{
                "role": "user",
                "content": MEMORY_EXTRACT_USER_TEMPLATE.format(message=message),
            }],
        )
    except Exception as e:  # noqa: BLE001
        print(f"[memory-extract] claude call failed: {e}", flush=True)
        return None
    raw = "".join(getattr(b, "text", "") for b in msg.content).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"[memory-extract] non-JSON response: {raw[:200]}", flush=True)
        return None


async def auto_extract_memory(
    message: str, source: str = "line_webhook_auto"
) -> None:
    """Extract memory items from an inbound message and persist them.

    Runs the existing status-keyword detection only as a tag (the real status
    update path runs in the inbound webhook). Then asks Claude to classify
    additional semantic / procedural / episodic items and inserts them into
    the memory tables. `source` is recorded on procedural rows so LINE-legacy
    and Slack inputs can be distinguished post-hoc.
    """
    # Status keyword tag (existing flow handles the actual apps UPDATE).
    if any(kw in message for kw in STATUS_TRIGGER_KEYWORDS):
        print(f"[memory-extract] status-keyword detected in: {message[:60]!r}",
              flush=True)

    extracted = await asyncio.to_thread(_extract_memory_sync, message)
    if not extracted:
        return

    semantic = extracted.get("semantic") or []
    procedural = extracted.get("procedural") or []
    episodic = extracted.get("episodic") or []

    if not (semantic or procedural or episodic):
        return

    async with aiosqlite.connect(DB_PATH) as db:
        for s in semantic:
            if not isinstance(s, dict):
                continue
            key = (s.get("key") or "").strip()
            value = (s.get("value") or "").strip()
            category = (s.get("category") or "").strip()
            agent = (s.get("agent") or "").strip()
            if not (key and value and category):
                continue
            if VALID_AGENTS is not None and agent not in VALID_AGENTS:
                agent = DEFAULT_AGENT_NAME
            await db.execute(
                """
                INSERT INTO semantic_memories (key, value, category, agent)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    agent = excluded.agent,
                    updated_at = datetime('now')
                """,
                (key, value, category, agent),
            )
        for p in procedural:
            if not isinstance(p, dict):
                continue
            agent = (p.get("agent") or "").strip()
            rule = (p.get("rule") or "").strip()
            if not (agent and rule):
                continue
            if VALID_AGENTS is not None and agent not in VALID_AGENTS:
                agent = DEFAULT_AGENT_NAME
            await db.execute(
                """
                INSERT INTO procedural_memories (agent, rule, source)
                VALUES (?, ?, ?)
                """,
                (agent, rule, source),
            )
        for e in episodic:
            if not isinstance(e, dict):
                continue
            agent = (e.get("agent") or "").strip()
            summary = (e.get("summary") or "").strip()
            if not (agent and summary):
                continue
            if VALID_AGENTS is not None and agent not in VALID_AGENTS:
                agent = DEFAULT_AGENT_NAME
            await db.execute(
                """
                INSERT INTO episodic_memories (agent, summary, topics, session_date)
                VALUES (?, ?, ?, ?)
                """,
                (agent, summary, e.get("topics"), e.get("session_date")),
            )
        await db.commit()
    print(
        f"[memory-extract] saved semantic={len(semantic)} "
        f"procedural={len(procedural)} episodic={len(episodic)}",
        flush=True,
    )


async def _safe_auto_extract(
    message: str, source: str = "line_webhook_auto"
) -> None:
    try:
        await auto_extract_memory(message, source=source)
    except Exception as e:  # noqa: BLE001
        print(f"[memory-extract] failed: {e}", flush=True)


STATUS_TRIGGER_KEYWORDS = ("販売中", "公開中", "審査中", "開発中", "停止中")
STATUS_VALUES = ("公開中", "審査中", "販売中", "開発中", "リジェクト", "配信停止", "提出済")
PLATFORM_TOKENS = ("iOS", "iPadOS", "watchOS", "macOS", "Android", "web")
VERSION_RE = re.compile(r"v?\d+\.\d+(?:\.\d+)?(?:\([\d]+\))?")


def _looks_like_status_update(text: str) -> bool:
    return any(kw in text for kw in STATUS_TRIGGER_KEYWORDS)


def _parse_status_update_local(text: str) -> list[dict] | None:
    """Deterministic local parser for the simple `<name> [<platform>] [<version>] <status>` shape.

    Returns a single-item list on success, or None when the input doesn't fit
    the shape (e.g. multiple statuses in one message) so the caller can fall
    back to the Claude-based parser.
    """
    total_hits = sum(text.count(s) for s in STATUS_VALUES)
    if total_hits != 1:
        return None
    status = next(s for s in STATUS_VALUES if s in text)
    work = text.replace(status, " ")

    version = None
    m = VERSION_RE.search(work)
    if m:
        version = m.group(0).lstrip("v")
        work = work[:m.start()] + " " + work[m.end():]

    platform = None
    work_lower = work.lower()
    for p in PLATFORM_TOKENS:
        idx = work_lower.find(p.lower())
        if idx >= 0:
            platform = p
            work = work[:idx] + " " + work[idx + len(p):]
            break

    name = " ".join(work.split()).strip(" 　、,。.")
    if not name:
        return None
    out: dict = {"name": name, "status": status}
    if platform:
        out["platform"] = platform
    if version:
        out["version"] = version
    return [out]


def _match_apps(name: str, platform: str | None, apps_rows: list[dict]) -> list[dict]:
    """Return rows whose name overlaps `name` (case-insensitive substring either direction),
    optionally narrowed by platform exact-match."""
    nlow = name.lower()
    matches = [
        r for r in apps_rows
        if nlow in (r["name"] or "").lower() or (r["name"] or "").lower() in nlow
    ]
    if platform:
        plow = platform.lower()
        matches = [r for r in matches if (r["platform"] or "").lower() == plow]
    return matches


async def _apply_status_updates(updates: list[dict]) -> tuple[list[str], int]:
    """Apply parsed status updates. Returns (lines, success_count).

    - name は部分一致で apps テーブルと照合（platform 未指定 OK）。
    - 一致が 1 件 → UPDATE。
    - 一致 0 件 & platform 指定あり → INSERT 新規。
    - 一致 0 件 & platform 未指定 → エラー（新規かどうか判別できない）。
    - 一致 2 件以上 → エラー（platform 指定を促す）。"""
    lines: list[str] = []
    successes = 0
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, name, platform, status, version FROM apps"
        ) as cur:
            apps_rows = [dict(r) for r in await cur.fetchall()]

        for u in updates:
            name = u["name"]
            platform = u.get("platform")
            status = u["status"]
            version = u.get("version")
            ts = now_iso()
            matches = _match_apps(name, platform, apps_rows)

            if len(matches) == 1:
                row = matches[0]
                old_status = row["status"]
                old_version = row["version"]
                new_version = version or old_version
                await db.execute(
                    "UPDATE apps SET status = ?, version = ?, updated_at = ? WHERE id = ?",
                    (status, new_version, ts, row["id"]),
                )
                bits = [f"{row['name']} ({row['platform']})",
                        f"{old_status or '—'} → {status}"]
                if version and version != old_version:
                    bits.append(f"v{old_version or '—'} → v{version}")
                lines.append("✏️ " + " / ".join(bits))
                successes += 1
            elif not matches and platform:
                await db.execute(
                    """
                    INSERT INTO apps (name, platform, status, version, updated_at,
                                      assigned_agent, progress, next_action)
                    VALUES (?, ?, ?, ?, ?, NULL, 0, NULL)
                    """,
                    (name, platform, status, version, ts),
                )
                tail = f" v{version}" if version else ""
                lines.append(f"➕ {name} ({platform}) → {status}{tail}")
                successes += 1
            elif not matches:
                lines.append(
                    f"⚠️ 「{name}」に一致するアプリがありません。"
                    "新規追加なら platform も指定してください"
                )
            else:
                plats = ", ".join((r["platform"] or "—") for r in matches)
                lines.append(
                    f"⚠️ 「{name}」は複数プラットフォーム該当（{plats}）。"
                    "platform を指定してください"
                )
        await db.commit()
    if successes > 0:
        await _regenerate_agent_status_md()
    return lines, successes


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    expected = os.environ.get("API_KEY", "")
    if not expected or api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

_AMS_ROOT = Path(os.environ.get("AMS_ROOT", str(Path(__file__).resolve().parent)))
DB_PATH = Path(os.environ.get("AMS_DB_PATH", str(_AMS_ROOT / "memory.db")))
AGENT_STATUS_PATH = _AMS_ROOT / "agent_status.md"
AGENT_CONTEXT_DIR = _AMS_ROOT / "agent_contexts"
AGENT_CONTEXT_REMOTE_NAME = "agent_context.md"
LAN_URL = os.environ.get("AMS_LAN_URL", "")
TAILSCALE_URL = os.environ.get("AMS_PUBLIC_URL", "http://localhost:8000")
MCP_URL = f"{TAILSCALE_URL}/mcp"
OWNER_HANDLE = os.environ.get("OWNER_HANDLE", "owner")
DEFAULT_AGENT_NAME = os.environ.get("DEFAULT_AGENT", "default").strip() or "default"


def _status_update_hint() -> str:
    """Fallback line shown when the status parser cannot extract updates.
    Configure via STATUS_UPDATE_HINT_EXAMPLES ('|' separates examples)."""
    raw = os.environ.get("STATUS_UPDATE_HINT_EXAMPLES", "").strip()
    if not raw:
        return "例: 「<app-name> <platform> <status>」形式で送ってください。"
    examples = " ".join(f"「{ex.strip()}」" for ex in raw.split("|") if ex.strip())
    return f"例: {examples}"


def _md_cell(value) -> str:
    if value is None or value == "":
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


# Roman-key → display-name (JSON dict via env). Used by the per-agent
# `agent_status.md` writer and by other logging paths.
try:
    AGENT_NAME_JA: dict[str, str] = json.loads(
        os.environ.get("AGENT_NAME_JA", "{}") or "{}"
    )
    if not isinstance(AGENT_NAME_JA, dict):
        AGENT_NAME_JA = {}
except json.JSONDecodeError:
    AGENT_NAME_JA = {}


# Structured agent registry for the dashboard (JSON list of dicts). Each entry:
#   {"name": "<display>", "role": "<human text>", "drive_id": "<opt id>"}
# Empty list = skip the per-agent block in the dashboard entirely.
try:
    AGENT_DEFINITIONS: list[dict] = json.loads(
        os.environ.get("AGENT_DEFINITIONS", "[]") or "[]"
    )
    if not isinstance(AGENT_DEFINITIONS, list):
        AGENT_DEFINITIONS = []
except json.JSONDecodeError:
    AGENT_DEFINITIONS = []


async def _regenerate_agent_status_md() -> None:
    """Rewrite agent_status.md from the current apps + open tasks state.
    Single source of truth for both LINE status-update flow and the
    sync_status_to_project broadcast — keeps the file from drifting."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT name, platform, status, version, notes FROM apps ORDER BY id"
        ) as cur:
            apps = [dict(r) for r in await cur.fetchall()]
        async with db.execute(
            "SELECT agent, content, status FROM tasks WHERE status = 'open' "
            "ORDER BY agent, created_at"
        ) as cur:
            tasks = [dict(r) for r in await cur.fetchall()]
        decisions: list[dict] = []
        try:
            async with db.execute(
                "SELECT key, value, updated_at FROM semantic_memories "
                "WHERE category IN "
                "('design_decision', 'surface', 'infra', 'agent_policy') "
                "ORDER BY updated_at DESC LIMIT 10"
            ) as cur:
                decisions = [dict(r) for r in await cur.fetchall()]
        except aiosqlite.OperationalError:
            decisions = []

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append("# エージェント共有ステータス")
    lines.append(f"更新日時：{now_str}")
    lines.append("")
    lines.append("## アプリ一覧")
    lines.append("| アプリ | プラットフォーム | ステータス | バージョン | メモ |")
    lines.append("|---|---|---|---|---|")
    for a in apps:
        lines.append(
            f"| {_md_cell(a['name'])} | {_md_cell(a['platform'])} "
            f"| {_md_cell(a['status'])} | {_md_cell(a['version'])} "
            f"| {_md_cell(a['notes'])} |"
        )
    lines.append("")
    lines.append("## オープンタスク")
    if tasks:
        grouped: dict[str, list[dict]] = {}
        for t in tasks:
            display = AGENT_NAME_JA.get(t["agent"], t["agent"])
            grouped.setdefault(display, []).append(t)
        lines.append("| # | エージェント | 内容 | 状況 |")
        lines.append("|---|---|---|---|")
        idx = 1
        for agent_name in sorted(grouped.keys()):
            for t in grouped[agent_name]:
                lines.append(
                    f"| {idx} | {_md_cell(agent_name)} "
                    f"| {_md_cell(t['content'])} | {_md_cell(t['status'])} |"
                )
                idx += 1
    else:
        lines.append("_オープンタスクはありません。_")
    lines.append("")
    lines.append("## エージェント定義")
    lines.append("| 名前 | 役割 | DriveフォルダID |")
    lines.append("|---|---|---|")
    for a in AGENT_DEFINITIONS:
        lines.append(
            f"| {_md_cell(a['name'])} | {_md_cell(a['role'])} "
            f"| {_md_cell(a['drive_id'])} |"
        )
    lines.append("")
    lines.append("## Agent Memory Server")
    lines.append("| 項目 | 内容 |")
    lines.append("|---|---|")
    lines.append(f"| LAN内URL | {LAN_URL} |")
    lines.append(f"| 外部URL (MCP) | {MCP_URL} |")
    lines.append(f"| 外部URL (AMS root) | {TAILSCALE_URL} |")
    lines.append("| 認証 | X-API-Keyヘッダー必須（各エージェントのプロジェクト設定参照） |")
    lines.append(f"| 更新日時 | {now_str} |")
    lines.append("")
    lines.append("## 直近の確定事項")
    if decisions:
        lines.append("| key | 内容 | 更新日時 |")
        lines.append("|---|---|---|")
        for d in decisions:
            lines.append(
                f"| {_md_cell(d['key'])} | {_md_cell(d['value'])} "
                f"| {_md_cell(d['updated_at'])} |"
            )
    else:
        lines.append("_design_decision カテゴリの確定事項はまだありません。_")
    lines.append("")

    AGENT_STATUS_PATH.write_text("\n".join(lines), encoding="utf-8")


def _agent_context_path(agent: str) -> Path:
    return AGENT_CONTEXT_DIR / f"{agent}_context.md"


async def _regenerate_agent_context_md(
    agent: str,
    semantic_limit: int = 30,
    episodic_limit: int = 10,
) -> Path:
    """Rewrite <agent>_context.md from AMS memory for one agent.

    Pulls the latest N semantic_memories (updated_at DESC), all procedural_memories
    (updated_at DESC), and the latest M episodic_memories (session_date DESC,
    created_at DESC as tiebreaker). Returns the written path.
    """
    AGENT_CONTEXT_DIR.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT key, value, category, updated_at "
            "FROM semantic_memories WHERE agent = ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (agent, semantic_limit),
        ) as cur:
            semantic = [dict(r) for r in await cur.fetchall()]
        async with db.execute(
            "SELECT rule, source, updated_at FROM procedural_memories "
            "WHERE agent = ? ORDER BY updated_at DESC",
            (agent,),
        ) as cur:
            procedural = [dict(r) for r in await cur.fetchall()]
        async with db.execute(
            "SELECT summary, topics, session_date, created_at "
            "FROM episodic_memories WHERE agent = ? "
            "ORDER BY COALESCE(session_date, '') DESC, created_at DESC "
            "LIMIT ?",
            (agent, episodic_limit),
        ) as cur:
            episodic = [dict(r) for r in await cur.fetchall()]

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append(f"# {agent} コンテキスト要約")
    lines.append(f"更新日時：{now_str}")
    lines.append("")
    lines.append(
        "この文書はAMSから自動生成される。会話冒頭で読めば、前回までの確定事項・"
        "運用ルール・直近セッションの流れを引き継げる。ソースオブトゥルースはAMSのDB。"
    )
    lines.append("")

    lines.append(f"## Semantic（直近{semantic_limit}件、updated_at降順）")
    if semantic:
        lines.append("| key | category | 内容 | 更新日時 |")
        lines.append("|---|---|---|---|")
        for m in semantic:
            lines.append(
                f"| {_md_cell(m['key'])} | {_md_cell(m['category'])} "
                f"| {_md_cell(m['value'])} | {_md_cell(m['updated_at'])} |"
            )
    else:
        lines.append(f"_{agent} 宛のsemantic_memoriesはまだありません。_")
    lines.append("")

    lines.append("## Procedural（全件、updated_at降順）")
    if procedural:
        lines.append("| rule | source | 更新日時 |")
        lines.append("|---|---|---|")
        for m in procedural:
            lines.append(
                f"| {_md_cell(m['rule'])} | {_md_cell(m['source'])} "
                f"| {_md_cell(m['updated_at'])} |"
            )
    else:
        lines.append(f"_{agent} 宛のprocedural_memoriesはまだありません。_")
    lines.append("")

    lines.append(f"## Episodic（直近{episodic_limit}件、session_date降順）")
    if episodic:
        lines.append("| session_date | topics | summary | created_at |")
        lines.append("|---|---|---|---|")
        for m in episodic:
            lines.append(
                f"| {_md_cell(m['session_date'])} | {_md_cell(m['topics'])} "
                f"| {_md_cell(m['summary'])} | {_md_cell(m['created_at'])} |"
            )
    else:
        lines.append(f"_{agent} 宛のepisodic_memoriesはまだありません。_")
    lines.append("")

    out_path = _agent_context_path(agent)
    # claude.ai's upload API rejects payloads containing NUL — sanitize before
    # writing so a single stray \x00 in a memory row can't break broadcast.
    content = "\n".join(lines).replace("\x00", "")
    out_path.write_text(content, encoding="utf-8")
    return out_path


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS apps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                platform TEXT,
                status TEXT,
                version TEXT,
                notes TEXT,
                updated_at DATETIME NOT NULL,
                assigned_agent TEXT,
                progress INTEGER DEFAULT 0,
                next_action TEXT
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                progress INTEGER DEFAULT 0,
                next_action TEXT
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at DATETIME NOT NULL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS instructions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                given_by TEXT NOT NULL DEFAULT 'owner',
                assigned_to TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                priority TEXT NOT NULL DEFAULT 'medium',
                auto_execute INTEGER NOT NULL DEFAULT 0,
                deadline TEXT,
                context TEXT,
                result TEXT,
                line_message_id TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS line_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line_message_id TEXT NOT NULL,
                instruction_id INTEGER,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                direction TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                FOREIGN KEY (instruction_id) REFERENCES instructions(id)
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS morning_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                sent_at DATETIME NOT NULL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL UNIQUE,
                value TEXT NOT NULL,
                category TEXT NOT NULL,
                agent TEXT,
                owner TEXT,
                scope TEXT NOT NULL DEFAULT 'agent',
                source TEXT,
                source_reference TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        # Migration: add owner column on pre-existing tables, then backfill from agent.
        async with db.execute("PRAGMA table_info(semantic_memories)") as _cur:
            _sem_cols = {row[1] for row in await _cur.fetchall()}
        if "owner" not in _sem_cols:
            await db.execute(
                "ALTER TABLE semantic_memories ADD COLUMN owner TEXT"
            )
        await db.execute(
            "UPDATE semantic_memories SET owner = agent "
            "WHERE owner IS NULL AND agent IS NOT NULL"
        )
        # Migration: codify scope column (already ALTER-added on prod; codified here so
        # init_db()-created temp DBs match production).
        if "scope" not in _sem_cols:
            await db.execute(
                "ALTER TABLE semantic_memories ADD COLUMN scope TEXT "
                "NOT NULL DEFAULT 'agent'"
            )
        # Migration: source / source_reference for Codex integration (Phase 1).
        if "source" not in _sem_cols:
            await db.execute(
                "ALTER TABLE semantic_memories ADD COLUMN source TEXT"
            )
        if "source_reference" not in _sem_cols:
            await db.execute(
                "ALTER TABLE semantic_memories ADD COLUMN source_reference TEXT"
            )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS procedural_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                rule TEXT NOT NULL,
                source TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS episodic_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                summary TEXT NOT NULL,
                topics TEXT,
                session_date TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT NOT NULL,
                agent TEXT,
                source TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT DEFAULT (datetime('now')),
                notified_at TEXT,
                scope TEXT NOT NULL DEFAULT 'agent',
                source_reference TEXT
            )
            """
        )
        async with db.execute("PRAGMA table_info(pending_decisions)") as _cur:
            _cols = {row[1] for row in await _cur.fetchall()}
        if "notified_at" not in _cols:
            await db.execute(
                "ALTER TABLE pending_decisions ADD COLUMN notified_at TEXT"
            )
        # Migration: scope / source_reference for Codex integration (Phase 1).
        # Codex proposals carry scope='global' so promoted rows reach every agent.
        if "scope" not in _cols:
            await db.execute(
                "ALTER TABLE pending_decisions ADD COLUMN scope TEXT "
                "NOT NULL DEFAULT 'agent'"
            )
        if "source_reference" not in _cols:
            await db.execute(
                "ALTER TABLE pending_decisions ADD COLUMN source_reference TEXT"
            )
        await db.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                key, value, category, agent,
                content=semantic_memories,
                content_rowid=id
            )
            """
        )
        # Keep FTS5 in sync with semantic_memories.
        await db.execute(
            """
            CREATE TRIGGER IF NOT EXISTS semantic_memories_ai
            AFTER INSERT ON semantic_memories BEGIN
                INSERT INTO memories_fts(rowid, key, value, category, agent)
                VALUES (new.id, new.key, new.value, new.category, new.agent);
            END
            """
        )
        await db.execute(
            """
            CREATE TRIGGER IF NOT EXISTS semantic_memories_ad
            AFTER DELETE ON semantic_memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, key, value, category, agent)
                VALUES ('delete', old.id, old.key, old.value, old.category, old.agent);
            END
            """
        )
        await db.execute(
            """
            CREATE TRIGGER IF NOT EXISTS semantic_memories_au
            AFTER UPDATE ON semantic_memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, key, value, category, agent)
                VALUES ('delete', old.id, old.key, old.value, old.category, old.agent);
                INSERT INTO memories_fts(rowid, key, value, category, agent)
                VALUES (new.id, new.key, new.value, new.category, new.agent);
            END
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS drive_sync_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT NOT NULL UNIQUE,
                file_name TEXT NOT NULL,
                agent TEXT NOT NULL,
                synced_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        # Low-friction capture inbox — raw items from LINE/Slack/MCP/API.
        # Owner-gate / category checks intentionally do NOT apply here; the
        # only validation is content non-empty. Items are later promoted into
        # semantic_memories via the operator's approval flow.
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS inbox (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                content     TEXT NOT NULL,
                source      TEXT NOT NULL DEFAULT 'line',
                media_path  TEXT,
                status      TEXT NOT NULL DEFAULT 'unprocessed',
                promoted_to INTEGER,
                created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
            """
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_inbox_status ON inbox(status)"
        )
        await db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Agent Memory Server", lifespan=lifespan)
app.include_router(build_memory_router(verify_api_key))
app.include_router(build_inbox_router(verify_api_key))


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def row_to_dict(row: aiosqlite.Row) -> dict:
    return {k: row[k] for k in row.keys()}


class AppCreate(BaseModel):
    name: str
    platform: Optional[str] = None
    status: Optional[str] = None
    version: Optional[str] = None
    notes: Optional[str] = None
    assigned_agent: Optional[str] = None
    progress: Optional[int] = 0
    next_action: Optional[str] = None


class AppUpdate(BaseModel):
    name: Optional[str] = None
    platform: Optional[str] = None
    status: Optional[str] = None
    version: Optional[str] = None
    notes: Optional[str] = None
    assigned_agent: Optional[str] = None
    progress: Optional[int] = None
    next_action: Optional[str] = None


class TaskCreate(BaseModel):
    agent: str
    content: str
    status: Optional[str] = "open"
    progress: Optional[int] = 0
    next_action: Optional[str] = None


class TaskUpdate(BaseModel):
    status: Optional[str] = None
    content: Optional[str] = None
    agent: Optional[str] = None
    progress: Optional[int] = None
    next_action: Optional[str] = None


def _clamp_progress(value):
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="progress must be an integer")
    if n < 0 or n > 100:
        raise HTTPException(status_code=400, detail="progress must be between 0 and 100")
    return n


class SessionCreate(BaseModel):
    agent: str
    summary: str


class InstructionCreate(BaseModel):
    content: str
    assigned_to: str = "hack"
    priority: str = "medium"
    auto_execute: bool = True
    context: Optional[str] = None
    deadline: Optional[str] = None
    given_by: str = OWNER_HANDLE


# Subject allowlist for the /api/study-map endpoint. Configured via env var
# STUDY_MAP_SUBJECTS (comma-separated). Empty = endpoint rejects every subject
# with 400 (feature effectively disabled).
STUDY_MAP_SUBJECTS: tuple[str, ...] = tuple(
    s.strip() for s in os.environ.get("STUDY_MAP_SUBJECTS", "").split(",") if s.strip()
)

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


async def _persist_study_map(subject, session, data) -> tuple[bool, dict, int]:
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
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """
                INSERT INTO semantic_memories (key, value, category, agent)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    agent = excluded.agent,
                    updated_at = datetime('now')
                """,
                (key, value, "study",
                 os.environ.get("STUDY_COACH_AGENT") or DEFAULT_AGENT_NAME),
            )
            await db.commit()
            try:
                from embeddings import load_vec_extension, upsert_embedding
                async with db.execute(
                    "SELECT id FROM semantic_memories WHERE key = ?", (key,)
                ) as cur:
                    row = await cur.fetchone()
                if row:
                    await load_vec_extension(db)
                    await upsert_embedding(db, row["id"], key, value)
                    await db.commit()
            except Exception:
                pass
    except Exception as e:  # noqa: BLE001
        return False, {"ok": False, "error": f"db upsert failed: {e}"}, 502
    return True, {"ok": True}, 200


async def study_map_options(request: Request) -> Response:
    return Response(status_code=204, headers=STUDY_MAP_CORS)


@app.post("/api/study-map")
async def study_map_post(request: Request) -> JSONResponse:
    api_key = request.headers.get("X-API-Key", "")
    expected = os.environ.get("API_KEY", "")
    if not expected or api_key != expected:
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
    _, resp_body, status = await _persist_study_map(
        body.get("subject"), body.get("session"), body.get("data")
    )
    return JSONResponse(resp_body, status_code=status, headers=STUDY_MAP_CORS)


@app.post("/api/study-map/generate")
async def study_map_generate_post(request: Request) -> JSONResponse:
    api_key = request.headers.get("X-API-Key", "")
    expected = os.environ.get("API_KEY", "")
    if not expected or api_key != expected:
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

    # Study-map system prompt is fully user-configured via env. Empty = feature
    # disabled (endpoint returns 503 to make the misconfiguration obvious).
    system_prompt_template = os.environ.get("STUDY_MAP_SYSTEM_PROMPT", "").strip()
    if not system_prompt_template:
        return JSONResponse(
            {"ok": False, "error": "STUDY_MAP_SYSTEM_PROMPT not configured"},
            status_code=503, headers=STUDY_MAP_CORS,
        )
    try:
        system_prompt = system_prompt_template.format(subject=subject, session=session)
    except (KeyError, IndexError):
        system_prompt = system_prompt_template

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

    ok, resp_body, status = await _persist_study_map(subject, session, data)
    if not ok:
        return JSONResponse(resp_body, status_code=status, headers=STUDY_MAP_CORS)
    return JSONResponse({"ok": True, "data": data}, headers=STUDY_MAP_CORS)


app.add_api_route(
    "/api/study-map", study_map_options,
    methods=["OPTIONS"], include_in_schema=False,
)
app.add_api_route(
    "/api/study-map/generate", study_map_options,
    methods=["OPTIONS"], include_in_schema=False,
)


class SessionCheckoutIn(BaseModel):
    agent: str
    summary_text: str


CHECKOUT_SYSTEM = (
    "あなたはAMS（Agent Memory Server）の書き込み漏れ検出器。"
    "指定エージェントのセッション要約と、そのエージェントが直近に保存した記憶リストを受け取り、"
    "要約には含まれているがAMSに未保存の『確定情報』（事実・設計判断・インフラ変更・運用ルール・"
    "アプリ状態・タスク確定・学習成果）を抽出する。"
    "未確定の進行中タスク・試行錯誤・質問への返答は『確定情報』ではないので除外する。"
    "出力はJSONのみ。説明文・Markdownコードフェンス禁止。"
)


CHECKOUT_USER_TEMPLATE = """## エージェント
{agent}

## セッション要約（このセッションで出た内容）
{summary_text}

## このエージェントが直近に保存済みの記憶（最大50件）
{existing_text}

上記の「セッション要約」のうち、AMSに未保存の確定情報のみを抽出してください。
出力形式（このJSONだけ。前後にテキストやコードフェンスを付けないこと）:
{{
  "unsaved_items": [
    {{
      "key_suggestion": "snake_case_key",
      "value": "20字以上の簡潔な記述（日本語OK）",
      "category": "surface | design_decision | infra | agent_policy | app_status | task | study",
      "reason": "なぜ未保存と判断したか（1文）"
    }}
  ]
}}

未保存項目が一つもなければ unsaved_items を空配列にしてください。"""


def _strip_json_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


@app.post("/api/session/checkout")
async def session_checkout(
    payload: SessionCheckoutIn, api_key: str = Depends(verify_api_key)
):
    """Detect confirmed info present in the session summary but not yet saved
    to AMS for this agent. Intended to be called from each agent's end-of-session
    action — if `unsaved_items` is non-empty the agent should follow up with
    `save_memory` for each item before exiting."""
    agent = (payload.agent or "").strip()
    summary_text = (payload.summary_text or "").strip()
    if not agent:
        raise HTTPException(status_code=400, detail="agent must be non-empty")
    if not summary_text:
        raise HTTPException(
            status_code=400, detail="summary_text must be non-empty"
        )

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT key, value, category FROM semantic_memories "
            "WHERE agent = ? OR owner = ? "
            "ORDER BY updated_at DESC LIMIT 50",
            (agent, agent),
        ) as cur:
            existing = [dict(r) for r in await cur.fetchall()]
    existing_text = (
        "\n".join(
            f"- [{r['category']}] {r['key']}: {(r['value'] or '')[:120]}"
            for r in existing
        )
        or "(まだ何も保存されていません)"
    )

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        raise HTTPException(
            status_code=500, detail="ANTHROPIC_API_KEY not set"
        )

    user_msg = CHECKOUT_USER_TEMPLATE.format(
        agent=agent,
        summary_text=summary_text,
        existing_text=existing_text,
    )
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        msg = await asyncio.to_thread(
            client.messages.create,
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=CHECKOUT_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"claude call failed: {e}"
        )

    raw = "".join(getattr(b, "text", "") for b in msg.content)
    raw = _strip_json_fence(raw)
    try:
        data = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"invalid JSON from Claude: {e}; raw={raw[:300]}",
        )
    items = data.get("unsaved_items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        items = []

    cleaned: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        cleaned.append({
            "key_suggestion": (it.get("key_suggestion") or "").strip(),
            "value": (it.get("value") or "").strip(),
            "category": (it.get("category") or "").strip(),
            "reason": (it.get("reason") or "").strip(),
        })

    return {
        "agent": agent,
        "unsaved_items": cleaned,
        "should_save": len(cleaned) > 0,
        "hint": (
            f"未保存の確定情報が {len(cleaned)} 件あります。"
            "save_memory ツールで保存してからセッションを閉じてください。"
            if cleaned
            else "未保存の確定情報はありません。"
        ),
    }


@app.get("/")
async def root(api_key: str = Depends(verify_api_key)):
    return {"status": "ok", "message": "agent memory server running"}


@app.get("/status")
async def get_status(api_key: str = Depends(verify_api_key)):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM apps ORDER BY id") as cur:
            apps = [row_to_dict(r) for r in await cur.fetchall()]
        async with db.execute(
            "SELECT * FROM tasks WHERE status = 'open' ORDER BY created_at"
        ) as cur:
            tasks = [row_to_dict(r) for r in await cur.fetchall()]
    return {"apps": apps, "open_tasks": tasks}


@app.get("/apps")
async def list_apps(api_key: str = Depends(verify_api_key)):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM apps ORDER BY id") as cur:
            return [row_to_dict(r) for r in await cur.fetchall()]


@app.post("/apps")
async def create_app(payload: AppCreate, api_key: str = Depends(verify_api_key)):
    progress = _clamp_progress(payload.progress) if payload.progress is not None else 0
    ts = now_iso()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            INSERT INTO apps (name, platform, status, version, notes, updated_at,
                              assigned_agent, progress, next_action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.name, payload.platform, payload.status, payload.version,
                payload.notes, ts,
                payload.assigned_agent, progress, payload.next_action,
            ),
        )
        await db.commit()
        new_id = cur.lastrowid
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM apps WHERE id = ?", (new_id,)) as q:
            row = await q.fetchone()
    return row_to_dict(row)


@app.patch("/apps/{app_id}")
async def update_app(app_id: int, payload: AppUpdate, api_key: str = Depends(verify_api_key)):
    fields = {k: v for k, v in payload.dict().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="no fields to update")
    if "progress" in fields:
        fields["progress"] = _clamp_progress(fields["progress"])
    fields["updated_at"] = now_iso()
    sets = ", ".join(f"{k} = ?" for k in fields.keys())
    values = list(fields.values()) + [app_id]
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(f"UPDATE apps SET {sets} WHERE id = ?", values)
        await db.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="app not found")
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM apps WHERE id = ?", (app_id,)) as q:
            row = await q.fetchone()
    return row_to_dict(row)


@app.get("/tasks")
async def list_tasks(
    agent: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    api_key: str = Depends(verify_api_key),
):
    query = "SELECT * FROM tasks WHERE 1=1"
    params: list = []
    if agent:
        query += " AND agent = ?"
        params.append(agent)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at"
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cur:
            return [row_to_dict(r) for r in await cur.fetchall()]


@app.post("/tasks")
async def create_task(payload: TaskCreate, api_key: str = Depends(verify_api_key)):
    progress = _clamp_progress(payload.progress) if payload.progress is not None else 0
    ts = now_iso()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            INSERT INTO tasks (agent, content, status, created_at, updated_at,
                               progress, next_action)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.agent, payload.content, payload.status or "open", ts, ts,
                progress, payload.next_action,
            ),
        )
        await db.commit()
        new_id = cur.lastrowid
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM tasks WHERE id = ?", (new_id,)) as q:
            row = await q.fetchone()
    return row_to_dict(row)


@app.patch("/tasks/{task_id}")
async def update_task(task_id: int, payload: TaskUpdate, api_key: str = Depends(verify_api_key)):
    fields = {k: v for k, v in payload.dict().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="no fields to update")
    if "progress" in fields:
        fields["progress"] = _clamp_progress(fields["progress"])
    fields["updated_at"] = now_iso()
    sets = ", ".join(f"{k} = ?" for k in fields.keys())
    values = list(fields.values()) + [task_id]
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(f"UPDATE tasks SET {sets} WHERE id = ?", values)
        await db.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="task not found")
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)) as q:
            row = await q.fetchone()
    return row_to_dict(row)


@app.post("/sessions")
async def create_session(payload: SessionCreate, api_key: str = Depends(verify_api_key)):
    ts = now_iso()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO sessions (agent, summary, created_at) VALUES (?, ?, ?)",
            (payload.agent, payload.summary, ts),
        )
        await db.commit()
        new_id = cur.lastrowid
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM sessions WHERE id = ?", (new_id,)) as q:
            row = await q.fetchone()
    return row_to_dict(row)


@app.get("/sessions")
async def list_sessions(
    agent: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=200),
    api_key: str = Depends(verify_api_key),
):
    query = "SELECT * FROM sessions"
    params: list = []
    if agent:
        query += " WHERE agent = ?"
        params.append(agent)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cur:
            return [row_to_dict(r) for r in await cur.fetchall()]


@app.post("/instructions")
async def add_instruction(payload: InstructionCreate, api_key: str = Depends(verify_api_key)):
    ts = now_iso()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            INSERT INTO instructions
                (content, given_by, assigned_to, status, priority, auto_execute,
                 context, deadline, created_at, updated_at)
            VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.content,
                payload.given_by,
                payload.assigned_to,
                payload.priority,
                1 if payload.auto_execute else 0,
                payload.context,
                payload.deadline,
                ts,
                ts,
            ),
        )
        await db.commit()
        new_id = cur.lastrowid
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM instructions WHERE id = ?", (new_id,)) as q:
            row = await q.fetchone()
    return row_to_dict(row)


@app.get("/instructions")
async def list_instructions(
    status: Optional[str] = Query(None),
    assigned_to: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    api_key: str = Depends(verify_api_key),
):
    query = "SELECT * FROM instructions WHERE 1=1"
    params: list = []
    if status:
        query += " AND status = ?"
        params.append(status)
    if assigned_to:
        query += " AND assigned_to = ?"
        params.append(assigned_to)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cur:
            return [row_to_dict(r) for r in await cur.fetchall()]


def _verify_line_signature(body: bytes, signature: str) -> bool:
    secret = os.environ.get("LINE_CHANNEL_SECRET", "")
    if not secret:
        return False
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature or "")


def _verify_slack_signature(body: bytes, timestamp: str, signature: str) -> bool:
    """Slack v0 signature: HMAC-SHA256 over `v0:{ts}:{body}` using the signing
    secret. Rejects timestamps older than 5 minutes as replay protection."""
    secret = os.environ.get("SLACK_SIGNING_SECRET", "")
    if not secret or not timestamp or not signature:
        return False
    try:
        if abs(time.time() - int(timestamp)) > 60 * 5:
            return False
    except ValueError:
        return False
    base = b"v0:" + timestamp.encode("utf-8") + b":" + body
    digest = hmac.new(secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    expected = f"v0={digest}"
    return hmac.compare_digest(expected, signature)


async def _process_inbound_text(
    text: str, source_msg_id: str, source: str = "line_webhook_auto"
) -> None:
    """Run the unified inbound-message pipeline for a text from LINE or Slack DM.
    Handles OK:/LATER: confirmations, status updates, chat replies, and new
    instructions. The `source_msg_id` is persisted in line_conversations.line_message_id
    regardless of source (Slack passes its event ts). `source` is forwarded to the
    auto-extractor so persisted rows record their channel of origin."""
    # Background memory extraction — never blocks the webhook reply.
    asyncio.create_task(_safe_auto_extract(text, source=source))

    async with aiosqlite.connect(DB_PATH) as db:
        ts = now_iso()
        line_msg_id = source_msg_id
        if text.startswith("OK:") and text[3:].isdigit():
            instruction_id = int(text[3:])
            cur_inst = await db.execute(
                "SELECT content, result FROM instructions WHERE id = ?",
                (instruction_id,),
            )
            inst_row = await cur_inst.fetchone()
            inst_content = inst_row[0] if inst_row else ""
            inst_result = inst_row[1] if inst_row else None
            await db.execute(
                "UPDATE instructions SET status = 'confirmed', updated_at = ? WHERE id = ?",
                (ts, instruction_id),
            )
            await db.execute(
                """
                INSERT INTO line_conversations
                    (line_message_id, instruction_id, message_type, content, direction, created_at)
                VALUES (?, ?, 'confirm', ?, 'inbound', ?)
                """,
                (line_msg_id, instruction_id, text, ts),
            )
            await db.commit()
            await send_slack_message(f"✅ 確認しました（ID:{instruction_id}）")
            if inst_result and not _looks_like_status_update(inst_content or ""):
                await send_slack_message(inst_result)

        elif text.startswith("LATER:") and text[6:].isdigit():
            instruction_id = int(text[6:])
            await db.execute(
                """
                INSERT INTO line_conversations
                    (line_message_id, instruction_id, message_type, content, direction, created_at)
                VALUES (?, ?, 'later', ?, 'inbound', ?)
                """,
                (line_msg_id, instruction_id, text, ts),
            )
            await db.commit()
            await send_slack_message(
                f"⏰ 24時間後に再度お知らせします（ID:{instruction_id}）"
            )

        elif _is_market_scout_query(text):
            await db.execute(
                """
                INSERT INTO line_conversations
                    (line_message_id, instruction_id, message_type, content, direction, created_at)
                VALUES (?, NULL, 'market_scout', ?, 'inbound', ?)
                """,
                (line_msg_id, text, ts),
            )
            await db.commit()
            asyncio.create_task(_handle_market_scout())

        elif _looks_like_status_update(text):
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT name, platform, status, version FROM apps ORDER BY name, platform"
            ) as cur:
                apps_rows = await cur.fetchall()
            apps_summary = "\n".join(
                f"- {r['name']} | {r['platform']} | {r['status'] or '—'} | {r['version'] or '—'}"
                for r in apps_rows
            )
            updates = _parse_status_update_local(text)
            if not updates:
                updates = await parse_status_updates(text, apps_summary)
            await db.execute(
                """
                INSERT INTO line_conversations
                    (line_message_id, instruction_id, message_type, content, direction, created_at)
                VALUES (?, NULL, 'status_update', ?, 'inbound', ?)
                """,
                (line_msg_id, text, ts),
            )
            await db.commit()
            if updates:
                result_lines, n_ok = await _apply_status_updates(updates)
                header = (
                    "📊 ステータス更新を反映しました：" if n_ok > 0
                    else "⚠️ ステータス更新を反映できませんでした："
                )
                body = header + "\n" + "\n".join(result_lines)
            else:
                body = (
                    "⚠️ ステータス更新の対象を抽出できませんでした。\n"
                    + _status_update_hint()
                )
            await send_slack_message(body)

        elif _is_task_update_message(text):
            await db.execute(
                """
                INSERT INTO line_conversations
                    (line_message_id, instruction_id, message_type, content, direction, created_at)
                VALUES (?, NULL, 'chat', ?, 'inbound', ?)
                """,
                (line_msg_id, text, ts),
            )
            await db.commit()
            asyncio.create_task(_handle_chat(text, line_msg_id))

        elif _is_chat_message(text):
            await db.execute(
                """
                INSERT INTO line_conversations
                    (line_message_id, instruction_id, message_type, content, direction, created_at)
                VALUES (?, NULL, 'chat', ?, 'inbound', ?)
                """,
                (line_msg_id, text, ts),
            )
            await db.commit()
            asyncio.create_task(_handle_chat(text, line_msg_id))

        else:
            cur = await db.execute(
                """
                INSERT INTO instructions
                    (content, given_by, assigned_to, status, priority, auto_execute,
                     line_message_id, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', 'high', 1, ?, ?, ?)
                """,
                (text, OWNER_HANDLE, DEFAULT_AGENT_NAME, line_msg_id, ts, ts),
            )
            instruction_id = cur.lastrowid
            await db.execute(
                """
                INSERT INTO line_conversations
                    (line_message_id, instruction_id, message_type, content, direction, created_at)
                VALUES (?, ?, 'new_instruction', ?, 'inbound', ?)
                """,
                (line_msg_id, instruction_id, text, ts),
            )
            await db.commit()
            preview = text if len(text) <= 80 else text[:80] + "…"
            receipt_msg = (
                f"📝 指示を受け付けました（ID:{instruction_id}）\n{preview}\n\n自動実行します。"
            )
            await send_slack_message(receipt_msg)
            asyncio.create_task(_trigger_immediate_scan())


INBOX_MEDIA_DIR = Path.home() / "Projects" / "agent-memory-server" / "inbox_media"


async def _handle_inbox_line_event(event: dict) -> None:
    """Process one LINE message event for the inbox: save raw content, reply
    via Reply API. Reply is awaited (not fire-and-forget) so failures surface
    in the log and we don't lose the task to GC before the request fires.
    LINE reply tokens expire fast (~30s) so don't add extra latency here."""
    from line_client import (
        send_line_reply,
        fetch_line_message_content,
    )
    message = event.get("message", {}) or {}
    msg_type = message.get("type")
    reply_token = event.get("replyToken", "")
    msg_id = message.get("id", "")
    print(
        f"[inbox] event type={msg_type} reply_token_len={len(reply_token)} "
        f"msg_id={msg_id}",
        flush=True,
    )

    try:
        if msg_type == "text":
            text = (message.get("text") or "").strip()
            if not text:
                return
            await save_inbox_item(text, source="line")
            await send_line_reply(reply_token, "📥 受け取った")
        elif msg_type == "image":
            INBOX_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
            content_bytes = await fetch_line_message_content(msg_id)
            media_path = None
            if content_bytes:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = INBOX_MEDIA_DIR / f"{ts}_{msg_id}.jpg"
                fname.write_bytes(content_bytes)
                media_path = str(fname)
            await save_inbox_item(
                "[画像]", source="line", media_path=media_path
            )
            await send_line_reply(reply_token, "📥 画像を受け取った")
        else:
            return
    except Exception as e:  # noqa: BLE001
        print(f"[inbox] save failed: {e}", flush=True)
        try:
            await send_line_reply(reply_token, "⚠️ 保存失敗、もう一度送って")
        except Exception as e2:  # noqa: BLE001
            print(f"[inbox] error reply also failed: {e2}", flush=True)


@app.post("/inbox/line-webhook")
async def inbox_line_webhook(request: Request):
    """Inbox-dedicated LINE webhook. Signature-verified; filters by
    LINE_USER_ID; saves raw text/image to `inbox` and replies with a
    short confirmation via the (free) Reply API. No classification or
    instruction-pipeline routing — friction-free capture entrypoint."""
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    if not _verify_line_signature(body, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    allowed_uid = os.environ.get("LINE_USER_ID", "")
    for event in data.get("events", []):
        if event.get("type") != "message":
            continue
        src = event.get("source", {}) or {}
        # Quietly drop anything not from the configured owner — the inbox is a
        # personal external brain, not a public capture surface.
        if allowed_uid and src.get("userId") != allowed_uid:
            continue
        await _handle_inbox_line_event(event)
    return {"status": "ok"}


@app.post("/line-webhook")
async def line_webhook(request: Request):
    """Receive the owner's LINE replies. Signature-verified, no API key (LINE-signed)."""
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    if not _verify_line_signature(body, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    for event in data.get("events", []):
        if event.get("type") != "message":
            continue
        message = event.get("message", {}) or {}
        if message.get("type") != "text":
            continue
        text = (message.get("text") or "").strip()
        line_msg_id = message.get("id", "")
        if not text:
            continue
        await _process_inbound_text(text, line_msg_id, source="line_webhook_auto")

    return {"status": "ok"}


@app.post("/slack-webhook")
async def slack_webhook(request: Request):
    """Receive the owner's Slack DMs. Verifies Slack signing-secret signature and
    routes valid messages through the shared inbound pipeline. Handles
    url_verification handshake and ignores the bot's own messages."""
    body = await request.body()
    sig = request.headers.get("X-Slack-Signature", "")
    sig_ts = request.headers.get("X-Slack-Request-Timestamp", "")
    if not _verify_slack_signature(body, sig_ts, sig):
        raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Slack initial endpoint verification handshake.
    if data.get("type") == "url_verification":
        return {"challenge": data.get("challenge", "")}

    if data.get("type") != "event_callback":
        return {"status": "ignored"}

    event = data.get("event") or {}
    if event.get("type") != "message":
        return {"status": "ignored"}

    # Prevent reply loops: skip any message authored by the bot itself.
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return {"status": "ignored"}

    # Only the owner's DM to the bot is routed through the instruction pipeline.
    owner_uid = os.environ.get("SLACK_USER_ID", "")
    if owner_uid and event.get("user") != owner_uid:
        return {"status": "ignored"}
    channel_type = event.get("channel_type")
    if channel_type and channel_type != "im":
        return {"status": "ignored"}

    text = (event.get("text") or "").strip()
    slack_msg_id = event.get("ts", "")
    if not text:
        return {"status": "ignored"}

    # Queue any <ams:candidates> blocks the user pastes via Slack
    try:
        from candidates_parser import extract_and_queue
        extract_and_queue(text, source=f"slack:{slack_msg_id}")
    except Exception as e:  # noqa: BLE001
        print(f"[slack-webhook] candidates parse failed: {e}", flush=True)

    await _process_inbound_text(text, slack_msg_id, source="slack_dm_auto")
    return {"status": "ok"}
