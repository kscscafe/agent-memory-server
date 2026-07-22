"""Pull owner↔orchestrator Slack DM history into episodic_memories.

Scheduled every 10 minutes from scheduler.py. Dedup via the slack_ingest_log
table; each row keyed by (channel_id, message_ts).

Slack scopes required on the bot token:
    - chat:write         (existing — for posting)
    - im:history         (for reading DM history)
    - im:read            (only used if SLACK_DM_CHANNEL unset)

If a scope is missing, this module fails silently with a printed message
instead of crashing the scheduler.

Configuration env vars:
    SLACK_BOT_TOKEN              required
    SLACK_USER_ID                required (owner's Slack user id)
    SLACK_DM_CHANNEL             optional — DM channel id like Dxxxxxxxx;
                                 if unset we auto-discover via conversations.list
    SLACK_INGEST_INITIAL_HOURS   optional — first-run backfill window (default 24h)
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

load_dotenv(Path(__file__).resolve().parent / ".env")

from candidates_parser import extract_and_queue  # noqa: E402

DB_PATH = Path(__file__).resolve().parent / "memory.db"
MAX_LEN = 4000
DEFAULT_INITIAL_HOURS = 24


def _log(msg: str) -> None:
    print(f"[slack-ingester] {msg}", flush=True)


def _init_log_table() -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS slack_ingest_log (
            channel_id TEXT NOT NULL,
            message_ts TEXT NOT NULL,
            ingested_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (channel_id, message_ts)
        )
        """
    )
    con.commit()
    con.close()


async def _get_bot_user_id(client: AsyncWebClient) -> Optional[str]:
    """Resolve the bot's own Slack user_id via auth.test (cached per run)."""
    try:
        resp = await client.auth_test()
        return resp.get("user_id")
    except SlackApiError as e:
        _log(f"auth.test failed: {e.response.get('error','?')}")
        return None


def _is_bot_message(msg: dict, bot_user_id: Optional[str]) -> bool:
    """True if this message originated from a bot (skip from episodic)."""
    if msg.get("subtype") == "bot_message":
        return True
    if msg.get("bot_id"):
        return True
    if bot_user_id and msg.get("user") == bot_user_id:
        return True
    return False


async def _resolve_channel_id(client: AsyncWebClient, user_id: str) -> Optional[str]:
    """Resolve the DM channel id for `user_id`.

    Priority: explicit env var > conversations.list(types=im) > None.
    """
    explicit = os.environ.get("SLACK_DM_CHANNEL", "").strip()
    if explicit:
        return explicit
    try:
        resp = await client.conversations_list(types="im", limit=200)
    except SlackApiError as e:
        err = e.response.get("error", "?")
        _log(f"conversations.list failed: {err} — set SLACK_DM_CHANNEL or grant im:read scope")
        return None
    for ch in resp.get("channels", []):
        if ch.get("user") == user_id:
            return ch.get("id")
    _log(f"no DM channel found for user {user_id}")
    return None


def _last_ts(channel_id: str) -> Optional[str]:
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT MAX(message_ts) FROM slack_ingest_log WHERE channel_id=?",
        (channel_id,),
    ).fetchone()
    con.close()
    return row[0] if row and row[0] else None


async def _fetch_new_messages(
    client: AsyncWebClient, channel_id: str, oldest: str
) -> list[dict]:
    """Fetch messages with ts > oldest, paginated, chronological order."""
    messages: list[dict] = []
    cursor: Optional[str] = None
    while True:
        kwargs = {
            "channel": channel_id,
            "limit": 200,
            "oldest": oldest,
            "inclusive": False,
        }
        if cursor:
            kwargs["cursor"] = cursor
        try:
            resp = await client.conversations_history(**kwargs)
        except SlackApiError as e:
            err = e.response.get("error", "?")
            _log(f"conversations.history failed: {err} — grant im:history scope")
            return []
        messages.extend(resp.get("messages") or [])
        cursor = (resp.get("response_metadata") or {}).get("next_cursor")
        if not cursor:
            break
    # Slack returns newest-first; reverse for chronological insertion
    return list(reversed(messages))


def _save_message(channel_id: str, msg: dict, bot_user_id: Optional[str]) -> bool:
    """Process one Slack message.

    Always records to slack_ingest_log so last_ts advances monotonically.
    Only inserts into episodic_memories (and parses candidates) for non-bot
    messages. Returns True if a new episodic row was written.
    """
    text = (msg.get("text") or "").strip()
    ts = msg.get("ts")
    if not text or not ts:
        return False

    is_bot = _is_bot_message(msg, bot_user_id)

    con = sqlite3.connect(DB_PATH)
    try:
        already = con.execute(
            "SELECT 1 FROM slack_ingest_log "
            "WHERE channel_id=? AND message_ts=?",
            (channel_id, ts),
        ).fetchone()
        if already:
            return False

        # Record in ingest_log unconditionally so we never re-fetch this ts.
        con.execute(
            "INSERT INTO slack_ingest_log(channel_id, message_ts) VALUES (?, ?)",
            (channel_id, ts),
        )

        if not is_bot:
            summary = text[:MAX_LEN]
            session_date = datetime.fromtimestamp(
                float(ts), tz=timezone.utc
            ).strftime("%Y-%m-%d")
            con.execute(
                "INSERT INTO episodic_memories"
                "(agent, summary, topics, session_date) VALUES (?, ?, ?, ?)",
                ("operator", summary, "slack,REDACTED_dm", session_date),
            )
        con.commit()
    finally:
        con.close()

    if is_bot:
        return False

    try:
        n = extract_and_queue(text, source=f"slack_ingest:{ts}")
        if n:
            _log(f"queued {n} ams:candidates from ts={ts}")
    except Exception as e:  # noqa: BLE001
        _log(f"candidates parse failed for ts={ts}: {e}")
    return True


async def sync_slack_history() -> int:
    """Ingest new Slack DM messages. Returns rows inserted."""
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    user_id = os.environ.get("SLACK_USER_ID", "")
    if not token or not user_id:
        _log("credentials missing (SLACK_BOT_TOKEN / SLACK_USER_ID)")
        return 0

    _init_log_table()
    client = AsyncWebClient(token=token)

    channel_id = await _resolve_channel_id(client, user_id)
    if not channel_id:
        return 0

    bot_user_id = await _get_bot_user_id(client)
    if bot_user_id:
        _log(f"bot user_id resolved: {bot_user_id}")

    last = _last_ts(channel_id)
    if last:
        oldest = last
    else:
        hours = int(os.environ.get("SLACK_INGEST_INITIAL_HOURS", DEFAULT_INITIAL_HOURS))
        oldest = str(
            datetime.now(tz=timezone.utc).timestamp() - hours * 3600
        )
        _log(f"first run for channel {channel_id} — backfilling last {hours}h")

    messages = await _fetch_new_messages(client, channel_id, oldest)
    skip_subtypes = {"channel_join", "channel_leave", "channel_topic"}

    inserted = 0
    skipped_bot = 0
    for msg in messages:
        if msg.get("subtype") in skip_subtypes:
            continue
        if _is_bot_message(msg, bot_user_id):
            # Still record in ingest_log inside _save_message; just count separately
            _save_message(channel_id, msg, bot_user_id)
            skipped_bot += 1
            continue
        if _save_message(channel_id, msg, bot_user_id):
            inserted += 1

    _log(
        f"ingested {inserted} new user message(s), skipped {skipped_bot} bot "
        f"message(s) from channel {channel_id}"
    )
    return inserted


if __name__ == "__main__":
    n = asyncio.run(sync_slack_history())
    sys.exit(0)
