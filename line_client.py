"""LINE Messaging API client — push text and confirm-template messages to the configured owner."""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
import aiosqlite

LINE_API_PUSH = "https://api.line.me/v2/bot/message/push"
LINE_API_REPLY = "https://api.line.me/v2/bot/message/reply"
LINE_API_CONTENT = "https://api-data.line.me/v2/bot/message/{id}/content"
DB_PATH = Path(
    os.environ.get("AMS_DB_PATH", str(Path(__file__).resolve().parent / "memory.db"))
)


async def _log_outbound(
    content: str,
    message_type: str,
    instruction_id: int | None = None,
) -> None:
    """Persist outbound LINE message into line_conversations so claude_executor
    can include assistant turns in its context window. Errors are swallowed."""
    ts = datetime.utcnow().isoformat()
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                INSERT INTO line_conversations
                    (line_message_id, instruction_id, message_type, content, direction, created_at)
                VALUES ('', ?, ?, ?, 'outbound', ?)
                """,
                (instruction_id, message_type, content, ts),
            )
            await db.commit()
    except Exception as e:  # noqa: BLE001
        print(f"[line_client] outbound log failed: {e}")


def _credentials() -> tuple[str, str]:
    return (
        os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", ""),
        os.environ.get("LINE_USER_ID", ""),
    )


def _headers(token: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


async def _push(payload: dict) -> dict:
    token, user_id = _credentials()
    if not token or not user_id:
        print("[line_client] credentials not configured; skipping push")
        return {}
    async with aiohttp.ClientSession() as session:
        async with session.post(LINE_API_PUSH, headers=_headers(token), json=payload) as resp:
            data = await resp.json()
            if resp.status >= 400:
                print(f"[line_client] push failed status={resp.status} body={data}")
            return data


async def send_line_message(text: str) -> dict:
    _, user_id = _credentials()
    payload = {
        "to": user_id,
        "messages": [{"type": "text", "text": text[:4900]}],
    }
    result = await _push(payload)
    await _log_outbound(text, message_type="message")
    return result


async def send_line_reply(reply_token: str, text: str) -> dict:
    """Reply via LINE Reply API (free — does NOT consume 200-msg/month push
    quota). Each replyToken is single-use and expires after ~30s.

    Skip the all-zero token LINE sends during the "verify" button test in the
    Developers console — that token is intentionally unusable.

    If Reply fails with "Invalid reply token" (most often because the LINE
    Official Account's auto-reply / 応答メッセージ feature already consumed it),
    fall back to Push so the owner still gets the confirmation. Push DOES count
    against quota — disable 応答メッセージ in LINE Official Account Manager
    to keep replies on the free path."""
    token, _ = _credentials()
    if not token or not reply_token:
        return {}
    if reply_token == "0" * 32:
        print("[line_client] skipping reply: LINE verify-test token")
        return {"verify_test_skipped": True}
    payload = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text[:4900]}],
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                LINE_API_REPLY, headers=_headers(token), json=payload
            ) as resp:
                data = await resp.json()
                if resp.status >= 400:
                    msg = (data or {}).get("message", "") if isinstance(data, dict) else ""
                    print(
                        f"[line_client] reply failed status={resp.status} "
                        f"body={data}"
                    )
                    if "Invalid reply token" in msg:
                        print(
                            "[line_client] falling back to push "
                            "(likely auto-reply consumed the token)"
                        )
                        push_result = await send_line_message(text)
                        return {
                            "reply_failed": data,
                            "push_fallback": push_result,
                        }
                return data
    except Exception as e:  # noqa: BLE001
        print(f"[line_client] reply error: {e}")
        return {}


async def fetch_line_message_content(message_id: str) -> Optional[bytes]:
    """Download an image/video/file message body via the LINE data API."""
    token, _ = _credentials()
    if not token or not message_id:
        return None
    url = LINE_API_CONTENT.format(id=message_id)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers={"Authorization": f"Bearer {token}"}
            ) as resp:
                if resp.status >= 400:
                    print(
                        f"[line_client] content fetch failed status={resp.status}"
                    )
                    return None
                return await resp.read()
    except Exception as e:  # noqa: BLE001
        print(f"[line_client] content fetch error: {e}")
        return None


async def send_line_confirm(text: str, instruction_id: int) -> dict:
    _, user_id = _credentials()
    payload = {
        "to": user_id,
        "messages": [
            {
                "type": "template",
                "altText": text[:399],
                "template": {
                    "type": "confirm",
                    "text": text[:160],
                    "actions": [
                        {"type": "message", "label": "OK", "text": f"OK:{instruction_id}"},
                        {"type": "message", "label": "あとで", "text": f"LATER:{instruction_id}"},
                    ],
                },
            }
        ],
    }
    result = await _push(payload)
    await _log_outbound(text, message_type="confirm", instruction_id=instruction_id)
    return result
