"""Smoke tests for the inbox capture path.

Covers:
1. Signature failure → 403 on /inbox/line-webhook
2. Foreign user IDs silently dropped (200, no row inserted)
3. Allowed user text → inbox row saved + Reply API called
4. Empty content rejected on /api/inbox
5. /api/inbox POST → GET → PATCH lifecycle

Runs against a temp memory.db. Run via:
    .venv/bin/python test_inbox.py

Exits non-zero on any failure.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


TEST_API_KEY = "inbox-smoke-key"
TEST_LINE_SECRET = "inbox-test-line-secret"
TEST_LINE_UID = "U_test_line_uid"
TEST_LINE_TOKEN = "fake-line-token"

TMP_DIR = Path(tempfile.mkdtemp(prefix="ams_inbox_smoke_"))
TMP_DB = TMP_DIR / "memory.db"


def _seed_env() -> None:
    os.environ["API_KEY"] = TEST_API_KEY
    os.environ["LINE_CHANNEL_SECRET"] = TEST_LINE_SECRET
    os.environ["LINE_CHANNEL_ACCESS_TOKEN"] = TEST_LINE_TOKEN
    os.environ["LINE_USER_ID"] = TEST_LINE_UID
    os.environ.setdefault("MCP_ADMIN_PASS", "x")
    os.environ.setdefault("MCP_JWT_SECRET", "x")


def _swap_db_paths(tmp_db: Path) -> None:
    import main
    import memory_api
    import inbox_api
    main.DB_PATH = tmp_db
    main.INBOX_MEDIA_DIR = TMP_DIR / "inbox_media"
    memory_api.DB_PATH = tmp_db
    inbox_api.DB_PATH = tmp_db


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")
    sys.exit(1)


def _expect(cond: bool, msg: str) -> None:
    if cond:
        _ok(msg)
    else:
        _fail(msg)


HEADERS = {"X-API-Key": TEST_API_KEY}


def _line_signature(body: bytes, secret: str) -> str:
    return base64.b64encode(
        hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    ).decode("utf-8")


def _webhook_event(user_id: str, text: str, msg_id: str = "msg-1") -> bytes:
    return json.dumps({
        "events": [
            {
                "type": "message",
                "replyToken": "reply-token-test",
                "source": {"userId": user_id, "type": "user"},
                "message": {"type": "text", "id": msg_id, "text": text},
            }
        ]
    }).encode("utf-8")


def _count_inbox(db_path: Path) -> int:
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute("SELECT COUNT(*) FROM inbox")
        return cur.fetchone()[0]
    finally:
        con.close()


def run_webhook_tests(client) -> None:
    print("\n[1/3] /inbox/line-webhook")

    # 1. Bad signature → 403
    body = _webhook_event(TEST_LINE_UID, "テスト1")
    r = client.post(
        "/inbox/line-webhook",
        headers={"X-Line-Signature": "deadbeef"},
        content=body,
    )
    _expect(r.status_code == 403, f"invalid signature rejected ({r.status_code})")
    _expect(_count_inbox(TMP_DB) == 0, "no row inserted on bad signature")

    # 2. Other user → silently dropped (200, no row)
    body = _webhook_event("U_stranger", "侵入")
    sig = _line_signature(body, TEST_LINE_SECRET)
    r = client.post(
        "/inbox/line-webhook",
        headers={"X-Line-Signature": sig},
        content=body,
    )
    _expect(r.status_code == 200, f"foreign user webhook returns 200 ({r.status_code})")
    _expect(_count_inbox(TMP_DB) == 0, "foreign user did not insert row")

    # 3. Allowed user text → row saved, Reply API called
    captured: list[tuple[str, str]] = []

    async def _fake_reply(token: str, text: str) -> dict:
        captured.append((token, text))
        return {"ok": True}

    body = _webhook_event(TEST_LINE_UID, "  これメモしといて  ")
    sig = _line_signature(body, TEST_LINE_SECRET)
    import line_client
    with patch.object(line_client, "send_line_reply", _fake_reply):
        r = client.post(
            "/inbox/line-webhook",
            headers={"X-Line-Signature": sig},
            content=body,
        )
        # Reply runs as asyncio.create_task — give it a tick to fire.
        async def _drain():
            await asyncio.sleep(0.05)
        asyncio.get_event_loop().run_until_complete(_drain())

    _expect(r.status_code == 200, f"allowed user webhook 200 ({r.status_code})")
    _expect(_count_inbox(TMP_DB) == 1, "allowed user text saved one inbox row")
    _expect(
        any("受け取った" in t for _, t in captured),
        f"Reply API called with confirmation (captured={captured!r})",
    )

    # The stored row content was stripped of surrounding whitespace.
    import sqlite3
    con = sqlite3.connect(TMP_DB)
    row = con.execute(
        "SELECT content, source, status FROM inbox ORDER BY id DESC LIMIT 1"
    ).fetchone()
    con.close()
    _expect(
        row[0] == "これメモしといて" and row[1] == "line" and row[2] == "unprocessed",
        f"row content/source/status correct ({row!r})",
    )


def run_api_tests(client) -> None:
    print("\n[2/3] /api/inbox POST/GET/PATCH")

    # 1. Empty content → 400
    r = client.post(
        "/api/inbox", headers=HEADERS, json={"content": "   "}
    )
    _expect(r.status_code == 400, f"empty content rejected ({r.status_code})")

    # 2. Happy POST → echoes row
    r = client.post(
        "/api/inbox",
        headers=HEADERS,
        json={"content": "MCP経由で投入したい一言", "source": "mcp"},
    )
    _expect(r.status_code == 200, f"POST returns 200 ({r.status_code})")
    body = r.json()
    new_id = body.get("id")
    _expect(
        body.get("content") == "MCP経由で投入したい一言" and body.get("source") == "mcp",
        "POST echoes content/source",
    )
    _expect(body.get("status") == "unprocessed", "default status=unprocessed")

    # 3. List filtered by status
    r = client.get("/api/inbox", headers=HEADERS, params={"status": "unprocessed"})
    _expect(r.status_code == 200, f"GET returns 200 ({r.status_code})")
    rows = r.json().get("results", [])
    _expect(
        any(row["id"] == new_id for row in rows),
        "GET lists the newly inserted row",
    )

    # 4. PATCH → promoted
    r = client.patch(
        f"/api/inbox/{new_id}",
        headers=HEADERS,
        json={"status": "promoted", "promoted_to": 42},
    )
    _expect(r.status_code == 200, f"PATCH returns 200 ({r.status_code})")
    patched = r.json()
    _expect(
        patched.get("status") == "promoted" and patched.get("promoted_to") == 42,
        f"PATCH applied ({patched!r})",
    )

    # 5. PATCH with invalid status → 400
    r = client.patch(
        f"/api/inbox/{new_id}",
        headers=HEADERS,
        json={"status": "bogus"},
    )
    _expect(r.status_code == 400, f"invalid status rejected ({r.status_code})")

    # 6. PATCH non-existent id → 404
    r = client.patch(
        "/api/inbox/999999",
        headers=HEADERS,
        json={"status": "discarded"},
    )
    _expect(r.status_code == 404, f"unknown id returns 404 ({r.status_code})")

    # 7. After promotion, status='unprocessed' filter should exclude it.
    r = client.get("/api/inbox", headers=HEADERS, params={"status": "unprocessed"})
    rows = r.json().get("results", [])
    _expect(
        all(row["id"] != new_id for row in rows),
        "promoted row drops out of unprocessed filter",
    )


def run_image_tests(client) -> None:
    print("\n[3/3] image message saves media + reply")

    captured_replies: list[tuple[str, str]] = []

    async def _fake_reply(token: str, text: str) -> dict:
        captured_replies.append((token, text))
        return {}

    async def _fake_fetch(message_id: str) -> bytes:
        return b"\xff\xd8\xff\xe0fake-jpeg-bytes"

    body = json.dumps({
        "events": [
            {
                "type": "message",
                "replyToken": "img-reply",
                "source": {"userId": TEST_LINE_UID, "type": "user"},
                "message": {"type": "image", "id": "img-1"},
            }
        ]
    }).encode("utf-8")
    sig = _line_signature(body, TEST_LINE_SECRET)

    import line_client
    with patch.object(line_client, "send_line_reply", _fake_reply), \
         patch.object(line_client, "fetch_line_message_content", _fake_fetch):
        r = client.post(
            "/inbox/line-webhook",
            headers={"X-Line-Signature": sig},
            content=body,
        )
        async def _drain():
            await asyncio.sleep(0.05)
        asyncio.get_event_loop().run_until_complete(_drain())

    _expect(r.status_code == 200, f"image webhook 200 ({r.status_code})")

    import sqlite3
    con = sqlite3.connect(TMP_DB)
    row = con.execute(
        "SELECT content, source, media_path FROM inbox "
        "WHERE content = '[画像]' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    con.close()
    _expect(row is not None, "image row inserted with content='[画像]'")
    _expect(
        row is not None and row[2] and Path(row[2]).exists(),
        f"media file written ({row!r})",
    )
    _expect(
        any("画像" in t for _, t in captured_replies),
        f"image reply sent (captured={captured_replies!r})",
    )


def main() -> None:
    _seed_env()
    _swap_db_paths(TMP_DB)
    import main as main_mod
    from fastapi.testclient import TestClient

    asyncio.get_event_loop().run_until_complete(main_mod.init_db())
    with TestClient(main_mod.app) as client:
        run_webhook_tests(client)
        run_api_tests(client)
        run_image_tests(client)
    print("\nall inbox smoke tests passed ✅")


if __name__ == "__main__":
    main()
