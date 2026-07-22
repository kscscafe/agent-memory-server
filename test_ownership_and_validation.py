"""Smoke tests for the owner-gate, category validation, thin-write warning,
and the new /api/session/checkout endpoint.

Runs against a temporary memory.db so it does not touch production data.
Invoke via:
    .venv/bin/python test_ownership_and_validation.py

The script exits with a non-zero status on any failure.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parent
TMP_DB = Path(tempfile.mkdtemp(prefix="ams_smoke_")) / "memory.db"
TEST_API_KEY = "smoke-test-key"


def _seed_env() -> None:
    """Point the app at our temp DB and a deterministic API key BEFORE import."""
    os.environ["API_KEY"] = TEST_API_KEY
    # Other env vars main.py reads at import time — keep them off the prod values
    # so we don't accidentally talk to the real LINE/Slack/Anthropic accounts.
    os.environ.setdefault("MCP_ADMIN_PASS", "x")
    os.environ.setdefault("MCP_JWT_SECRET", "x")


def _swap_db_paths(tmp_db: Path) -> None:
    import main
    import memory_api
    main.DB_PATH = tmp_db
    memory_api.DB_PATH = tmp_db


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


def run_owner_and_validation_tests(client) -> None:
    print("\n[1/3] owner gate + category + thin-write")

    # 1. First write claims ownership.
    key = "smoke_key_owner_gate_01"
    r = client.post(
        "/memory/semantic",
        headers=HEADERS,
        json={
            "key": key,
            "value": "初回書き込みでagent_aが所有者になることを確認するための十分な長さ",
            "category": "design_decision",
            "agent": "agent_a",
        },
    )
    _expect(r.status_code == 200, f"first write succeeds ({r.status_code})")
    body = r.json()
    _expect(body.get("owner") == "agent_a", f"owner=agent_a recorded ({body.get('owner')!r})")
    _expect("warning" not in body, "no thin-write warning on full-length value")

    # 2. Same owner updates → 200.
    r = client.post(
        "/memory/semantic",
        headers=HEADERS,
        json={
            "key": key,
            "value": "agent_aが自分のキーを更新できることを確認する2回目の書き込み",
            "category": "design_decision",
            "agent": "agent_a",
        },
    )
    _expect(r.status_code == 200, f"owner can update its own key ({r.status_code})")

    # 3. Different agent → 403.
    r = client.post(
        "/memory/semantic",
        headers=HEADERS,
        json={
            "key": key,
            "value": "agent_bがagent_aのキーを上書きしようとして拒否されるべきケース",
            "category": "design_decision",
            "agent": "agent_b",
        },
    )
    _expect(
        r.status_code == 403,
        f"non-owner blocked with 403 (got {r.status_code})",
    )
    _expect(
        "owned by 'agent_a'" in r.json().get("detail", ""),
        "error message names the rightful owner",
    )

    # 4. Invalid category → 422 (pydantic).
    r = client.post(
        "/memory/semantic",
        headers=HEADERS,
        json={
            "key": "smoke_key_bad_category",
            "value": "カテゴリが不正なので拒否されるべき書き込みテスト",
            "category": "mac_mini_repo",  # removed from ALLOWED_CATEGORIES
            "agent": "agent_a",
        },
    )
    _expect(
        r.status_code == 422,
        f"unknown category rejected with 422 (got {r.status_code})",
    )

    # 5. Thin value → 200 + warning flag.
    r = client.post(
        "/memory/semantic",
        headers=HEADERS,
        json={
            "key": "smoke_key_thin_write",
            "value": "短い",  # only 2 chars
            "category": "task",
            "agent": "agent_a",
        },
    )
    _expect(r.status_code == 200, f"thin write still succeeds ({r.status_code})")
    body = r.json()
    _expect(
        body.get("warning") == "thin_write",
        f"thin-write warning flag present ({body.get('warning')!r})",
    )

    # 6. Backward compat: a fresh key by a different agent is fine.
    r = client.post(
        "/memory/semantic",
        headers=HEADERS,
        json={
            "key": "smoke_key_kirishima_owns",
            "value": "agent_bが新規キーを作るのは当然できる（既存挙動の後方互換確認）",
            "category": "surface",
            "agent": "agent_b",
        },
    )
    _expect(
        r.status_code == 200 and r.json().get("owner") == "agent_b",
        "fresh key by another agent succeeds and records its own owner",
    )


def run_pending_promotion_test(client) -> None:
    print("\n[2/3] owner gate on pending → semantic promotion")

    # agent_a owns smoke_key_owner_gate_01 from the previous test.
    # A pending decision from agent_b with the same key must NOT be approvable.
    r = client.post(
        "/memory/pending",
        headers=HEADERS,
        json={
            "key": "smoke_key_owner_gate_01",
            "value": "agent_bがpending経由で上書きを試みるが、承認時に拒否されるべき",
            "category": "design_decision",
            "agent": "agent_b",
            "source": "smoke-test",
        },
    )
    _expect(r.status_code == 200, f"pending submission accepted ({r.status_code})")
    pending_id = r.json().get("id")

    r = client.patch(
        f"/memory/pending/{pending_id}",
        headers=HEADERS,
        json={"action": "approve"},
    )
    _expect(
        r.status_code == 403,
        f"pending approval gated by owner (got {r.status_code})",
    )

    # Rejecting the same pending decision should still work.
    r = client.patch(
        f"/memory/pending/{pending_id}",
        headers=HEADERS,
        json={"action": "reject"},
    )
    _expect(
        r.status_code == 200 and r.json().get("status") == "rejected",
        "pending decision can still be rejected after failed approval",
    )


def run_checkout_test(client) -> None:
    print("\n[3/3] /api/session/checkout")

    # 1. Empty agent → 400.
    r = client.post(
        "/api/session/checkout",
        headers=HEADERS,
        json={"agent": "", "summary_text": "x"},
    )
    _expect(r.status_code == 400, f"empty agent rejected ({r.status_code})")

    # 2. Empty summary → 400.
    r = client.post(
        "/api/session/checkout",
        headers=HEADERS,
        json={"agent": "agent_a", "summary_text": "   "},
    )
    _expect(r.status_code == 400, f"empty summary rejected ({r.status_code})")

    # 3. Happy path with a mocked Claude response — must not require real API key.
    fake_claude_json = json.dumps({
        "unsaved_items": [
            {
                "key_suggestion": "smoke_unsaved_item_demo",
                "value": "checkoutが未保存項目を返すかを確認するためのダミーレコード",
                "category": "design_decision",
                "reason": "summaryには出てくるが既存記憶リストに無い",
            }
        ]
    })

    class _FakeBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class _FakeMessage:
        content = [_FakeBlock(fake_claude_json)]

    class _FakeMessages:
        def create(self, **kwargs):
            assert kwargs.get("model") == "claude-sonnet-4-20250514", kwargs.get("model")
            return _FakeMessage()

    class _FakeAnthropic:
        def __init__(self, api_key: str) -> None:
            self.messages = _FakeMessages()

    fake_anthropic_mod = type("anthropic_stub", (), {"Anthropic": _FakeAnthropic})

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake"}):
        with patch.dict(sys.modules, {"anthropic": fake_anthropic_mod}):
            r = client.post(
                "/api/session/checkout",
                headers=HEADERS,
                json={
                    "agent": "agent_a",
                    "summary_text": (
                        "今日はREDACTEDの週ランキングを実装し、Supabase関数を追加した。"
                        "operatorから次にAMSの所有者ゲートを実装する指示を受けた。"
                    ),
                },
            )

    _expect(r.status_code == 200, f"checkout returns 200 ({r.status_code})")
    body = r.json()
    _expect(body.get("should_save") is True, "should_save=True when items found")
    _expect(
        isinstance(body.get("unsaved_items"), list)
        and len(body["unsaved_items"]) == 1
        and body["unsaved_items"][0]["key_suggestion"] == "smoke_unsaved_item_demo",
        "unsaved_items parsed correctly",
    )

    # 4. Empty Claude result → should_save=False.
    fake_empty = json.dumps({"unsaved_items": []})

    class _FakeMessageEmpty:
        content = [_FakeBlock(fake_empty)]

    class _FakeMessagesEmpty:
        def create(self, **kwargs):
            return _FakeMessageEmpty()

    class _FakeAnthropicEmpty:
        def __init__(self, api_key: str) -> None:
            self.messages = _FakeMessagesEmpty()

    fake_mod_empty = type(
        "anthropic_stub2", (), {"Anthropic": _FakeAnthropicEmpty}
    )

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake"}):
        with patch.dict(sys.modules, {"anthropic": fake_mod_empty}):
            r = client.post(
                "/api/session/checkout",
                headers=HEADERS,
                json={"agent": "agent_a", "summary_text": "雑談だけだった"},
            )

    _expect(r.status_code == 200, f"empty-result checkout returns 200 ({r.status_code})")
    _expect(
        r.json().get("should_save") is False,
        "should_save=False when nothing missing",
    )


def run_catalog_tests(client) -> None:
    print("\n[bonus] semantic-memory catalogue")

    r = client.get(
        "/memory/semantic?agent=agent_a&limit=2&offset=0", headers=HEADERS
    )
    _expect(r.status_code == 200, f"catalogue list succeeds ({r.status_code})")
    body = r.json()
    _expect(body.get("total", 0) >= 2, "catalogue returns total count")
    _expect(len(body.get("results", [])) == 2, "catalogue respects page size")
    _expect(
        all("value_preview" in row and "value" not in row for row in body["results"]),
        "catalogue returns previews without full values",
    )

    r = client.get(
        "/memory/semantic/smoke_key_owner_gate_01", headers=HEADERS
    )
    _expect(r.status_code == 200, f"exact-key lookup succeeds ({r.status_code})")
    _expect(
        r.json().get("key") == "smoke_key_owner_gate_01" and "value" in r.json(),
        "exact-key lookup returns full memory",
    )

    r = client.get("/memory/semantic/does_not_exist", headers=HEADERS)
    _expect(r.status_code == 404, "missing exact key returns 404")

    r = client.get("/memory/semantic?scope=shared", headers=HEADERS)
    _expect(r.status_code == 400, "invalid catalogue scope returns 400")

    r = client.get("/memory/semantic?limit=0", headers=HEADERS)
    _expect(r.status_code == 422, "invalid catalogue page size returns 422")


def run_codex_integration_tests(client) -> None:
    """Phase 1: Codex-source pending → auto-approver exclusion → manual promotion
    with scope/source/source_reference preserved → owner-gate still enforced."""
    print("\n[bonus] Codex integration (scope + source persistence, T-CDX-01..05)")

    # T-CDX-01: Codex-style pending accepted; body echoes scope/source/source_reference.
    r = client.post(
        "/memory/pending",
        headers=HEADERS,
        json={
            "key": "smoke_codex_v1",
            "value": (
                "Codex由来のグローバル情報のスモークテスト。手動承認まで昇格しない。"
            ),
            "category": "design_decision",
            "agent": "operator",
            "source": "codex",
            "scope": "global",
            "source_reference": "https://example.com/codex-smoke",
        },
    )
    _expect(r.status_code == 200, f"T-CDX-01: codex pending accepted ({r.status_code})")
    body = r.json()
    codex_id = body.get("id")
    _expect(
        body.get("source") == "codex",
        f"T-CDX-01: pending source stored as 'codex' ({body.get('source')!r})",
    )
    _expect(
        body.get("scope") == "global",
        f"T-CDX-01: pending scope stored as 'global' ({body.get('scope')!r})",
    )
    _expect(
        body.get("source_reference") == "https://example.com/codex-smoke",
        f"T-CDX-01: pending source_reference stored ({body.get('source_reference')!r})",
    )

    # T-CDX-01b: unknown scope value rejected by validator.
    r = client.post(
        "/memory/pending",
        headers=HEADERS,
        json={
            "key": "smoke_codex_bad_scope",
            "value": "スコープ値が無効な場合の拒否テスト。'shared'は許容されない。",
            "category": "design_decision",
            "agent": "operator",
            "scope": "shared",
        },
    )
    _expect(
        r.status_code == 422,
        f"T-CDX-01b: unknown scope rejected 422 (got {r.status_code})",
    )

    # T-CDX-01c: over-length source_reference rejected.
    r = client.post(
        "/memory/pending",
        headers=HEADERS,
        json={
            "key": "smoke_codex_long_ref",
            "value": "source_reference が上限を超える場合のバリデーション確認テスト。",
            "category": "design_decision",
            "agent": "operator",
            "source_reference": "x" * 2001,
        },
    )
    _expect(
        r.status_code == 422,
        f"T-CDX-01c: over-length source_reference rejected 422 (got {r.status_code})",
    )

    # T-CDX-02: source='codex' is skipped by the auto-approver, while a control
    # source='mcp' row of the same category IS auto-approved (existing behaviour).
    r = client.post(
        "/memory/pending",
        headers=HEADERS,
        json={
            "key": "smoke_mcp_control_v1",
            "value": "コントロール群のMCP由来pending。既存挙動どおり自動承認される想定。",
            "category": "design_decision",
            "agent": "operator",
            "source": "mcp",
        },
    )
    _expect(
        r.status_code == 200,
        f"T-CDX-02: mcp control pending accepted ({r.status_code})",
    )

    import pending_approver as pa
    _prev_db_path = pa.DB_PATH
    pa.DB_PATH = TMP_DB
    try:
        approved = asyncio.get_event_loop().run_until_complete(pa.approve_pending())
    finally:
        pa.DB_PATH = _prev_db_path

    _expect(
        "smoke_mcp_control_v1" in approved,
        "T-CDX-02: mcp control auto-approved (existing behaviour preserved)",
    )
    _expect(
        "smoke_codex_v1" not in approved,
        "T-CDX-02: Codex candidate skipped by auto-approver",
    )

    con = sqlite3.connect(TMP_DB)
    con.row_factory = sqlite3.Row
    codex_row = con.execute(
        "SELECT status FROM pending_decisions WHERE id=?", (codex_id,)
    ).fetchone()
    con.close()
    _expect(
        codex_row is not None and codex_row["status"] == "pending",
        f"T-CDX-02: Codex row still status='pending' "
        f"({codex_row['status'] if codex_row else '<missing>'!r})",
    )

    # T-CDX-03: manual PATCH approve promotes with scope/source/source_reference
    # preserved, owner set to target_agent (existing manual-approval behaviour).
    r = client.patch(
        f"/memory/pending/{codex_id}",
        headers=HEADERS,
        json={"action": "approve"},
    )
    _expect(
        r.status_code == 200,
        f"T-CDX-03: manual approve on Codex candidate succeeds ({r.status_code})",
    )

    con = sqlite3.connect(TMP_DB)
    con.row_factory = sqlite3.Row
    sem = con.execute(
        "SELECT scope, source, source_reference, owner "
        "FROM semantic_memories WHERE key=?", ("smoke_codex_v1",)
    ).fetchone()
    con.close()
    _expect(sem is not None, "T-CDX-03: promoted semantic row exists")
    _expect(
        sem["scope"] == "global",
        f"T-CDX-03: promoted scope='global' (got {sem['scope']!r})",
    )
    _expect(
        sem["source"] == "codex",
        f"T-CDX-03: promoted source='codex' (got {sem['source']!r})",
    )
    _expect(
        sem["source_reference"] == "https://example.com/codex-smoke",
        f"T-CDX-03: promoted source_reference preserved ({sem['source_reference']!r})",
    )
    _expect(
        sem["owner"] == "operator",
        f"T-CDX-03: manual promotion sets owner=agent ({sem['owner']!r})",
    )

    # T-CDX-04: /memory/context for an unrelated agent surfaces the global row.
    r = client.get("/memory/context?agent=agent_g", headers=HEADERS)
    _expect(
        r.status_code == 200,
        f"T-CDX-04: get_context for agent_g succeeds ({r.status_code})",
    )
    keys = [row.get("key") for row in r.json().get("semantic", [])]
    _expect(
        "smoke_codex_v1" in keys,
        f"T-CDX-04: global row visible to unrelated agent agent_g "
        f"(sample keys: {keys[:6]})",
    )

    # T-CDX-05: owner-gate still blocks Codex approval when the target key is
    # already owned by another agent.
    r = client.post(
        "/memory/semantic",
        headers=HEADERS,
        json={
            "key": "smoke_codex_ownergate_target",
            "value": "agent_bが既に所有しているキー。Codex由来pending承認時に403となる対象。",
            "category": "design_decision",
            "agent": "agent_b",
        },
    )
    _expect(
        r.status_code == 200,
        f"T-CDX-05: prep write establishes agent_b ownership ({r.status_code})",
    )

    r = client.post(
        "/memory/pending",
        headers=HEADERS,
        json={
            "key": "smoke_codex_ownergate_target",
            "value": (
                "operator宛てのCodex提案が既存owner=agent_bとの衝突で承認時403となるはず。"
            ),
            "category": "design_decision",
            "agent": "operator",
            "source": "codex",
            "scope": "global",
        },
    )
    _expect(
        r.status_code == 200,
        f"T-CDX-05: Codex pending accepted despite existing ownership "
        f"(gate is on approve, not on submit) ({r.status_code})",
    )
    conflict_id = r.json().get("id")

    r = client.patch(
        f"/memory/pending/{conflict_id}",
        headers=HEADERS,
        json={"action": "approve"},
    )
    _expect(
        r.status_code == 403,
        f"T-CDX-05: owner-gate blocks Codex approval on owned key "
        f"({r.status_code})",
    )


def _verify_existing_data_migrated() -> None:
    """In the real prod DB the migration UPDATE backfills owner from agent.
    This smoke test uses a fresh temp DB so we simulate the case by seeding a
    row with owner NULL and running the same migration query."""
    print("\n[bonus] owner backfill migration")
    con = sqlite3.connect(TMP_DB)
    con.execute("DELETE FROM semantic_memories WHERE key = ?", ("smoke_legacy_row",))
    con.execute(
        "INSERT INTO semantic_memories (key, value, category, agent, owner) "
        "VALUES (?, ?, ?, ?, NULL)",
        ("smoke_legacy_row", "owner未設定の旧データを模した行", "task", "agent_b"),
    )
    con.commit()
    con.execute(
        "UPDATE semantic_memories SET owner = agent "
        "WHERE owner IS NULL AND agent IS NOT NULL"
    )
    con.commit()
    row = con.execute(
        "SELECT owner FROM semantic_memories WHERE key = ?", ("smoke_legacy_row",)
    ).fetchone()
    con.close()
    _expect(row and row[0] == "agent_b", "legacy NULL-owner row backfilled to agent")


def main() -> None:
    _seed_env()
    _swap_db_paths(TMP_DB)
    # Import after env + DB swap so init_db points at the temp DB.
    import main as main_mod
    from fastapi.testclient import TestClient

    asyncio.get_event_loop().run_until_complete(main_mod.init_db())
    with TestClient(main_mod.app) as client:
        run_owner_and_validation_tests(client)
        run_pending_promotion_test(client)
        run_codex_integration_tests(client)
        run_catalog_tests(client)
        run_checkout_test(client)
    _verify_existing_data_migrated()
    print("\nall smoke tests passed ✅")


if __name__ == "__main__":
    main()
