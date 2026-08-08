"""Background executor: pick up pending auto_execute instructions, run them
through Claude, store the result, and push a notification. Also produces the
morning report."""
from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite

from claude_executor import execute_instruction
from code_runner import is_code_task, run_claude_code
from slack_client import send_slack_message

DB_PATH = Path(
    os.environ.get("AMS_DB_PATH", str(Path(__file__).resolve().parent / "memory.db"))
)

BATCH_SIZE = 3
PRIORITY_ORDER = "CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 WHEN 'low' THEN 2 ELSE 3 END"


async def _sync_lessons_to_ams(week: int, lessons: list[dict]) -> None:
    """Snapshot lesson-tracker progress into semantic_memories under a
    weekly key. Agent is taken from STUDY_COACH_AGENT (falling back to
    DEFAULT_AGENT), so a fresh install won't accidentally mint rows under a
    placeholder identity."""
    import json
    summary = {
        "week": week,
        "synced_at": datetime.utcnow().isoformat(),
        "subjects": [
            {
                "subject": r.get("subject") or r.get("title"),
                "status": r.get("status"),
                "due": r.get("due"),
            }
            for r in lessons
        ],
    }
    value = json.dumps(summary, ensure_ascii=False)
    key = f"lesson_week{week:02d}_notion_sync"
    agent = (
        os.environ.get("STUDY_COACH_AGENT")
        or os.environ.get("DEFAULT_AGENT", "default")
    )
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO semantic_memories (key, value, category, agent, created_at, updated_at)
            VALUES (?, ?, 'study', ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, value, agent, now, now),
        )
        await db.commit()
    print(f"[morning_report] AMS lesson_sync done: {key}", flush=True)


async def scan_and_execute() -> int:
    """Process up to BATCH_SIZE pending auto_execute instructions. Returns count processed."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            f"""
            SELECT * FROM instructions
            WHERE status = 'pending' AND auto_execute = 1
            ORDER BY {PRIORITY_ORDER}, created_at ASC
            LIMIT ?
            """,
            (BATCH_SIZE,),
        )
        rows = await cursor.fetchall()

    processed = 0
    for row in rows:
        inst = dict(row)
        ts = datetime.utcnow().isoformat()
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE instructions SET status = 'in_progress', updated_at = ? WHERE id = ?",
                (ts, inst["id"]),
            )
            await db.commit()

        is_code = is_code_task(inst["content"])
        if is_code:
            result = await run_claude_code(inst["content"], inst.get("context"))
            badge = "🛠️ Claude Code完了"
        else:
            result = await execute_instruction(inst)
            badge = "✅ 完了"

        ts2 = datetime.utcnow().isoformat()
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE instructions SET status = 'done', result = ?, updated_at = ? WHERE id = ?",
                (result, ts2, inst["id"]),
            )
            await db.commit()

        icon = "🛠️" if is_code else "✅"
        body = result if len(result) <= 4500 else result[:4500] + "\n…(truncated)"
        notification = f"{icon} ID:{inst['id']} 実行結果\n\n{body}"
        await send_slack_message(notification)

        # Slack lacks an equivalent of LINE's confirm-template buttons.
        # Until a Block-Kit interactive flow is wired up, ask the owner to reply
        # with the OK:/LATER: text — _process_inbound_text already handles those.
        preview = inst["content"] if len(inst["content"]) <= 80 else inst["content"][:80] + "…"
        await send_slack_message(
            f"{badge}【ID:{inst['id']}】\n{preview}\n\n"
            f"確認なら `OK:{inst['id']}` 、延期なら `LATER:{inst['id']}` と返信してください。"
        )
        processed += 1

    return processed


LIVE_STATUSES = {"公開中", "販売中"}
PLATFORM_ORDER = ["iOS", "iPadOS", "macOS", "watchOS", "Android", "web"]
STATUS_BADGES = {
    "審査中": "⏳",
    "開発中": "🔧",
    "提出済": "📤",
    "リジェクト": "❌",
    "配信停止": "⏸",
}


def _platform_sort_key(p: str) -> int:
    try:
        return PLATFORM_ORDER.index(p)
    except ValueError:
        return len(PLATFORM_ORDER)


def _format_apps_summary(apps: list[dict]) -> list[str]:
    """Aggregate live apps by platform combo; list non-live apps explicitly."""
    live_by_name: dict[str, set[str]] = {}
    exceptions: list[dict] = []
    for a in apps:
        status = (a.get("status") or "").strip()
        name = a.get("name") or ""
        platform = (a.get("platform") or "").strip()
        if status in LIVE_STATUSES:
            live_by_name.setdefault(name, set()).add(platform)
        elif status:
            exceptions.append(a)

    combo_counts: dict[str, int] = {}
    for pf_set in live_by_name.values():
        key = "+".join(sorted(pf_set, key=_platform_sort_key)) or "-"
        combo_counts[key] = combo_counts.get(key, 0) + 1

    out: list[str] = []
    if combo_counts:
        sorted_combos = sorted(
            combo_counts.items(),
            key=lambda kv: (-kv[1], _platform_sort_key(kv[0].split("+")[0])),
        )
        out.append("\n📱 公開中：" + " / ".join(f"{k} {v}件" for k, v in sorted_combos))
    for a in exceptions:
        status = (a.get("status") or "-").strip()
        badge = STATUS_BADGES.get(status, "📌")
        platform = a.get("platform") or "-"
        out.append(f"  {badge} {status}：{a.get('name','-')} ({platform})")
    return out


async def send_morning_report() -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute(
            f"""
            SELECT * FROM instructions
            WHERE status IN ('pending', 'in_progress')
            ORDER BY {PRIORITY_ORDER}, created_at ASC
            LIMIT 5
            """
        )
        pending = [dict(r) for r in await cursor.fetchall()]

        yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()
        cursor = await db.execute(
            "SELECT * FROM instructions WHERE status IN ('done','confirmed') AND updated_at >= ? ORDER BY updated_at DESC",
            (yesterday,),
        )
        done_yesterday = [dict(r) for r in await cursor.fetchall()]

        cursor = await db.execute(
            "SELECT name, platform, status, version FROM apps ORDER BY updated_at DESC LIMIT 100"
        )
        apps = [dict(r) for r in await cursor.fetchall()]

        cursor = await db.execute(
            "SELECT agent, COUNT(*) AS n FROM tasks WHERE status = 'open' GROUP BY agent ORDER BY n DESC, agent"
        )
        agent_counts = [dict(r) for r in await cursor.fetchall()]

        cursor = await db.execute(
            "SELECT agent, content FROM tasks WHERE status = 'open' ORDER BY agent, created_at"
        )
        open_tasks = [dict(r) for r in await cursor.fetchall()]

    def _clean(s: str) -> str:
        return (s or "").replace("**", "")

    def _filter_for_report(rows: list[dict]) -> list[dict]:
        """Drop code-routed internal instructions and collapse duplicate contents."""
        seen: set[str] = set()
        out: list[dict] = []
        for r in rows:
            content = r.get("content") or ""
            if is_code_task(content):
                continue
            key = content.strip()
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    done_yesterday = _filter_for_report(done_yesterday)
    pending = _filter_for_report(pending)

    lines = ["おはようございます☀️\n"]
    if done_yesterday:
        lines.append(f"✅ 昨日の完了：{len(done_yesterday)}件")
        for t in done_yesterday[:3]:
            lines.append(f"  • {_clean(t['content'])[:40]}")
    if pending:
        lines.append(f"\n⏳ 未完了指示：{len(pending)}件")
        for t in pending[:3]:
            lines.append(f"  • [{t['priority']}] {_clean(t['content'])[:40]}")
    if apps:
        for line in _format_apps_summary(apps):
            lines.append(line)
    if agent_counts:
        summary = " / ".join(f"{c['agent']}:{c['n']}" for c in agent_counts)
        lines.append(f"\n👥 エージェント状態（オープンタスク数）：\n  {summary}")
    if open_tasks:
        lines.append(f"\n📋 オープンタスク（{len(open_tasks)}件）：")
        current_agent = None
        for t in open_tasks:
            if t["agent"] != current_agent:
                lines.append(f"  ［{t['agent']}］")
                current_agent = t["agent"]
            lines.append(f"    • {_clean(t['content'])[:60]}")

    try:
        import notion_cu
        cu_summary = await asyncio.to_thread(notion_cu.get_current_week_summary)
        cu_line = notion_cu.format_current_week_summary(cu_summary)
        if cu_line:
            lines.append(f"\n{cu_line}")
        if cu_summary and cu_summary.get("week"):
            week = cu_summary["week"]
            lessons = await asyncio.to_thread(notion_cu.get_lessons_by_week, week)
            await _sync_lessons_to_ams(week, lessons)
    except Exception as e:  # noqa: BLE001
        print(f"[morning_report] notion_cu fetch failed: {e}", flush=True)

    lines.append("\n今日もよろしくお願いします。")

    report = "\n".join(lines)

    try:
        await send_slack_message(report)
    except Exception as e:  # noqa: BLE001
        print(f"[morning_report] Slack send failed: {e}", flush=True)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO morning_reports (content, sent_at) VALUES (?, ?)",
            (report, datetime.utcnow().isoformat()),
        )
        await db.commit()
    return report


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if cmd == "morning":
        asyncio.run(send_morning_report())
    else:
        n = asyncio.run(scan_and_execute())
        print(f"processed {n} instruction(s)")
