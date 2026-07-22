"""Run a single instruction through the Anthropic API, routed by agent persona.

Agent personas are loaded from prompt files under ``AGENT_PROMPT_DIR`` (defaults
to ``./prompts/``). For each canonical agent name we try, in order:
    <AGENT_PROMPT_DIR>/<name>.md         # local / private override
    <AGENT_PROMPT_DIR>/<name>.example.md # tracked template shipped with the repo
Fallback if neither exists: a minimal stub ``You are agent <name>.``
"""
import json
import os
import re
import asyncio
from pathlib import Path

import aiosqlite
import anthropic

import slack_client

DB_PATH = Path(
    os.environ.get("AMS_DB_PATH", str(Path(__file__).resolve().parent / "memory.db"))
)
PROMPT_DIR = Path(
    os.environ.get("AGENT_PROMPT_DIR", str(Path(__file__).resolve().parent / "prompts"))
)
HISTORY_LIMIT = 10
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2000


def _load_agent_prompt(name: str) -> str:
    """Return the prompt for ``name``. Falls back to ``.example.md`` then a stub."""
    for candidate in (PROMPT_DIR / f"{name}.md", PROMPT_DIR / f"{name}.example.md"):
        if candidate.exists():
            try:
                return candidate.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
    return f"You are agent {name}."

def _build_api_error_message(code: int | None) -> str:
    if code == 529:
        body = "Claude APIが混雑しています。しばらく待って自動再試行します。"
    elif code in (401, 403):
        body = (
            "Claude APIエラーが発生しました。"
            "課金切れまたはAPIキーの問題の可能性があります。"
            "Anthropicダッシュボードを確認してください。"
        )
    elif code == 429:
        body = "Claude APIのレートリミットに達しました。しばらく待って再試行します。"
    else:
        body = f"Claude APIで予期しないエラーが発生しました。(ステータスコード: {code})"
    return f"⚠️ {body}"


async def _push_api_error_async(message: str) -> None:
    """Push the API-error alert to Slack.

    LINE leg removed (operator moved to Slack — LinのLINE Botは別フローで稼働中)."""
    try:
        await slack_client.send_slack_message(message)
    except Exception as e:  # noqa: BLE001
        print(f"[claude_executor] Slack notification failed: {e}")


def _notify_line_api_error(code: int | None = None) -> None:
    """Best-effort Slack push when an Anthropic API call fails. Never raises.

    Name retained for historical reasons (was Slack+LINE; LINE leg dropped
    after operator moved to Slack). `code` is the HTTP status code from the
    Anthropic exception (e.g. `getattr(exc, "status_code", None)`)."""
    message = _build_api_error_message(code)
    try:
        asyncio.run(_push_api_error_async(message))
    except Exception as e:  # noqa: BLE001
        print(f"[claude_executor] API-error notification failed: {e}")

OPERATOR_PROMPT = _load_agent_prompt("operator")

AGENT_PROMPTS: dict[str, str] = {
    "operator": OPERATOR_PROMPT,
    "hack": _load_agent_prompt("hack"),
    "kirishima": _load_agent_prompt("kirishima"),
    "rik": _load_agent_prompt("rik"),
}
AGENT_PROMPTS["default"] = OPERATOR_PROMPT

# Aliases (case variants / Japanese names) for the built-in personas above.
# Extra prompts can be added by dropping <name>.md into AGENT_PROMPT_DIR and
# registering the alias here.
_ALIASES = {
    "operator": "operator",
    "雲": "operator",
    "agent_a": "hack",
    "agent_b": "kirishima",
    "agent_c": "rik",
    "agent_c": "rik",
}
for alias, canonical in _ALIASES.items():
    if canonical in AGENT_PROMPTS:
        AGENT_PROMPTS[alias] = AGENT_PROMPTS[canonical]


def _load_agent_status() -> str:
    """agent_status.mdを読み込んで返す。失敗したら空文字を返す"""
    try:
        status_path = os.path.join(os.path.dirname(__file__), "agent_status.md")
        with open(status_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _build_system_prompt() -> str:
    status = _load_agent_status()
    context = f"\n\n---\n## 現在の状態\n{status}" if status else ""
    return OPERATOR_PROMPT + context


def _system_prompt(agent: str) -> str:
    if not agent:
        return _build_system_prompt()
    resolved = AGENT_PROMPTS.get(agent.lower(), AGENT_PROMPTS.get(agent, AGENT_PROMPTS["default"]))
    if resolved is OPERATOR_PROMPT:
        return _build_system_prompt()
    return resolved


async def _fetch_history(limit: int, exclude_msg_id: str | None) -> list[dict]:
    """Fetch the last `limit` LINE messages (inbound + outbound), oldest first.
    Excludes the row that matches `exclude_msg_id` so the current instruction
    is not double-counted when it is also appended as the final user turn."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if exclude_msg_id:
            cur = await db.execute(
                """
                SELECT direction, content FROM line_conversations
                WHERE direction IN ('inbound', 'outbound')
                  AND (line_message_id IS NULL OR line_message_id != ?)
                ORDER BY created_at DESC LIMIT ?
                """,
                (exclude_msg_id, limit),
            )
        else:
            cur = await db.execute(
                """
                SELECT direction, content FROM line_conversations
                WHERE direction IN ('inbound', 'outbound')
                ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            )
        rows = [dict(r) for r in await cur.fetchall()]
    rows.reverse()
    return rows


def _build_messages(history: list[dict], current_user_msg: str) -> list[dict]:
    """Turn line_conversations rows into Claude messages. Coalesce consecutive
    same-role messages (the API requires strict user/assistant alternation),
    drop any leading assistant turn, and append the current instruction."""
    msgs: list[dict] = []
    for h in history:
        content = (h.get("content") or "").strip()
        if not content:
            continue
        role = "user" if h["direction"] == "inbound" else "assistant"
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] = msgs[-1]["content"] + "\n\n" + content
        else:
            msgs.append({"role": role, "content": content})
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] = msgs[-1]["content"] + "\n\n" + current_user_msg
    else:
        msgs.append({"role": "user", "content": current_user_msg})
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


def _run_sync(content: str, context: str, agent: str, history: list[dict]) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "ERROR: ANTHROPIC_API_KEY が設定されていません"
    client = anthropic.Anthropic(api_key=api_key)
    user_message = (
        "以下の指示を実行してください。\n\n"
        f"指示内容：{content}\n\n"
        f"背景・文脈：{context or 'なし'}\n\n"
        "具体的な成果物または実行結果を返してください。"
    )
    messages = _build_messages(history, user_message)
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=_system_prompt(agent),
            messages=messages,
        )
        return "".join(getattr(b, "text", "") for b in msg.content) or "(empty response)"
    except Exception as e:  # noqa: BLE001
        _notify_line_api_error(getattr(e, "status_code", None))
        return f"ERROR: {e}"


async def execute_instruction(instruction: dict) -> str:
    history = await _fetch_history(
        HISTORY_LIMIT,
        instruction.get("line_message_id"),
    )
    return await asyncio.to_thread(
        _run_sync,
        instruction.get("content", ""),
        instruction.get("context") or "",
        instruction.get("assigned_to") or "default",
        history,
    )


def _run_chat_sync(text: str, agent: str, history: list[dict]) -> str:
    """Claude call for short conversational replies (no '指示を実行してください' wrapper)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "ERROR: ANTHROPIC_API_KEY が設定されていません"
    client = anthropic.Anthropic(api_key=api_key)
    messages = _build_messages(history, text)
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=_system_prompt(agent),
            messages=messages,
        )
        return "".join(getattr(b, "text", "") for b in msg.content) or "(empty response)"
    except Exception as e:  # noqa: BLE001
        _notify_line_api_error(getattr(e, "status_code", None))
        return f"ERROR: {e}"


CU_LESSON_KEYWORDS = (
    "今週の授業", "残りの授業", "未視聴", "授業どこまで", "REDACTED", "CU",
)

# 第6週 / 第6回 / 6週目 / CU第6週 / CU6週 / CU第6回 / CU6回 すべて拾う
CU_WEEK_PATTERN = re.compile(r"第\s*(\d+)\s*[週回]|(\d+)\s*週目|CU\s*(\d+)\s*[週回]")


def _extract_week_number(text: str) -> int | None:
    if not text:
        return None
    m = CU_WEEK_PATTERN.search(text)
    if not m:
        return None
    raw = m.group(1) or m.group(2) or m.group(3)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


async def _handle_cu_week_query(week: int) -> str:
    """Fetch all lessons for the given week number and dress in operator's voice."""
    import notion_cu
    try:
        lessons = await asyncio.to_thread(notion_cu.get_lessons_by_week, week)
        raw = notion_cu.format_lessons_by_week(lessons)
    except Exception as e:  # noqa: BLE001
        raw = f"Notionの取得に失敗しました：{e}"
    return await format_with_operator_persona(raw)

KEI_TRIGGER = "agent_d"
KEI_CU_KEYWORDS = ("授業", "サイバー", "CU", "今週", "残り", "未視聴")

KEI_PROMPT = (
    "あなたはagent_d。REDACTEDのREDACTEDの学習を伴走する学習コーチです。"
    "明るくフラットな敬語で、励まし・後押しを大切にしながら、要点だけ簡潔に伝える。"
    "授業の進捗を聞かれたら、残件を見える化したうえで一言だけ前向きに背中を押す。"
    "余計な前置きや長い説教はしない。短く、具体的に、温かく。"
    "返答は必ず日本語。絵文字は📚📖✏️✨🎯のような学習系を控えめに使ってよい。"
)


def _is_cu_lesson_query(text: str) -> bool:
    if not text:
        return False
    return any(kw in text for kw in CU_LESSON_KEYWORDS)


def _is_kei_cu_query(text: str) -> bool:
    if not text or KEI_TRIGGER not in text:
        return False
    return any(kw in text for kw in KEI_CU_KEYWORDS)


async def _handle_cu_lesson_query(text: str) -> str:
    """Fetch lessons from Notion, format for LINE, then dress in operator's voice."""
    import notion_cu
    try:
        lessons = await asyncio.to_thread(notion_cu.get_this_week_lessons)
        raw = notion_cu.format_lessons_for_line(lessons)
    except Exception as e:  # noqa: BLE001
        raw = f"Notionの取得に失敗しました：{e}"
    return await format_with_operator_persona(raw)


async def _handle_kei_cu_query(text: str) -> str:
    """Same as the operator handler but routes the raw text through Kei's voice."""
    import notion_cu
    try:
        lessons = await asyncio.to_thread(notion_cu.get_this_week_lessons)
        raw = notion_cu.format_lessons_for_line(lessons)
    except Exception as e:  # noqa: BLE001
        raw = f"Notionの取得に失敗しました：{e}"
    return await format_with_kei_persona(raw)


TASK_DONE_KEYWORDS = ("完了", "done", "終わった", "終了", "クローズ", "close")
TASK_DELETE_KEYWORDS = ("削除", "delete", "消して", "消す", "なかったことに")


async def _handle_task_update(text: str) -> str | None:
    """
    テキストにタスク完了・削除キーワードが含まれていれば、
    tasksテーブルを更新してメッセージを返す。
    該当しなければNoneを返す。
    """
    is_done = any(kw in text for kw in TASK_DONE_KEYWORDS)
    is_delete = any(kw in text for kw in TASK_DELETE_KEYWORDS)
    if not (is_done or is_delete):
        return None

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, agent, content FROM tasks WHERE status = 'open' ORDER BY id"
        ) as cur:
            open_tasks = [dict(r) for r in await cur.fetchall()]

    if not open_tasks:
        return None

    task_list_str = "\n".join(
        f"id={t['id']} [{t['agent']}] {t['content']}" for t in open_tasks
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    client = anthropic.Anthropic(api_key=api_key)
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=200,
            system=(
                "あなたはタスクIDパーサーです。"
                "ユーザーの発言と一致するタスクのIDを抽出し、JSON配列のみ返してください。"
                "例: [7, 12] / 該当なし: []"
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"ユーザー発言：「{text}」\n\n"
                    f"オープンタスク一覧：\n{task_list_str}\n\n"
                    "発言に一致するタスクのIDをJSON配列で返してください。"
                )
            }],
        )
    except Exception:
        return None

    raw = "".join(getattr(b, "text", "") for b in msg.content).strip()
    try:
        ids = json.loads(raw)
        if not isinstance(ids, list) or not ids:
            return None
    except json.JSONDecodeError:
        return None

    new_status = "done"
    updated = []
    async with aiosqlite.connect(DB_PATH) as db:
        for task_id in ids:
            await db.execute(
                "UPDATE tasks SET status = ?, updated_at = datetime('now') WHERE id = ?",
                (new_status, task_id),
            )
            updated.append(task_id)
        await db.commit()

    if not updated:
        return None

    updated_contents = [
        t["content"] for t in open_tasks if t["id"] in updated
    ]
    lines = "\n".join(f"・{c}" for c in updated_contents)
    return f"以下のタスクを{new_status}にしました。\n{lines}"


async def chat_reply(text: str, line_message_id: str | None, agent: str = "operator") -> str:
    """Return a chat-style operator reply without persisting as an instruction."""
    task_result = await _handle_task_update(text)
    if task_result is not None:
        return task_result

    week = _extract_week_number(text)
    if week is not None:
        return await _handle_cu_week_query(week)
    if _is_kei_cu_query(text):
        return await _handle_kei_cu_query(text)
    if _is_cu_lesson_query(text):
        return await _handle_cu_lesson_query(text)
    history = await _fetch_history(HISTORY_LIMIT, line_message_id)
    return await asyncio.to_thread(_run_chat_sync, text, agent, history)


STATUS_PARSE_SYSTEM = (
    "あなたはアプリ管理DBのステータス更新パーサーです。"
    "ユーザーの自然言語メッセージから、apps テーブルへの更新内容を抽出し JSON 配列で返してください。"
    "JSON 以外の文章は一切返さないでください。"
)

STATUS_PARSE_USER_TEMPLATE = """既存アプリ一覧（name | platform | 現ステータス | 現バージョン）：
{apps_summary}

許可ステータス: 公開中, 審査中, 販売中, 開発中, リジェクト, 配信停止, 提出済

ユーザーメッセージ：
{text}

このメッセージに含まれるステータス変更を、以下スキーマの JSON 配列だけで返してください。
[{{"name": "<アプリ名>", "platform": "<iOS|Android|KDP|web|... 任意>", "status": "<許可ステータス>", "version": "<任意>"}}]

ルール:
- name は必須。既存アプリ名に一致するなら表記をその通りに揃える（部分指定でも構わない、後段で照合する）。
- platform はメッセージに明示されていれば入れる。明示されていなければキーごと省略してよい。
- version も同様に、明示されている場合のみ入れる。
- ステータスが特定できない要素は配列に含めない。何も無ければ空配列 [] を返す。
- JSON 以外の文字を出力しない。"""


def _parse_status_updates_sync(text: str, apps_summary: str) -> list[dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return []
    client = anthropic.Anthropic(api_key=api_key)
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=800,
            system=STATUS_PARSE_SYSTEM,
            messages=[{
                "role": "user",
                "content": STATUS_PARSE_USER_TEMPLATE.format(
                    apps_summary=apps_summary, text=text,
                ),
            }],
        )
    except Exception as e:  # noqa: BLE001
        _notify_line_api_error(getattr(e, "status_code", None))
        return []
    raw = "".join(getattr(b, "text", "") for b in msg.content).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    cleaned: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        status = (item.get("status") or "").strip()
        if not name or not status:
            continue
        out: dict = {"name": name, "status": status}
        platform = (item.get("platform") or "").strip()
        if platform:
            out["platform"] = platform
        version = item.get("version")
        if version:
            out["version"] = str(version).strip()
        cleaned.append(out)
    return cleaned


async def parse_status_updates(text: str, apps_summary: str) -> list[dict]:
    return await asyncio.to_thread(_parse_status_updates_sync, text, apps_summary)


def _format_with_operator_persona_sync(raw_result: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return raw_result
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=_build_system_prompt(),
            messages=[
                {"role": "user", "content": f"以下の実行結果をREDACTEDに報告してください。\n\n{raw_result}"}
            ],
        )
        text = "".join(getattr(b, "text", "") for b in response.content).strip()
        return text or raw_result
    except Exception:
        return raw_result


async def format_with_operator_persona(raw_result: str) -> str:
    """コード系実行結果をoperatorのペルソナで整形して返す。失敗時は生のテキストをそのまま返す。"""
    return await asyncio.to_thread(_format_with_operator_persona_sync, raw_result)


def _format_with_kei_persona_sync(raw_result: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return raw_result
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=KEI_PROMPT,
            messages=[
                {"role": "user", "content": f"以下の未視聴授業リストをREDACTEDに伝えてください。\n\n{raw_result}"}
            ],
        )
        text = "".join(getattr(b, "text", "") for b in response.content).strip()
        return text or raw_result
    except Exception:
        return raw_result


async def format_with_kei_persona(raw_result: str) -> str:
    """REDACTEDの未視聴授業リストをagent_dのペルソナで整形して返す。失敗時は生のテキストをそのまま返す。"""
    return await asyncio.to_thread(_format_with_kei_persona_sync, raw_result)
