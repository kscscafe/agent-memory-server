"""Run a single instruction through the Anthropic API, routed by agent persona.

Agent personas are loaded from prompt files under ``AGENT_PROMPT_DIR`` (defaults
to ``./prompts/``). The default operator's prompt basename is set via
``DEFAULT_AGENT`` (default ``default``, shipping template
``prompts/default.example.md``). Additional personas can be registered by
setting the ``AGENT_PROMPT_MAP`` env var to a JSON dict
``{"<display-name>": "<basename>", ...}``.

For each basename we try, in order:
    <AGENT_PROMPT_DIR>/<basename>.md         # local / private override
    <AGENT_PROMPT_DIR>/<basename>.example.md # tracked template shipped with the repo
Fallback if neither exists: a minimal stub ``You are agent <name>.``

The optional study-coach and lesson-tracker flows are inert unless the
corresponding env vars are configured (see .env.example).
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

    LINE leg was removed when the operator moved to Slack; a separate LINE
    bot lives under its own project and is out of scope for this module."""
    try:
        await slack_client.send_slack_message(message)
    except Exception as e:  # noqa: BLE001
        print(f"[claude_executor] Slack notification failed: {e}")


def _notify_line_api_error(code: int | None = None) -> None:
    """Best-effort Slack push when an Anthropic API call fails. Never raises.

    Function name retained for historical reasons (was Slack+LINE; the LINE
    leg was dropped when the operator moved to Slack). `code` is the HTTP
    status code from the Anthropic exception
    (e.g. `getattr(exc, "status_code", None)`)."""
    message = _build_api_error_message(code)
    try:
        asyncio.run(_push_api_error_async(message))
    except Exception as e:  # noqa: BLE001
        print(f"[claude_executor] API-error notification failed: {e}")


DEFAULT_AGENT = os.environ.get("DEFAULT_AGENT", "default").strip() or "default"
PRIMARY_PROMPT = _load_agent_prompt(DEFAULT_AGENT)

AGENT_PROMPTS: dict[str, str] = {
    DEFAULT_AGENT: PRIMARY_PROMPT,
    "default": PRIMARY_PROMPT,
}

# Additional personas: {"<display>": "<basename>"}. Each display name and each
# basename are registered as lookup keys, so callers can refer to a persona
# either way.
try:
    _AGENT_PROMPT_MAP = json.loads(os.environ.get("AGENT_PROMPT_MAP", "{}") or "{}")
except json.JSONDecodeError:
    _AGENT_PROMPT_MAP = {}
if isinstance(_AGENT_PROMPT_MAP, dict):
    for _display, _basename in _AGENT_PROMPT_MAP.items():
        if not (isinstance(_display, str) and isinstance(_basename, str)):
            continue
        prompt = _load_agent_prompt(_basename)
        AGENT_PROMPTS[_display] = prompt
        AGENT_PROMPTS[_basename] = prompt


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
    return PRIMARY_PROMPT + context


def _system_prompt(agent: str) -> str:
    if not agent:
        return _build_system_prompt()
    resolved = AGENT_PROMPTS.get(agent.lower(), AGENT_PROMPTS.get(agent, AGENT_PROMPTS[DEFAULT_AGENT]))
    if resolved is PRIMARY_PROMPT:
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


# ── Optional lesson-tracker flow ─────────────────────────────────────
# Inert unless LESSON_TRACKER_KEYWORDS (JSON array) is configured. Optionally
# a LESSON_TRACKER_WEEK_PATTERN regex extracts a week number from the user's
# message and routes through notion_cu.get_lessons_by_week().
try:
    LESSON_TRACKER_KEYWORDS: tuple[str, ...] = tuple(
        json.loads(os.environ.get("LESSON_TRACKER_KEYWORDS", "[]") or "[]")
    )
except json.JSONDecodeError:
    LESSON_TRACKER_KEYWORDS = ()

_lesson_pattern_src = os.environ.get("LESSON_TRACKER_WEEK_PATTERN", "").strip()
LESSON_TRACKER_WEEK_PATTERN: re.Pattern | None = (
    re.compile(_lesson_pattern_src) if _lesson_pattern_src else None
)


def _extract_week_number(text: str) -> int | None:
    if not text or LESSON_TRACKER_WEEK_PATTERN is None:
        return None
    m = LESSON_TRACKER_WEEK_PATTERN.search(text)
    if not m:
        return None
    for grp in m.groups() or ():
        if grp:
            try:
                return int(grp)
            except (TypeError, ValueError):
                continue
    return None


async def _handle_lesson_week_query(week: int) -> str:
    """Fetch all lessons for the given week number, format via the default persona."""
    try:
        import notion_cu
        lessons = await asyncio.to_thread(notion_cu.get_lessons_by_week, week)
        raw = notion_cu.format_lessons_by_week(lessons)
    except Exception as e:  # noqa: BLE001
        raw = f"Notion fetch failed: {e}"
    return await format_with_default_persona(raw)


def _is_lesson_query(text: str) -> bool:
    if not text or not LESSON_TRACKER_KEYWORDS:
        return False
    return any(kw in text for kw in LESSON_TRACKER_KEYWORDS)


async def _handle_lesson_query(text: str) -> str:
    """Fetch this-week lessons via notion_cu, then format via default persona."""
    try:
        import notion_cu
        lessons = await asyncio.to_thread(notion_cu.get_this_week_lessons)
        raw = notion_cu.format_lessons_for_line(lessons)
    except Exception as e:  # noqa: BLE001
        raw = f"Notion fetch failed: {e}"
    return await format_with_default_persona(raw)


# ── Optional study-coach persona flow ────────────────────────────────
# Inert unless STUDY_COACH_TRIGGER *and* STUDY_COACH_SYSTEM_PROMPT are set.
STUDY_COACH_TRIGGER = os.environ.get("STUDY_COACH_TRIGGER", "").strip()
try:
    STUDY_COACH_KEYWORDS: tuple[str, ...] = tuple(
        json.loads(os.environ.get("STUDY_COACH_KEYWORDS", "[]") or "[]")
    )
except json.JSONDecodeError:
    STUDY_COACH_KEYWORDS = ()
STUDY_COACH_SYSTEM_PROMPT = os.environ.get("STUDY_COACH_SYSTEM_PROMPT", "").strip()


def _is_study_coach_query(text: str) -> bool:
    if not text or not (STUDY_COACH_TRIGGER and STUDY_COACH_SYSTEM_PROMPT):
        return False
    if STUDY_COACH_TRIGGER not in text:
        return False
    if STUDY_COACH_KEYWORDS and not any(kw in text for kw in STUDY_COACH_KEYWORDS):
        return False
    return True


async def _handle_study_coach_query(text: str) -> str:
    """Route the raw text through the study-coach persona (if configured)."""
    try:
        import notion_cu
        lessons = await asyncio.to_thread(notion_cu.get_this_week_lessons)
        raw = notion_cu.format_lessons_for_line(lessons)
    except Exception as e:  # noqa: BLE001
        raw = f"Notion fetch failed: {e}"
    return await format_with_study_coach_persona(raw)


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


async def chat_reply(text: str, line_message_id: str | None, agent: str | None = None) -> str:
    """Return a chat-style reply without persisting as an instruction."""
    if not agent:
        agent = DEFAULT_AGENT
    task_result = await _handle_task_update(text)
    if task_result is not None:
        return task_result

    week = _extract_week_number(text)
    if week is not None:
        return await _handle_lesson_week_query(week)
    if _is_study_coach_query(text):
        return await _handle_study_coach_query(text)
    if _is_lesson_query(text):
        return await _handle_lesson_query(text)
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
[{{"name": "<アプリ名>", "platform": "<iOS|Android|web|... 任意>", "status": "<許可ステータス>", "version": "<任意>"}}]

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


def _format_with_default_persona_sync(raw_result: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return raw_result
    owner = os.environ.get("OWNER_HANDLE", "the user")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=_build_system_prompt(),
            messages=[
                {"role": "user", "content": f"以下の実行結果を{owner}に報告してください。\n\n{raw_result}"}
            ],
        )
        text = "".join(getattr(b, "text", "") for b in response.content).strip()
        return text or raw_result
    except Exception:
        return raw_result


async def format_with_default_persona(raw_result: str) -> str:
    """Reshape a raw execution result through the default operator persona.
    Falls back to the raw text on any failure."""
    return await asyncio.to_thread(_format_with_default_persona_sync, raw_result)


def _format_with_study_coach_persona_sync(raw_result: str) -> str:
    """Study-coach persona formatting; inert (returns raw) unless configured."""
    if not STUDY_COACH_SYSTEM_PROMPT:
        return raw_result
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return raw_result
    owner = os.environ.get("OWNER_HANDLE", "the user")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=STUDY_COACH_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"以下の未視聴授業リストを{owner}に伝えてください。\n\n{raw_result}"}
            ],
        )
        text = "".join(getattr(b, "text", "") for b in response.content).strip()
        return text or raw_result
    except Exception:
        return raw_result


async def format_with_study_coach_persona(raw_result: str) -> str:
    """Reshape a lesson list via the optional study-coach persona (if configured).
    Falls back to the raw text on any failure or when unconfigured."""
    return await asyncio.to_thread(_format_with_study_coach_persona_sync, raw_result)
