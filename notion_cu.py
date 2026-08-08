"""Notion API client for the optional lesson-tracker database.

Reads `NOTION_TOKEN` and `NOTION_LESSON_DATABASE_ID` from env. Module is named
`notion_cu` (not `notion_client`) to avoid shadowing the upstream package —
the historical `cu` suffix is retained for import compatibility only.
"""
import os
from datetime import date, datetime, timedelta
from typing import Optional

from notion_client import Client

STATUS_PROP = "ステータス"
DUE_PROP = "終了日"
WEEK_PROP = "回"
SUBJECT_PROP = "科目"
PENDING_VALUE = "未視聴"


def _client() -> Optional[Client]:
    token = os.environ.get("NOTION_TOKEN", "").strip()
    if not token:
        return None
    return Client(auth=token)


def _db_id() -> str:
    # Prefer the new NOTION_LESSON_DATABASE_ID; keep NOTION_CU_DATABASE_ID as
    # a legacy alias so an existing operator's .env keeps working.
    return (
        os.environ.get("NOTION_LESSON_DATABASE_ID", "")
        or os.environ.get("NOTION_CU_DATABASE_ID", "")
    ).strip()


_DATA_SOURCE_ID_CACHE: dict[str, str] = {}


def _data_source_id(client: Client, db_id: str) -> str:
    """Resolve the data_source_id for a database (2025-09-03 Notion API)."""
    cached = _DATA_SOURCE_ID_CACHE.get(db_id)
    if cached:
        return cached
    db = client.databases.retrieve(database_id=db_id)
    sources = db.get("data_sources") or []
    if not sources:
        raise RuntimeError(f"database {db_id} has no data_sources")
    ds_id = sources[0]["id"]
    _DATA_SOURCE_ID_CACHE[db_id] = ds_id
    return ds_id


def _title_property_name(client: Client, db_id: str) -> str:
    """Find the title property name by reading the data source schema."""
    ds_id = _data_source_id(client, db_id)
    ds = client.data_sources.retrieve(data_source_id=ds_id)
    for name, prop in (ds.get("properties") or {}).items():
        if prop.get("type") == "title":
            return name
    return "名前"


def _extract_title(page: dict, title_prop: str) -> str:
    prop = page.get("properties", {}).get(title_prop, {})
    parts = prop.get("title", []) or []
    return "".join(p.get("plain_text", "") for p in parts).strip() or "(無題)"


def _extract_due(page: dict) -> Optional[str]:
    prop = page.get("properties", {}).get(DUE_PROP, {})
    d = prop.get("date") or {}
    return d.get("start")


def _extract_status(page: dict) -> str:
    prop = page.get("properties", {}).get(STATUS_PROP, {})
    if prop.get("type") == "status":
        return (prop.get("status") or {}).get("name", "") or ""
    if prop.get("type") == "select":
        return (prop.get("select") or {}).get("name", "") or ""
    return ""


def _query_pending(client: Client, db_id: str, target_date: Optional[date]) -> list[dict]:
    """Query the CU data source for pending lessons, optionally bounded by due date.

    Tries `status` first (newer Notion type); falls back to `select` on schema
    mismatch so it works whether ステータス is a status or select property.
    """
    ds_id = _data_source_id(client, db_id)
    filters: list[dict] = []
    if target_date:
        filters.append({
            "property": DUE_PROP,
            "date": {"on_or_before": target_date.isoformat()},
        })

    def run(status_type: str) -> list[dict]:
        status_filter = {"property": STATUS_PROP, status_type: {"equals": PENDING_VALUE}}
        and_clauses = [status_filter] + filters
        query_filter = and_clauses[0] if len(and_clauses) == 1 else {"and": and_clauses}
        results: list[dict] = []
        cursor: Optional[str] = None
        while True:
            payload = {"data_source_id": ds_id, "filter": query_filter, "page_size": 100}
            if cursor:
                payload["start_cursor"] = cursor
            resp = client.data_sources.query(**payload)
            results.extend(resp.get("results", []))
            if not resp.get("has_more"):
                break
            cursor = resp.get("next_cursor")
        return results

    # CU DB の ステータス は select 型と確認済み。select を先に試して警告ノイズを抑える
    try:
        return run("select")
    except Exception:
        return run("status")


def get_pending_lessons(target_date: Optional[date] = None) -> list[dict]:
    """ステータスが「未視聴」のレコードを取得。target_date 指定時は 終了日 <= target_date に絞る。"""
    client = _client()
    db_id = _db_id()
    if not client or not db_id:
        raise RuntimeError("NOTION_TOKEN or NOTION_CU_DATABASE_ID が未設定")
    title_prop = _title_property_name(client, db_id)
    pages = _query_pending(client, db_id, target_date)
    rows: list[dict] = []
    for p in pages:
        rows.append({
            "title": _extract_title(p, title_prop),
            "due": _extract_due(p),
            "status": _extract_status(p),
            "url": p.get("url", ""),
        })
    rows.sort(key=lambda r: (r["due"] or "9999-12-31", r["title"]))
    return rows


def _end_of_this_week(today: Optional[date] = None) -> date:
    """Return this week's Sunday (week ends Sunday). If today is Sunday, returns today."""
    today = today or date.today()
    # Monday=0 ... Sunday=6 → days until Sunday
    days_until_sunday = (6 - today.weekday()) % 7
    return today + timedelta(days=days_until_sunday)


def get_this_week_lessons() -> list[dict]:
    """終了日が今週末（日曜日）以前 かつ ステータスが「未視聴」のレコード。"""
    return get_pending_lessons(target_date=_end_of_this_week())


def _format_due(due: Optional[str]) -> str:
    if not due:
        return "締め切り未設定"
    try:
        d = datetime.fromisoformat(due).date() if "T" in due else date.fromisoformat(due)
        return f"締め切り：{d.month}/{d.day}"
    except Exception:
        return f"締め切り：{due}"


def _extract_subject(page: dict) -> str:
    prop = page.get("properties", {}).get(SUBJECT_PROP, {})
    if prop.get("type") == "select":
        return (prop.get("select") or {}).get("name", "") or ""
    if prop.get("type") == "multi_select":
        opts = prop.get("multi_select") or []
        return "・".join(o.get("name", "") for o in opts)
    return ""


def _extract_week(page: dict) -> Optional[int]:
    prop = page.get("properties", {}).get(WEEK_PROP, {})
    n = prop.get("number")
    return int(n) if n is not None else None


def get_lessons_by_week(week_number: int) -> list[dict]:
    """指定した回数（week_number）のレコードを全科目分取得。ステータス問わず全件返す。"""
    client = _client()
    db_id = _db_id()
    if not client or not db_id:
        raise RuntimeError("NOTION_TOKEN or NOTION_CU_DATABASE_ID が未設定")
    title_prop = _title_property_name(client, db_id)
    ds_id = _data_source_id(client, db_id)
    query_filter = {"property": WEEK_PROP, "number": {"equals": week_number}}
    pages: list[dict] = []
    cursor: Optional[str] = None
    while True:
        payload = {"data_source_id": ds_id, "filter": query_filter, "page_size": 100}
        if cursor:
            payload["start_cursor"] = cursor
        resp = client.data_sources.query(**payload)
        pages.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")
    rows: list[dict] = []
    for p in pages:
        rows.append({
            "title": _extract_title(p, title_prop),
            "subject": _extract_subject(p),
            "week": _extract_week(p),
            "due": _extract_due(p),
            "status": _extract_status(p),
            "url": p.get("url", ""),
        })
    # 未視聴を先に、その後 科目名 昇順
    rows.sort(key=lambda r: (0 if r["status"] == PENDING_VALUE else 1, r.get("subject") or "", r["title"]))
    return rows


def format_lessons_by_week(lessons: list[dict]) -> str:
    """週別レコードをLINEで読みやすい形式に整形する。"""
    if not lessons:
        return "📚 該当する授業がありません"
    week = next((r.get("week") for r in lessons if r.get("week") is not None), None)
    header = f"📚 第{week}回 授業一覧" if week is not None else "📚 授業一覧"
    lines = [header]
    for r in lessons:
        name = r.get("subject") or r.get("title") or "(無題)"
        status = r.get("status") or ""
        if status == PENDING_VALUE:
            lines.append(f"⬜ {name}（未視聴・{_format_due(r.get('due'))}）")
        else:
            label = status or "視聴済"
            lines.append(f"✅ {name}（{label}）")
    return "\n".join(lines)


def get_current_week_summary() -> Optional[dict]:
    """未視聴レコードの中で最小の「回」をその週として返す。

    Returns {"week": int, "pending_count": int, "due": str|None} or None if 0件.
    due は同じ週内の最も遅い日付（= 締め切り）。
    """
    client = _client()
    db_id = _db_id()
    if not client or not db_id:
        return None
    ds_id = _data_source_id(client, db_id)
    pages = _query_pending(client, db_id, target_date=None)
    rows = [{"week": _extract_week(p), "due": _extract_due(p)} for p in pages]
    week_rows = [r for r in rows if r["week"] is not None]
    if not week_rows:
        return None
    current_week = min(r["week"] for r in week_rows)
    same_week = [r for r in week_rows if r["week"] == current_week]
    dues = [r["due"] for r in same_week if r.get("due")]
    due = max(dues) if dues else None
    return {"week": current_week, "pending_count": len(same_week), "due": due}


def format_current_week_summary(summary: Optional[dict]) -> str:
    """`📚 <label>：第X週 残りY科目（期限：MM/DD）` 形式。0件/None なら空文字。
    Label prefix is configured via LESSON_TRACKER_LABEL env (empty → 'Lessons')."""
    if not summary or not summary.get("pending_count"):
        return ""
    week = summary["week"]
    count = summary["pending_count"]
    due = summary.get("due")
    label = os.environ.get("LESSON_TRACKER_LABEL", "").strip() or "Lessons"
    base = f"📚 {label}：第{week}週 残り{count}科目"
    if not due:
        return base
    try:
        d = date.fromisoformat(due.split("T")[0])
        return f"{base}（期限：{d.month}/{d.day}）"
    except Exception:
        return base


def format_lessons_for_line(lessons: list[dict], header: str = "📚 今週の未視聴授業") -> str:
    if not lessons:
        return f"{header}\n（該当なし）"
    lines = [header]
    for r in lessons:
        lines.append(f"・{r['title']}（{_format_due(r.get('due'))}）")
    return "\n".join(lines)
