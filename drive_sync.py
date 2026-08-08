"""Pull per-agent .md notes from Google Drive into the memory layer.

Each agent has a Drive folder configured via the ``AGENT_DRIVE_FOLDERS`` env
var (JSON: ``{"<display-name>": "<folder-id>", ...}``). Every 10 minutes the
scheduler calls sync_all_agents(). For each new .md file (deduped by
Drive file_id, NOT by name) we:

- name contains 「まとめ」 → episodic_memories (summary=full text)
- name contains 「TASKS」  → tasks table (one row per non-empty bullet line)
- otherwise              → episodic_memories

After any successful imports we regenerate agent_status.md.

If ``AGENT_DRIVE_FOLDERS`` is empty (or invalid JSON) sync_all_agents() is a
no-op — a fresh install can leave the env var unset. If service_account.json
is missing, sync_all_agents() logs and returns 0 without touching anything.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "memory.db"
SERVICE_ACCOUNT_PATH = PROJECT_DIR / "service_account.json"

try:
    AGENT_FOLDERS: dict[str, str] = json.loads(
        os.environ.get("AGENT_DRIVE_FOLDERS", "{}") or "{}"
    )
    if not isinstance(AGENT_FOLDERS, dict):
        AGENT_FOLDERS = {}
except json.JSONDecodeError:
    AGENT_FOLDERS = {}

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

DATE_RE = re.compile(r"(20\d{2})[-_/.]?(\d{2})[-_/.]?(\d{2})")
BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+(.*\S)\s*$")


def _log(msg: str) -> None:
    print(f"[drive-sync] {msg}", flush=True)


def _notify_slack(text: str) -> None:
    """Fire-and-forget Slack DM. Never raises."""
    try:
        from slack_client import send_slack_message
        asyncio.run(send_slack_message(text))
    except Exception as e:  # noqa: BLE001
        _log(f"slack notify failed: {e}")


def _build_drive_service():
    """Build a Drive v3 client. Returns None if creds or libs are missing."""
    if not SERVICE_ACCOUNT_PATH.exists():
        _log(f"service account JSON not found at {SERVICE_ACCOUNT_PATH} — skipping")
        return None
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError as e:
        _log(f"google-api-python-client/google-auth not installed: {e}")
        return None
    creds = service_account.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_PATH), scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _list_md_files(service, folder_id: str) -> list[dict]:
    """Return [{id, name}] for every .md file in the folder (non-recursive)."""
    q = (
        f"'{folder_id}' in parents and trashed = false "
        "and (mimeType = 'text/markdown' or name contains '.md')"
    )
    files: list[dict] = []
    page_token: str | None = None
    while True:
        resp = service.files().list(
            q=q,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
            pageSize=100,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        for f in resp.get("files", []):
            name = f.get("name") or ""
            if not name.lower().endswith(".md"):
                continue
            files.append({"id": f["id"], "name": name})
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def _download_text(service, file_id: str) -> str:
    """Read a Drive file as UTF-8 text. Falls back via export for Google Docs."""
    try:
        data = service.files().get_media(
            fileId=file_id, supportsAllDrives=True
        ).execute()
    except Exception as e:  # noqa: BLE001
        _log(f"get_media failed for {file_id}: {e} — trying export")
        try:
            data = service.files().export(
                fileId=file_id, mimeType="text/plain"
            ).execute()
        except Exception as e2:  # noqa: BLE001
            _log(f"export also failed for {file_id}: {e2}")
            return ""
    if isinstance(data, bytes):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")
    return str(data)


def _extract_session_date(name: str) -> str:
    """Pull a YYYY-MM-DD from the filename, else today's date."""
    m = DATE_RE.search(name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return datetime.now().strftime("%Y-%m-%d")


def _topics_from_name(name: str) -> str:
    """Strip extension + date and use the rest as a comma-friendly topic blob."""
    stem = name.rsplit(".", 1)[0]
    stem = DATE_RE.sub("", stem).strip(" _-")
    return stem or name


def _parse_task_lines(text: str) -> list[str]:
    """Pick bullet/numbered lines out of a TASKS markdown file."""
    tasks: list[str] = []
    for line in text.splitlines():
        m = BULLET_RE.match(line)
        if m:
            content = m.group(1).strip()
            if content:
                tasks.append(content)
    return tasks


def _already_synced(conn: sqlite3.Connection, file_id: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM drive_sync_log WHERE file_id = ?", (file_id,)
    )
    return cur.fetchone() is not None


def _record_sync(conn: sqlite3.Connection, file_id: str, name: str, agent: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO drive_sync_log (file_id, file_name, agent) "
        "VALUES (?, ?, ?)",
        (file_id, name, agent),
    )


def _save_episodic(
    conn: sqlite3.Connection, agent: str, name: str, text: str
) -> None:
    conn.execute(
        """
        INSERT INTO episodic_memories (agent, summary, topics, session_date)
        VALUES (?, ?, ?, ?)
        """,
        (agent, text, _topics_from_name(name), _extract_session_date(name)),
    )
    try:
        from candidates_parser import extract_and_queue
        n = extract_and_queue(text, source=f"drive_sync:{agent}:{name}")
        if n:
            _log(f"{agent}: extracted {n} ams:candidates from {name}")
    except Exception as e:  # noqa: BLE001
        _log(f"{agent}: candidates parse failed for {name}: {e}")


def _save_tasks(
    conn: sqlite3.Connection, agent: str, text: str
) -> int:
    lines = _parse_task_lines(text)
    if not lines:
        return 0
    ts = datetime.utcnow().isoformat()
    for content in lines:
        conn.execute(
            """
            INSERT INTO tasks (agent, content, status, created_at, updated_at,
                               progress, next_action)
            VALUES (?, ?, 'open', ?, ?, 0, NULL)
            """,
            (agent, content, ts, ts),
        )
    return len(lines)


def _sync_one_agent(
    service, agent: str, folder_id: str, error_state: dict | None = None
) -> int:
    """Sync one agent's folder. Returns number of newly imported Drive files.

    `error_state["first_error_notified"]` gates Slack notification on list
    failure so only the first agent in a run pushes a DM (others log only).
    """
    try:
        files = _list_md_files(service, folder_id)
    except Exception as e:  # noqa: BLE001
        _log(f"{agent}: list failed: {e}")
        if error_state is not None and not error_state.get("first_error_notified"):
            error_state["first_error_notified"] = True
            _notify_slack(
                f"⚠️ drive_sync: {agent} の Drive list 失敗（Google API接続エラー）: {e}"
            )
        return 0

    imported = 0
    conn = sqlite3.connect(DB_PATH)
    try:
        for f in files:
            file_id = f["id"]
            name = f["name"]
            if _already_synced(conn, file_id):
                continue
            try:
                text = _download_text(service, file_id)
            except Exception as e:  # noqa: BLE001
                _log(f"{agent}: download failed for {name} ({file_id}): {e}")
                continue
            if not text:
                _log(f"{agent}: empty body for {name} ({file_id}) — skipping")
                continue
            try:
                if "まとめ" in name:
                    _save_episodic(conn, agent, name, text)
                elif "TASKS" in name:
                    n = _save_tasks(conn, agent, text)
                    if n == 0:
                        _log(f"{agent}: TASKS file {name} had no bullet lines")
                else:
                    _save_episodic(conn, agent, name, text)
                _record_sync(conn, file_id, name, agent)
                conn.commit()
                imported += 1
                _log(f"{agent}: imported {name} ({file_id})")
            except Exception as e:  # noqa: BLE001
                conn.rollback()
                _log(f"{agent}: insert failed for {name}: {e}")
    finally:
        conn.close()
    return imported


def sync_all_agents() -> int:
    """Entry point for the scheduler. Returns total files imported.

    Wrapped in broad try/except per agent — one agent's failure (auth,
    permissions, network) must not block the others.
    """
    if not AGENT_FOLDERS:
        _log("AGENT_DRIVE_FOLDERS is empty — drive sync disabled")
        return 0

    service = _build_drive_service()
    if service is None:
        _notify_slack(
            "⚠️ drive_sync: Google Drive サービス初期化に失敗（service_account.json 欠落 or ライブラリ未インストール）"
        )
        return 0

    error_state = {"first_error_notified": False}
    total = 0
    for agent, folder_id in AGENT_FOLDERS.items():
        if folder_id == "TBD":
            _log(f"{agent}: folder_id is TBD — skipping")
            continue
        try:
            total += _sync_one_agent(service, agent, folder_id, error_state)
        except Exception as e:  # noqa: BLE001
            _log(f"{agent}: unexpected failure: {e}")

    if total > 0:
        try:
            from main import _regenerate_agent_status_md  # local import to avoid cycles at import time
            asyncio.run(_regenerate_agent_status_md())
            _log("regenerated agent_status.md")
        except Exception as e:  # noqa: BLE001
            _log(f"agent_status.md regenerate failed: {e}")

    _log(f"sync_all_agents complete: imported {total} file(s)")
    return total


if __name__ == "__main__":
    sync_all_agents()
