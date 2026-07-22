#!/usr/bin/env python3
"""Migrate agent session markdown notes from Google Drive into the sessions table.

Place credentials.json (service account JSON or OAuth2 installed-app client JSON)
in the same directory before running:

    python3 migrate_drive_sessions.py
"""

import io
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials as UserCredentials
    from google_auth_oauthlib.flow import Flow
    from google.auth.transport.requests import Request
except ImportError as e:
    sys.exit(
        f"Missing Google API libraries: {e}\n"
        "Install with: pip install google-api-python-client google-auth google-auth-oauthlib"
    )

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "memory.db"
CREDENTIALS_PATH = PROJECT_DIR / "credentials.json"
TOKEN_PATH = PROJECT_DIR / "token.json"

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

AGENT_FOLDERS = {
    "agent_b":   "REDACTED_DRIVE_ID_2",
    "agent_c":   "REDACTED_DRIVE_ID_3",
    "agent_e":   "REDACTED_DRIVE_ID_5",
    "agent_d":   "REDACTED_DRIVE_ID_4",
    "agent_h":     "REDACTED_DRIVE_ID_6",
    "agent_a": "REDACTED_DRIVE_ID_1",
}

# Patterns tried in order. First match wins.
DATE_PATTERNS = [
    re.compile(r"(20\d{2})[-_./](\d{1,2})[-_./](\d{1,2})"),
    re.compile(r"(20\d{2})(\d{2})(\d{2})"),
]


def get_drive_service():
    if not CREDENTIALS_PATH.exists():
        sys.exit(
            f"credentials.json not found at {CREDENTIALS_PATH}\n"
            "Place a service account JSON or OAuth2 client JSON there first."
        )

    with open(CREDENTIALS_PATH) as f:
        creds_json = json.load(f)

    if creds_json.get("type") == "service_account":
        creds = service_account.Credentials.from_service_account_file(
            str(CREDENTIALS_PATH), scopes=SCOPES
        )
        print("[auth] using service account")
    else:
        creds = None
        if TOKEN_PATH.exists():
            creds = UserCredentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # OOB (out-of-band) flow: user opens the URL manually,
                # then pastes the resulting authorization code back in here.
                flow = Flow.from_client_secrets_file(
                    str(CREDENTIALS_PATH),
                    scopes=SCOPES,
                    redirect_uri="urn:ietf:wg:oauth:2.0:oob",
                )
                auth_url, _ = flow.authorization_url(
                    access_type="offline", prompt="consent"
                )
                print("\n以下のURLをブラウザで開いて認証してください:\n")
                print(auth_url)
                print()
                code = input("認証コードを入力してください: ").strip()
                flow.fetch_token(code=code)
                creds = flow.credentials
            TOKEN_PATH.write_text(creds.to_json())
        print("[auth] using OAuth2 user credentials")

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def ensure_schema(conn: sqlite3.Connection):
    cur = conn.execute("PRAGMA table_info(sessions)")
    cols = {row[1] for row in cur.fetchall()}
    if "source_filename" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN source_filename TEXT")
        print("[schema] added sessions.source_filename column")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_agent_source "
        "ON sessions(agent, source_filename)"
    )
    conn.commit()


def extract_date(filename: str, fallback: str) -> str:
    base = filename.rsplit(".", 1)[0]
    for pat in DATE_PATTERNS:
        m = pat.search(base)
        if m:
            y, mo, d = m.groups()
            try:
                return datetime(int(y), int(mo), int(d)).isoformat()
            except ValueError:
                continue
    return fallback


def list_md_files(drive, folder_id: str) -> list[dict]:
    files: list[dict] = []
    page_token = None
    while True:
        q = f"'{folder_id}' in parents and trashed = false"
        resp = (
            drive.files()
            .list(
                q=q,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                pageSize=200,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return [f for f in files if f["name"].lower().endswith(".md")]


def download_text(drive, file_id: str) -> str:
    request = drive.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue().decode("utf-8", errors="replace")


def main() -> int:
    drive = get_drive_service()
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    totals = {"inserted": 0, "skipped": 0, "errored": 0}

    for agent, folder_id in AGENT_FOLDERS.items():
        print(f"\n--- {agent} ({folder_id}) ---")
        try:
            files = list_md_files(drive, folder_id)
        except Exception as e:
            print(f"  ! failed to list folder: {e}")
            totals["errored"] += 1
            continue
        print(f"  found {len(files)} md file(s)")

        for f in files:
            name = f["name"]
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE agent = ? AND source_filename = ?",
                (agent, name),
            ).fetchone()
            if row:
                print(f"  · skip (already imported): {name}")
                totals["skipped"] += 1
                continue
            try:
                content = download_text(drive, f["id"])
            except Exception as e:
                print(f"  ! download failed for {name}: {e}")
                totals["errored"] += 1
                continue
            created_at = extract_date(
                name, f.get("modifiedTime", datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO sessions (agent, summary, created_at, source_filename) "
                "VALUES (?, ?, ?, ?)",
                (agent, content, created_at, name),
            )
            conn.commit()
            totals["inserted"] += 1
            print(f"  + {name} ({created_at[:10]})")

    conn.close()
    print(
        f"\n=== done. inserted={totals['inserted']} "
        f"skipped={totals['skipped']} errored={totals['errored']} ==="
    )
    return 0 if totals["errored"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
