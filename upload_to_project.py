#!/usr/bin/env python3
"""Upload (or update) a single file to an existing Claude.ai project.

If a file with the same name already exists in the project, it is deleted
first and re-uploaded — this matches claude-pyrojects' update semantics,
since the Claude internal API has no native update endpoint.

Usage:
    python3 upload_to_project.py \\
        --project-id <PROJECT_UUID> \\
        --file-path  <PATH_TO_LOCAL_FILE> \\
        --session-key <CLAUDE_SESSION_KEY>

The session key is the value of the `sessionKey` cookie set by claude.ai
in your browser after you log in (DevTools → Application → Cookies).
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from claude_pyrojects.api import ClaudeAPI
except ImportError as e:
    sys.exit(
        f"claude-pyrojects not importable ({e}).\n"
        "Install with: pip install claude-pyrojects"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload or update a file in an existing Claude.ai project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project-id", required=True,
                   help="UUID of the target Claude project")
    p.add_argument("--file-path", required=True,
                   help="Local path to the file to upload")
    p.add_argument("--session-key", required=True,
                   help="sessionKey cookie from claude.ai")
    p.add_argument("--remote-name",
                   help="File name to use inside the project "
                        "(default: basename of --file-path)")
    return p.parse_args()


def upload_file_to_project(
    project_id: str,
    file_path: str | os.PathLike,
    session_key: str | None = None,
    remote_name: str | None = None,
    verbose: bool = True,
    api: ClaudeAPI | None = None,
) -> dict:
    """Upload (or replace if same name exists) a file in an existing Claude project.

    Pass `api` to reuse an existing ClaudeAPI instance across many calls and
    avoid re-fetching the organization id; otherwise `session_key` is used to
    build a fresh one.

    Returns the JSON response from the add call (contains the new file's uuid).
    Raises RuntimeError on any failure.
    """
    fp = Path(file_path).expanduser()
    if not fp.is_file():
        raise RuntimeError(f"file not found: {fp}")

    name = remote_name or fp.name
    content = fp.read_text(encoding="utf-8", errors="replace")

    def log(msg: str):
        if verbose:
            print(msg)

    if api is None:
        if not session_key:
            raise RuntimeError("either api or session_key must be provided")
        log("[init] connecting to Claude.ai…")
        api = ClaudeAPI(session_key=session_key)
        log(f"[init] organization_id = {api.organization_id}")

    try:
        existing = api.list_files_in_project(project_id)
    except Exception as e:
        raise RuntimeError(
            f"failed to list project docs (check --project-id and "
            f"--session-key): {e}"
        ) from e

    already = any(f.get("file_name") == name for f in existing)
    action = "update" if already else "add"
    log(f"[upload] {action}: {name} "
        f"({len(content)} chars, {fp.stat().st_size} bytes on disk)")

    try:
        result = api.update_file_in_project(
            project_uuid=project_id,
            file_path=name,
            content=content,
        )
    except Exception as e:
        raise RuntimeError(f"upload failed: {e}") from e

    file_uuid = result.get("uuid") if isinstance(result, dict) else None
    log(f"[done] {action} OK"
        + (f" — new file uuid: {file_uuid}" if file_uuid else ""))
    return result if isinstance(result, dict) else {}


def main() -> int:
    args = parse_args()
    try:
        upload_file_to_project(
            project_id=args.project_id,
            file_path=args.file_path,
            session_key=args.session_key,
            remote_name=args.remote_name,
        )
    except RuntimeError as e:
        sys.exit(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
