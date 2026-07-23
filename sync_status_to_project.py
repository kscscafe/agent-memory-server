#!/usr/bin/env python3
"""Broadcast agent_status.md to every agent's Claude project.

The markdown is regenerated from memory.db by main._regenerate_agent_status_md
(single source of truth — same code path the LINE status-update flow uses),
then uploaded to every project listed in AGENT_PROJECTS, replacing any prior
file of the same name.

Run manually:
    python3 sync_status_to_project.py --session-key <KEY>

Session key resolution goes through session_key.get_session_key
(Keychain → --session-key CLI flag).
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from claude_pyrojects.api import ClaudeAPI

from main import (
    AGENT_CONTEXT_REMOTE_NAME,
    AGENT_STATUS_PATH,
    _regenerate_agent_context_md,
    _regenerate_agent_status_md,
)
from session_key import get_session_key
from upload_to_project import upload_file_to_project

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "memory.db"
OUT_PATH = AGENT_STATUS_PATH

# Files broadcast keeps in each project. Anything else is swept by cleanup.
KEEP_FILENAMES: set[str] = {
    "agent_status.md",
    AGENT_CONTEXT_REMOTE_NAME,
}

# Broadcast targets: agent name -> Claude Project UUID.
# Configure via the AGENT_PROJECTS env var as a JSON object, e.g.:
#   AGENT_PROJECTS='{"alice": "019d915d-...", "bob": "019d8c01-..."}'
# Empty (the default) disables broadcast — the script will exit early if no
# targets are configured.
AGENT_PROJECTS: dict[str, str] = json.loads(
    os.environ.get("AGENT_PROJECTS", "{}")
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-key",
                   default=None,
                   help="claude.ai sessionKey cookie "
                        "(used only if Keychain lookup returns no value)")
    p.add_argument("--only",
                   default=None,
                   help="only broadcast to the named agent "
                        "(matches an AGENT_PROJECTS key, e.g. 'agent_a')")
    return p.parse_args()


def cleanup_project(api: ClaudeAPI, project_id: str, agent_name: str) -> int:
    """Delete every doc in the project not on the KEEP_FILENAMES whitelist.

    Returns the number of files deleted. Logs a single-line summary in the
    format the operator-facing log expects.
    """
    files = api.list_files_in_project(project_id)
    to_delete = [
        f for f in files if f.get("file_name") not in KEEP_FILENAMES
    ]
    if not to_delete:
        print(f"[cleanup] nothing to clean in {agent_name}")
        return 0
    for f in to_delete:
        api.delete_file_from_project(
            project_id, f["uuid"], f.get("file_name", "")
        )
    print(f"[cleanup] deleted {len(to_delete)} file(s) from {agent_name}")
    return len(to_delete)


def main() -> int:
    args = parse_args()
    session_key = get_session_key(cli_value=args.session_key)
    if not session_key:
        sys.exit(
            "session key not found (checked Keychain service "
            "'claude-session-key' and --session-key CLI flag)"
        )

    if not DB_PATH.exists():
        sys.exit(f"DB not found: {DB_PATH}")

    if args.only:
        if args.only not in AGENT_PROJECTS:
            sys.exit(
                f"--only '{args.only}' not in AGENT_PROJECTS "
                f"(available: {', '.join(AGENT_PROJECTS)})"
            )
        targets = {args.only: AGENT_PROJECTS[args.only]}
    else:
        targets = AGENT_PROJECTS

    asyncio.run(_regenerate_agent_status_md())
    print(f"[generated] {OUT_PATH}")

    print(f"[broadcast] uploading to {len(targets)} project(s)…")
    print("[init] connecting to Claude.ai…")
    api = ClaudeAPI(session_key=session_key)
    print(f"[init] organization_id = {api.organization_id}")

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    for agent_name, project_id in targets.items():
        print(f"\n── {agent_name} ({project_id}) ──")
        try:
            cleanup_project(api, project_id, agent_name)
            upload_file_to_project(
                project_id=project_id,
                file_path=OUT_PATH,
                api=api,
            )
            context_path = asyncio.run(
                _regenerate_agent_context_md(agent_name)
            )
            print(f"[generated] {context_path}")
            upload_file_to_project(
                project_id=project_id,
                file_path=context_path,
                remote_name=AGENT_CONTEXT_REMOTE_NAME,
                api=api,
            )
            succeeded.append(agent_name)
        except Exception as e:
            print(f"[error] {agent_name}: {e}")
            failed.append((agent_name, str(e)))

    print("\n=== summary ===")
    print(f"  succeeded ({len(succeeded)}): {', '.join(succeeded) or '—'}")
    if failed:
        print(f"  failed    ({len(failed)}):")
        for name, err in failed:
            print(f"    - {name}: {err}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
