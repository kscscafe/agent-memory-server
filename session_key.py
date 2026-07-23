"""Central resolver for the claude.ai session key.

Resolution order:
  1. macOS Keychain generic-password (service=claude-session-key, account=$USER)
  2. Explicit value from CLI (--session-key), used as a manual override
     when no Keychain entry is available

The environment-variable fallback (CLAUDE_SESSION_KEY) was removed on
2026-07-23 to eliminate silent plaintext fallbacks in launchd plists.
Keychain failures now print the underlying `security` error to stderr
instead of returning None silently.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

KEYCHAIN_SERVICE = "claude-session-key"


def _from_keychain() -> Optional[str]:
    user = os.environ.get("USER")
    if not user:
        try:
            user = os.getlogin()
        except OSError:
            print(
                "[session_key] cannot determine current user "
                "(USER unset, getlogin failed)",
                file=sys.stderr,
            )
            return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password",
             "-a", user, "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[session_key] security invocation failed: {e}", file=sys.stderr)
        return None
    if result.returncode != 0:
        err = result.stderr.strip() or f"exit code {result.returncode}"
        print(
            f"[session_key] Keychain read failed "
            f"(service={KEYCHAIN_SERVICE!r}, account={user!r}): {err}",
            file=sys.stderr,
        )
        return None
    value = result.stdout.strip()
    return value or None


def get_session_key(cli_value: Optional[str] = None) -> Optional[str]:
    """Return the session key from Keychain, or the CLI override if given.

    Returns None if neither source yields a value. Callers decide whether to
    abort (sys.exit) or skip (best-effort hooks).
    """
    value = _from_keychain()
    if value:
        return value
    if cli_value:
        return cli_value.strip() or None
    return None
