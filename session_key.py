"""Central resolver for the claude.ai session key.

Fallback order (spec: keychain first):
  1. macOS Keychain generic-password (service=claude-session-key, account=$USER)
  2. Environment variable CLAUDE_SESSION_KEY
  3. Explicit value from CLI (--session-key)

Callers should use `get_session_key(cli_value=args.session_key)` and let this
module decide which source wins. Do not read CLAUDE_SESSION_KEY directly.
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional

KEYCHAIN_SERVICE = "claude-session-key"


def _from_keychain() -> Optional[str]:
    user = os.environ.get("USER")
    if not user:
        try:
            user = os.getlogin()
        except OSError:
            return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password",
             "-a", user, "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _from_env() -> Optional[str]:
    value = os.environ.get("CLAUDE_SESSION_KEY", "").strip()
    return value or None


def get_session_key(cli_value: Optional[str] = None) -> Optional[str]:
    """Return the session key from the first available source, or None."""
    value = _from_keychain()
    if value:
        return value
    value = _from_env()
    if value:
        return value
    if cli_value:
        return cli_value.strip() or None
    return None
