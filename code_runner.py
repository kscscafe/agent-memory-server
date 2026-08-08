"""Route code-like instructions to the Claude Code CLI via subprocess.

`is_code_task(text)` decides whether an instruction needs the full Claude Code
agent (file edits, git ops, build/deploy) rather than a plain API call.
`run_claude_code(content, context)` shells out to `claude --print`.
"""
import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from claude_executor import format_with_default_persona

CODE_KEYWORDS = [
    # Japanese
    "コード", "実装", "修正して", "リファクタ", "リファクター",
    "デプロイ", "ビルド", "テスト追加", "テスト書",
    "コミット", "マージ", "リリース", "バグ修正",
    "リポジトリ", "ディレクトリ", "ファイル", "内容を知りたい",
    "調べて", "確認して", "一覧", "構成",
    # English / mixed (case-insensitive)
    "push", "build", "deploy", "fix", "implement", "refactor",
    "tests", "pull request", "commit", "merge", "release",
    "claude code",
]

# Case-sensitive keywords (uppercase tokens we want to keep distinct from
# lowercase substrings like "april", "expression")
CODE_KEYWORDS_EXACT = ["PR"]

DEFAULT_CWD = Path.home() / "Projects" / "agent-memory-server"
TIMEOUT_SEC = 15 * 60
FALLBACK_PATHS = ("/opt/homebrew/bin/claude", "/usr/local/bin/claude")


def is_code_task(content: str) -> bool:
    if not content:
        return False
    lower = content.lower()
    if any(kw.lower() in lower for kw in CODE_KEYWORDS):
        return True
    return any(kw in content for kw in CODE_KEYWORDS_EXACT)


def _find_claude_bin() -> Optional[str]:
    found = shutil.which("claude")
    if found:
        return found
    for p in FALLBACK_PATHS:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def _resolve_cwd(context: Optional[str]) -> Path:
    """If context starts with `cwd:` or is a real directory path, use it."""
    if context:
        cand = context.strip()
        if cand.startswith("cwd:"):
            cand = cand[4:].strip()
        p = Path(cand).expanduser()
        if p.is_dir():
            return p
    return DEFAULT_CWD


async def run_claude_code(content: str, context: Optional[str] = None) -> str:
    claude_bin = _find_claude_bin()
    if not claude_bin:
        return "ERROR: claude CLI が PATH に見つかりません"

    cwd = _resolve_cwd(context)

    # Ensure HOME / PATH are set so claude can read its auth + helpers
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env.setdefault("HOME", str(Path.home()))
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + env.get("PATH", "")

    add_dir = str(Path.home() / "Projects")
    cmd = [claude_bin, "--add-dir", add_dir, "--print", content]
    logging.info(f"[code_runner] 起動: {content[:50]}")
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=env,
        )
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=TIMEOUT_SEC
        )
        logging.info(f"[code_runner] 完了: returncode={proc.returncode}")
    except asyncio.TimeoutError:
        return f"ERROR: claude CLI timeout ({TIMEOUT_SEC}s)"
    except Exception as e:  # noqa: BLE001
        logging.error(f"[code_runner] エラー: {e}")
        return f"ERROR: subprocess failed: {e}"

    stdout = stdout_b.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        stderr = stderr_b.decode("utf-8", errors="replace").strip()
        raw = (
            f"ERROR (exit {proc.returncode})\n"
            f"stderr: {stderr[:800]}\n"
            f"stdout: {stdout[:800]}"
        )
    else:
        raw = stdout or "(empty output)"

    return await format_with_default_persona(raw)
