"""Daily maintenance: SQLite online backup + log rotation.

Run from scheduler.py at 03:00 (backup) and 03:30 (rotate).
"""
from __future__ import annotations

import gzip
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "memory.db"
BACKUP_DIR = BASE / "backups"
ARCHIVE_DIR = BASE / "logs_archive"
RETAIN_DAYS = 7

LOG_FILES = (
    "server.log",
    "mcp_server.log",
    "scheduler.log",
    "cloudflared.log",
    "sync_status.log",
)


def _prune_by_mtime(directory: Path, pattern: str, keep_days: int) -> int:
    cutoff = (datetime.now() - timedelta(days=keep_days)).timestamp()
    removed = 0
    for f in directory.glob(pattern):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            removed += 1
    return removed


def backup_db() -> Path:
    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    dest = BACKUP_DIR / f"memory.db.{stamp}"
    src = sqlite3.connect(DB_PATH)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    removed = _prune_by_mtime(BACKUP_DIR, "memory.db.*", RETAIN_DAYS)
    size_mb = dest.stat().st_size / 1024 / 1024
    print(
        f"[maintenance] db backup -> {dest.name} ({size_mb:.2f} MB), "
        f"pruned {removed} old",
        flush=True,
    )
    return dest


def rotate_logs() -> int:
    """Gzip current logs to dated archives, then truncate originals.

    Why truncate, not rm: launchd opens stdout/stderr with O_APPEND, so
    writes always seek to EOF. Truncating the file makes the next write
    land at byte 0 of the now-empty file. Deleting or renaming would
    orphan the launchd FD (it'd still point to the unlinked inode).
    """
    ARCHIVE_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    rotated = 0
    for name in LOG_FILES:
        src = BASE / name
        if not src.exists() or src.stat().st_size == 0:
            continue
        archive = ARCHIVE_DIR / f"{name}.{stamp}.gz"
        with src.open("rb") as fin, gzip.open(archive, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        with src.open("w"):
            pass
        rotated += 1
    removed = _prune_by_mtime(ARCHIVE_DIR, "*.gz", RETAIN_DAYS)
    print(
        f"[maintenance] rotated {rotated} log file(s), pruned {removed} old",
        flush=True,
    )
    return rotated


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "both"
    if cmd in ("backup", "both"):
        backup_db()
    if cmd in ("rotate", "both"):
        rotate_logs()
