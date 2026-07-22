"""Long-running scheduler: scan_and_execute hourly + morning report at 06:00 JST."""
import asyncio
import atexit
import os
import signal
import traceback
import time
from pathlib import Path

PIDFILE = Path(__file__).resolve().parent / "scheduler.pid"


def _acquire_pidfile():
    if PIDFILE.exists():
        try:
            old_pid = int(PIDFILE.read_text().strip())
            os.kill(old_pid, signal.SIGTERM)
            print(f"[scheduler] killed stale process {old_pid}", flush=True)
        except (ProcessLookupError, ValueError):
            pass
    PIDFILE.write_text(str(os.getpid()))
    atexit.register(lambda: PIDFILE.unlink(missing_ok=True))


_acquire_pidfile()

import schedule

from auto_executor import scan_and_execute, send_morning_report
from drive_sync import sync_all_agents
from maintenance import backup_db, rotate_logs
from pending_approver import approve_and_notify, notify_stale_pending
from slack_ingester import sync_slack_history


def _run(coro_fn):
    try:
        asyncio.run(coro_fn())
    except Exception as e:  # noqa: BLE001
        print(f"[scheduler] job failed: {e}", flush=True)
        traceback.print_exc()


def job_scan():
    print("[scheduler] running scan_and_execute", flush=True)
    _run(scan_and_execute)


def job_morning():
    print("[scheduler] running send_morning_report", flush=True)
    _run(send_morning_report)


def job_drive_sync():
    print("[scheduler] running drive_sync", flush=True)
    try:
        sync_all_agents()
    except Exception as e:  # noqa: BLE001
        print(f"[scheduler] drive_sync failed: {e}", flush=True)


def job_pending_approve():
    print("[scheduler] running pending_approve", flush=True)
    _run(approve_and_notify)


def job_stale_pending_notify():
    print("[scheduler] running notify_stale_pending", flush=True)
    _run(notify_stale_pending)


def job_backup_db():
    print("[scheduler] running backup_db", flush=True)
    try:
        backup_db()
    except Exception as e:  # noqa: BLE001
        print(f"[scheduler] backup_db failed: {e}", flush=True)
        traceback.print_exc()


def job_rotate_logs():
    print("[scheduler] running rotate_logs", flush=True)
    try:
        rotate_logs()
    except Exception as e:  # noqa: BLE001
        print(f"[scheduler] rotate_logs failed: {e}", flush=True)
        traceback.print_exc()


def job_slack_ingest():
    print("[scheduler] running slack_ingest", flush=True)
    _run(sync_slack_history)


schedule.every(1).minutes.do(job_scan)
schedule.every().day.at("06:00").do(job_morning)
schedule.every(10).minutes.do(job_drive_sync)
schedule.every(10).minutes.do(job_slack_ingest)
schedule.every(10).minutes.do(job_pending_approve)
schedule.every().day.at("07:00").do(job_stale_pending_notify)
schedule.every().day.at("03:00").do(job_backup_db)
schedule.every().day.at("03:30").do(job_rotate_logs)

print("[scheduler] started", flush=True)
while True:
    schedule.run_pending()
    time.sleep(30)
