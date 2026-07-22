"""Slack Web API client — push text DMs to the configured user. Used alongside
LINE so notifications keep landing while LINE access is unavailable."""
import os
from pathlib import Path

from dotenv import load_dotenv
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError

# Load credentials from the project .env. Pinning the path makes this work
# when the module is imported from a script run in a different cwd; an already-
# loaded env (e.g. main.py called load_dotenv first) is not overwritten.
load_dotenv(Path(__file__).resolve().parent / ".env")

MAX_TEXT_LEN = 3900


def _credentials() -> tuple[str, str]:
    return (
        os.environ.get("SLACK_BOT_TOKEN", ""),
        os.environ.get("SLACK_USER_ID", ""),
    )


async def send_slack_message(text: str) -> dict:
    """DM `text` to the configured user via chat.postMessage. Returns {} on any failure."""
    token, user_id = _credentials()
    if not token or not user_id:
        print("[slack_client] credentials not configured; skipping push")
        return {}
    client = AsyncWebClient(token=token)
    try:
        resp = await client.chat_postMessage(channel=user_id, text=text[:MAX_TEXT_LEN])
        return resp.data if hasattr(resp, "data") else {}
    except SlackApiError as e:
        print(f"[slack_client] push failed status={e.response.status_code} body={e.response.data}")
        return {}
    except Exception as e:  # noqa: BLE001
        print(f"[slack_client] push failed: {e}")
        return {}
