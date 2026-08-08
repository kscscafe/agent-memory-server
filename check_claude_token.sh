#!/bin/bash
set -euo pipefail

# Resolve the repo root from the script location so this file is portable
# (no assumption about where the checkout lives on disk).
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

CREDS="$HOME/.claude/.credentials.json"
ENV_FILE="$SCRIPT_DIR/.env"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/check_claude_token.log"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

if [ ! -f "$CREDS" ]; then
  log "ERROR: credentials file not found at $CREDS"
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  log "ERROR: env file not found at $ENV_FILE"
  exit 1
fi

SLACK_BOT_TOKEN=$(grep -E '^SLACK_BOT_TOKEN=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")
if [ -z "$SLACK_BOT_TOKEN" ]; then
  log "ERROR: SLACK_BOT_TOKEN not found in $ENV_FILE"
  exit 1
fi

# Slack channel that receives the token-expiry alert. Read from env or from
# .env (same pattern as SLACK_BOT_TOKEN). Never hardcoded — see .env.example.
SLACK_ALERT_CHANNEL="${SLACK_ALERT_CHANNEL:-}"
if [ -z "$SLACK_ALERT_CHANNEL" ]; then
  SLACK_ALERT_CHANNEL=$(grep -E '^SLACK_ALERT_CHANNEL=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")
fi
if [ -z "$SLACK_ALERT_CHANNEL" ]; then
  log "ERROR: SLACK_ALERT_CHANNEL not set (env var or in $ENV_FILE)"
  exit 1
fi

EXPIRES_MS=$(/usr/bin/python3 -c "import json; print(json.load(open('$CREDS'))['claudeAiOauth']['expiresAt'])")
NOW_MS=$(/usr/bin/python3 -c "import time; print(int(time.time()*1000))")
REMAIN_MS=$((EXPIRES_MS - NOW_MS))
REMAIN_HOURS=$((REMAIN_MS / 1000 / 3600))

log "Token expires in ${REMAIN_HOURS}h (expiresAt=$EXPIRES_MS)"

if [ "$REMAIN_HOURS" -ge 24 ]; then
  log "OK: more than 24h remaining, no notification needed"
  exit 0
fi

MESSAGE="⚠️ Claude Codeトークンの期限が24時間以内です。/login を実行してください。 (残り約${REMAIN_HOURS}時間)"

RESPONSE=$(curl -sS -X POST https://slack.com/api/chat.postMessage \
  -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  --data "$(/usr/bin/python3 -c "import json,sys; print(json.dumps({'channel':'$SLACK_ALERT_CHANNEL','text':'''$MESSAGE'''}))")")

OK=$(echo "$RESPONSE" | /usr/bin/python3 -c "import json,sys; print(json.load(sys.stdin).get('ok', False))")

if [ "$OK" = "True" ]; then
  log "Slack notification sent (${REMAIN_HOURS}h remaining)"
else
  log "ERROR: Slack notification failed: $RESPONSE"
  exit 1
fi
