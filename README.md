# agent-memory-server

A local memory & orchestration server for multi-agent AI workflows. Runs as a
FastAPI service backed by SQLite (with FTS5 + `sqlite-vec` for hybrid search)
and exposes both a REST API and an MCP (Model Context Protocol) endpoint that
Claude Desktop, Claude Code, or Claude.ai Web can query directly.

## What it provides

- **Semantic / procedural / episodic memory tables** with hybrid retrieval
  (FTS5 full-text + vector search fused via Reciprocal Rank Fusion).
- **REST API** (`main.py`, default port 8000) for storing/searching memories,
  tracking instructions, and running an inbox workflow.
- **MCP server** (`mcp_server.py`, default port 8001) exposing memory tools
  (`search_memory`, `get_memory`, `list_memories`, `get_context`, `save_memory`,
  `save_candidate`, `delete_memory`, `inbox_*`) with OAuth 2.1 + Bearer-key
  authentication.
- **Optional integrations**: LINE webhook receiver, Slack webhook + Web API
  client, Notion database sync, Google Drive session upload — each stays dark
  unless the corresponding env vars are set.
- **Scheduler** (`scheduler.py`) and **executors** (`auto_executor.py`,
  `claude_executor.py`) for scheduled or auto-run instruction pipelines.

## Repository layout

```
main.py                          # FastAPI app (REST, webhooks, instruction pipeline)
memory_api.py                    # /memory router — semantic/procedural/episodic tables
mcp_server.py                    # MCP server (port 8001) with OAuth 2.1
inbox_api.py                     # Inbox endpoints
vector_store.py                  # sqlite-vec loader + embedding upsert/search
embeddings.py                    # Embedding provider wrapper
candidates_parser.py             # Parses <ams:candidates> blocks from LLM output
pending_approver.py              # Approval workflow for pending memory candidates
scheduler.py                     # Cron-style scheduler entry point
auto_executor.py                 # Auto-executes queued instructions
claude_executor.py               # Anthropic Claude call wrapper (prompt-driven)
line_client.py                   # LINE Messaging API push client
slack_client.py                  # Slack chat.postMessage client
slack_ingester.py                # Ingests Slack DM history into episodic memory
notion_cu.py                     # Notion database read/sync
drive_sync.py                    # Google Drive session upload
upload_to_project.py             # Utility: upload files to Claude Projects
migrate_add_vectors.py           # One-off migration: add vector columns
migrate_drive_sessions.py        # One-off migration: Drive session table
session_key.py                   # Rotating session-key generator
sync_status_to_project.py        # Push status snapshot to a Claude Project
maintenance.py                   # DB maintenance / integrity checks
test_inbox.py                    # Inbox smoke tests
test_ownership_and_validation.py # Ownership & validation smoke tests
check_claude_token.sh            # Diagnostic script for MCP OAuth tokens
prompts/                         # Agent prompt templates (*.example.md tracked;
                                 #   drop <name>.md to override locally)
requirements.txt
.env.example                     # Copy to .env and fill in
config.example.yaml              # Copy to config.yaml to override non-secret settings
```

## Prerequisites

- Python 3.11+
- macOS or Linux (Windows untested)
- SQLite 3.38+ (for FTS5). `sqlite-vec` is installed via pip.
- Optional: `cloudflared` or `tailscale` if you want to expose the MCP endpoint
  publicly.

## Setup

### 1. Clone and install

```sh
git clone https://github.com/<your-account>/agent-memory-server.git
cd agent-memory-server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```sh
cp .env.example .env
# Fill in at least API_KEY, MCP_JWT_SECRET, MCP_ADMIN_PASS.
```

**Required (server will refuse to start without these):**

| Key | Purpose |
|---|---|
| `API_KEY` | Value clients must send as `X-API-Key` when calling the REST API |
| `MCP_JWT_SECRET` | JWT signing secret for the MCP OAuth flow (32+ random bytes) |
| `MCP_ADMIN_PASS` | Single-user password for the MCP `/authorize` endpoint |

**Optional (each feature is disabled if its keys are absent):**

| Key | Feature |
|---|---|
| `ANTHROPIC_API_KEY` | Claude calls from `claude_executor.py` / `main.py` |
| `LINE_CHANNEL_SECRET`, `LINE_CHANNEL_ACCESS_TOKEN`, `LINE_USER_ID` | LINE webhook + push |
| `SLACK_BOT_TOKEN`, `SLACK_USER_ID`, `SLACK_SIGNING_SECRET` | Slack webhook + DM push |
| `SLACK_DM_CHANNEL`, `SLACK_INGEST_INITIAL_HOURS` | Slack DM ingester overrides |
| `NOTION_TOKEN`, `NOTION_CU_DATABASE_ID` | Notion sync |
| `AMS_PUBLIC_URL` | Public URL advertised in status pages / MCP OAuth redirects |
| `AMS_LAN_URL` | Optional LAN URL for status pages |
| `MCP_ISSUER_URL` | MCP OAuth issuer (defaults to `http://localhost:8001`) |
| `MCP_PORT` | MCP server port (default `8001`) |
| `OWNER_HANDLE` | String stored in `instructions.given_by`. Defaults to `owner` |
| `AMS_DB_PATH` | Override the SQLite DB path. Defaults to `./memory.db` |
| `AGENT_PROMPT_DIR` | Directory of agent prompt files. Defaults to `./prompts/` |

### 3. Customize agent prompts (optional)

The four tracked templates in `prompts/*.example.md` are minimal, generic
personas (operator, engineer, business, planner). To override, drop a
`prompts/<name>.md` in the same directory — the executor prefers `.md` over
`.example.md`. Local `<name>.md` files are gitignored.

### 4. Initialize the database

The database schema is created automatically on first startup (all tables use
`CREATE TABLE IF NOT EXISTS`). If you have older data to migrate:

```sh
python migrate_add_vectors.py
python migrate_drive_sessions.py
```

### 5. Run

**REST API only (port 8000):**

```sh
uvicorn main:app --host 127.0.0.1 --port 8000
```

**MCP server (port 8001, separate process):**

```sh
python mcp_server.py
```

**Scheduler (background process):**

```sh
python scheduler.py
```

## Using the REST API

Authenticate every call with the header `X-API-Key: <API_KEY>`.

```sh
# Fetch a per-agent context bundle
curl -H "X-API-Key: $API_KEY" 'http://localhost:8000/memory/context?agent=your-agent-name'

# Full-text search
curl -H "X-API-Key: $API_KEY" 'http://localhost:8000/memory/search?q=your+query'

# Insert a semantic memory
curl -X POST -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
     -d '{"key":"example_key","value":"...","category":"design_decision","agent":"my-agent"}' \
     http://localhost:8000/memory/semantic
```

Interactive docs: `http://localhost:8000/docs`.

## Using the MCP server from Claude

Register the MCP endpoint in Claude Desktop / Claude Code as:

```
URL:  http://localhost:8001/mcp   (or your public tunnel URL)
Auth: Bearer <API_KEY>            (raw key, single-user shortcut)
      or OAuth 2.1 flow via /authorize (multi-client)
```

Available tools: `search_memory`, `list_memories`, `get_memory`, `get_context`,
`save_memory`, `save_candidate`, `inbox_add`, `inbox_list`, `inbox_resolve`,
`list_pending_candidates`, `delete_memory`, `save_codex_candidate`.

## Testing

```sh
python test_inbox.py
python test_ownership_and_validation.py
```

Both suites use ephemeral SQLite databases and stub external services — no real
API keys required.

## Security notes

- The MCP server binds to `0.0.0.0` by default so it can be reached over a
  private network (Tailscale / LAN). If you don't need remote access, restrict
  it in your firewall or bind to `127.0.0.1`.
- The MCP OAuth flow is single-user: `/authorize` skips consent and issues a
  code directly. Do not expose the MCP port to the public Internet without a
  reverse proxy that adds additional auth.
- The `X-API-Key` on the REST side is a shared secret. Rotate it if you
  suspect exposure.

## License

MIT. See [LICENSE](LICENSE).
