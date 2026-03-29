# helena-agent
# sek

LangGraph multi-agent service for Helena security operations. Receives chat messages from the NestJS backend via HTTP, runs a stateful agent graph, and streams responses back as Server-Sent Events (SSE).

## Overview

```
NestJS backend
      │  POST /chat/stream
      │  Bearer <Keycloak JWT>
      ▼
┌─────────────────────────────────────────────────┐
│  FastAPI  (uvicorn, port 8000)                  │
│                                                 │
│  JWT validation ──► LangGraph graph             │
│                          │                      │
│                    PostgreSQL                   │
│                    checkpointer                 │
│                    (state per thread)           │
└────────────┬────────────────────────────────────┘
             │  SSE stream
             │  token | rfc_step_progress | closed_questions
             │  execute_workflow | interrupt_detected | done
             ▼
      NestJS backend
```

The graph is compiled once at startup and reused across requests. State is persisted per `thread_id` in PostgreSQL so conversations survive process restarts.

---

## Agent graph

```
START
  │
  ▼
triage ──────────────────────────────────► fallback_response ──► END
  │                                              ▲
  │ intent = "rfc"                               │ (unknown intent /
  ▼                                              │  rfc fully complete)
rfc_open_questions
  │ rfc_open_complete = True (auto-chain)
  ▼
rfc_closed_questions
  │ rfc_closed_complete = True (auto-chain)
  ▼
rfc_summary_confirm
  │ rfc_confirmed = True (user says "confirm")
  ▼
rfc_execute ──────────────────────────────────────────────────► END
```

Auto-chain edges advance the graph to the next phase within the **same turn** — the user doesn't need to send an extra message for the transition.

### Nodes

| Node | Purpose |
|---|---|
| `triage` | Classifies intent (rfc / incident / knowledge / escalation / unknown). Silent — emits no tokens. |
| `rfc_open_questions` | Collects open-ended RFC fields across 5 conversational steps. |
| `rfc_closed_questions` | Collects 8 structured single-select answers, rendered as a form in the frontend. |
| `rfc_summary_confirm` | Presents a formatted RFC summary and waits for user confirmation or corrections. |
| `rfc_execute` | Emits an `execute_workflow` SSE event; NestJS handles webhook invocation and result formatting. |
| `fallback_response` | Responds to unknown intents and explains available capabilities. |

#### `rfc_open_questions` — 5 steps

| Step | Fields collected |
|---|---|
| 1 | title, description, change type, category |
| 2 | business justification, affected systems/users, impact level |
| 3 | implementation steps, rollback plan, dependencies, resources |
| 4 | start/end dates, environment, change window |
| 5 | risk level, mitigation, testing plan, approvers |

Each step uses two LLM calls: one to **extract** field values from the user's free-text answer (JSON output), and one to **compose** the conversational reply asking for any missing fields or announcing the next step.

#### `rfc_closed_questions` — 8 structured questions

| Field | Options |
|---|---|
| `change_type` | Estándar / Normal / Emergencia / Mayor |
| `service_impact` | Sin impacto / Mínimo / Parcial / Total |
| `monitoring_loss` | No / Parcialmente / Totalmente / N/A |
| `remote_access_loss` | No / Temporalmente / Totalmente / N/A |
| `backup_status` | Sí validado / Sí no validado / No / N/A |
| `rollback_available` | Sí completamente / Sí parcialmente / No / N/A |
| `risk_level` | Bajo / Medio / Alto / Crítico |
| `execution_location` | Remoto / En sitio / Híbrido / N/A |

Emits a `closed_questions` SSE event with the question schema so the frontend can render a form component instead of free text.

#### `rfc_execute`

Emits an `execute_workflow` SSE event and returns `rfc_execute_confirmed: True`. The agent never holds webhook URLs — the NestJS backend resolves them, POSTs to N8N, and emits the formatted result as a new chat bubble.

---

## SSE event types

Events are emitted as `data: <json>\n\n` lines on the `/chat/stream` response.

| `type` | Payload fields | Description |
|---|---|---|
| `token` | `content: str` | One LLM delta token |
| `rfc_step_progress` | `step`, `total_open_steps`, `topic` | RFC collection step indicator |
| `closed_questions` | `questions: list` | Question schema for frontend form |
| `execute_workflow` | `conversation_id`, `workflows`, `rfc_data` | Signals NestJS to trigger N8N webhooks |
| `node_update` | `node`, `data` | Graph node state (sensitive fields stripped) |
| `interrupt_detected` | `interrupt_id`, `payload` | Human approval required |
| `done` | — | Stream finished |
| `error` | `message: str` | Agent-level error |

---

## State

`AgentState` is the TypedDict that LangGraph persists per thread.

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # Full conversation history
    thread_id: str                             # User-scoped ID
    project_id: str                            # NestJS project UUID
    conversation_id: str                       # NestJS conversation UUID
    project_workflows: list                    # [{workflow_id, name, description}]
    intent: str                                # triage output
    rfc_step: int                              # 0–7
    rfc_data: dict                             # Accumulated RFC fields
    rfc_open_complete: bool
    rfc_closed_complete: bool
    rfc_confirmed: bool
    rfc_execute_confirmed: bool                # Prevents re-execution
```

Thread IDs are namespaced per user: `{keycloak_sub}:{client_thread_id}`. This ensures one user cannot read another user's graph state.

---

## HTTP API

### `POST /chat/stream`

Stream a user message through the graph.

**Headers:** `Authorization: Bearer <keycloak_token>`

**Request body:**
```json
{
  "thread_id": "conv-uuid",
  "message": "Quiero crear un RFC",
  "openai_api_key": "sk-...",
  "project_id": "proj-uuid",
  "conversation_id": "conv-uuid",
  "project_steps": [
    {
      "type": "WORKFLOW",
      "order": 1,
      "workflows": [
        {
          "workflowId": "wf-uuid",
          "workflowName": "Abrir ticket ServiceNow",
          "description": "Crea un ticket en ServiceNow"
        }
      ]
    }
  ]
}
```

**Response:** `text/event-stream` — sequence of `data: <json>` lines.

`project_steps` is provided by the NestJS backend on every request. WORKFLOW-type steps populate `project_workflows` in the graph state, making the workflows available to `rfc_execute`.

### `POST /chat/resume`

Resume a graph paused at a `interrupt()` call.

**Headers:** `Authorization: Bearer <keycloak_token>`

**Request body:**
```json
{
  "thread_id": "conv-uuid",
  "interrupt_id": "uuid",
  "approved": true,
  "openai_api_key": "sk-..."
}
```

**Response:** same SSE stream format as `/chat/stream`.

### `GET /health`

Returns `{"status": "ok"}`. Used as Docker healthcheck.

### `GET /metrics`

Prometheus metrics endpoint.

---

## Configuration

All settings are loaded from environment variables (or a `.env` file).

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | — | PostgreSQL connection string. Used for LangGraph checkpoints, approval_requests, and RAG embeddings. |
| `KEYCLOAK_ISSUER` | Yes | — | Full issuer URL, e.g. `https://keycloak.example.com/realms/my-realm` |
| `KEYCLOAK_JWKS_URL` | Yes | — | JWKS endpoint for RS256 token validation |
| `KEYCLOAK_TLS_VERIFY` | No | `true` | Set to `false` for self-signed certificates |
| `OPENAI_API_KEY` | No | — | Fallback API key. NestJS passes a per-request key in the request body; this is used only if none is provided. |
| `LANGFUSE_SECRET_KEY` | No | — | Langfuse secret. Tracing is disabled if left empty. |
| `LANGFUSE_PUBLIC_KEY` | No | — | Langfuse public key. |
| `LANGFUSE_HOST` | No | `http://localhost:3010` | Langfuse server URL. |
| `NESTJS_BASE_URL` | No | — | Backend base URL (reserved for future agent→backend calls). |

**Example `.env`:**
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ticket_support
KEYCLOAK_ISSUER=https://keycloak.example.com/realms/my-realm
KEYCLOAK_JWKS_URL=https://keycloak.example.com/realms/my-realm/protocol/openid-connect/certs

# Optional
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_HOST=http://localhost:3010
```

---

## Running locally

### With Docker Compose (recommended)

From the repo root:

```bash
docker compose up agent --build
```

The agent starts at `http://localhost:8000`. Hot reload is enabled via `--reload`.

### Without Docker

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
cd helena-agent

# Install dependencies
uv sync

# Copy and edit environment
cp .env.example .env

# Start the server
uv run uvicorn src.main:app --reload --port 8000
```

On startup the service:
1. Creates an asyncpg connection pool
2. Runs `migrations/init.sql` (idempotent)
3. Compiles the LangGraph graph with an `AsyncPostgresSaver` checkpointer

### LangGraph Studio

For visual graph inspection and interactive testing:

```bash
docker compose up langgraph-studio --build
# Open http://localhost:2024
```

The Studio build uses `Dockerfile.studio` and mounts `langgraph.json` which points to `src/graphs/studio_graph.py:graph`. Studio manages its own in-memory checkpointer so a live database is not required.

---

## Testing

Tests use `MemorySaver` (in-memory checkpointer) so no database is needed.

```bash
cd helena-agent
uv run pytest src/tests/ -v
```

| Test file | What it covers |
|---|---|
| `tests/test_graph.py` | Graph routing, API key never leaks into state, LLM output is natural language |
| `tests/test_interrupt.py` | Interrupt emitted correctly, resume with approval/rejection, API key never leaks into interrupt payload |

---

## Database schema

Managed by `migrations/init.sql`, run automatically at startup.

| Table | Purpose |
|---|---|
| `documents` | RAG knowledge base — pgvector embeddings (1536-dim), IVFFlat index |
| `approval_requests` | Interrupt audit log — `thread_id`, `payload` (JSONB), `status` (pending / approved / rejected), timestamps |
| `checkpoints` | LangGraph state snapshots |
| `checkpoint_blobs` | Large binary state data |
| `checkpoint_writes` | Task-level write log |
| `checkpoint_migrations` | LangGraph schema versioning |

---

## Observability

### Prometheus metrics

Available at `GET /metrics`.

| Metric | Type | Description |
|---|---|---|
| `agent_requests_total` | Counter | Total `/chat/stream` requests |
| `agent_interrupt_events_total` | Counter | Total interrupts triggered |
| `agent_node_invocations_total` | Counter | Per-node invocations (label: `node`) |
| `agent_tool_errors_total` | Counter | Per-tool errors (label: `tool`) |
| `rfc_chains_started_total` | Counter | RFC flows started |
| `rfc_chains_submitted_total` | Counter | RFC flows completed (workflow triggered) |
| `rfc_chains_rejected_total` | Counter | RFC flows cancelled by user |

### Langfuse tracing

Each LLM call within a node creates a Langfuse generation with model name, input messages, output, and token usage. Tracing is opt-in — the service runs normally without Langfuse credentials.

---

## Project layout

```
helena-agent/
├── src/
│   ├── main.py                # FastAPI app, lifespan, endpoints, SSE streaming
│   ├── config.py              # Pydantic settings
│   ├── llm.py                 # AsyncOpenAI client + API key resolution
│   ├── observability.py       # Prometheus counters + Langfuse integration
│   ├── auth/
│   │   └── keycloak.py        # JWKS validation, get_current_user, scoped_thread_id
│   ├── db/
│   │   └── postgres.py        # asyncpg pool init/close + migration runner
│   ├── agents/
│   │   ├── triage.py          # Intent classification
│   │   ├── rfc_open.py        # Open-ended RFC questions (5 steps)
│   │   ├── rfc_closed.py      # Structured RFC questions (8 fields)
│   │   ├── rfc_summary.py     # Summary presentation + confirmation
│   │   ├── rfc_execute.py     # Workflow execution trigger
│   │   ├── fallback.py        # Unknown intent handler
│   │   ├── escalation.py      # (placeholder)
│   │   ├── orchestrator.py    # (placeholder)
│   │   ├── rag.py             # (placeholder)
│   │   └── workflow.py        # (placeholder)
│   └── graphs/
│       ├── state.py           # AgentState TypedDict
│       ├── main_graph.py      # Graph definition + conditional edges
│       └── studio_graph.py    # LangGraph Studio entrypoint
├── migrations/
│   └── init.sql               # Schema (idempotent)
├── tests/
│   ├── test_graph.py
│   └── test_interrupt.py
├── pyproject.toml
├── uv.lock
├── langgraph.json             # Studio config
├── Dockerfile                 # Production (non-root, two-stage)
└── Dockerfile.studio          # LangGraph Studio dev build
```
