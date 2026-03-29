# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the service (dev, with hot reload)
uv run uvicorn src.main:app --reload --port 8000

# Run tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_graph.py -v

# Run a single test class or method
uv run pytest tests/test_graph.py::TestGraphWiring::test_graph_routes_faq_to_faq_response -v

# Run with Docker Compose
docker compose up agent --build

# LangGraph Studio (visual graph debugging, in-memory checkpointer)
docker compose up langgraph-studio --build
# Opens at http://localhost:2024
```

## Architecture

Helena Agent is a **LangGraph-based multi-agent service** for WhatsApp customer attention. A NestJS backend sends chat messages via HTTP; this service processes them through a state machine and streams responses as Server-Sent Events (SSE). State is persisted per conversation in PostgreSQL using LangGraph's `AsyncPostgresSaver`. All LLM calls use the OpenAI API (model: `gpt-5`) via `AsyncOpenAI` — the API key is passed per-request from NestJS or falls back to `OPENAI_API_KEY` env var. All user-facing responses are in Spanish.

### Request Flow

```
NestJS → POST /chat/stream (JWT + message body)
  → FastAPI (src/main.py): validate Cognito JWT, scope thread_id as "{cognito_sub}:{client_thread_id}"
  → LangGraph graph.astream() → PostgreSQL checkpointer
  → SSE events streamed back: token | step_progress | execute_workflow | node_update | interrupt_detected | done | error
```

### Graph State Machine (`src/graphs/main_graph.py`)

```
START → triage
  → sales_collect    → order_summary → execute → END
  → tracking_collect → execute → END
  → complaint_collect → execute → END
  → faq_response     → END
```

- **triage** (`src/agents/triage.py`): Silent LLM intent classification — `sales | tracking | complaint | faq`. Skips re-classification if a flow is already in progress. Also contains `route_from_triage()` which resumes the flow at the correct phase based on state flags.
- **sales_collect** (`src/agents/sales_collect.py`): 3 conversational steps collecting order data via two LLM calls per turn (extraction → JSON, then conversational reply). Emits `step_progress` SSE. Steps: (1) customer info, (2) products/quantities from catalog, (3) delivery details.
- **order_summary** (`src/agents/order_summary.py`): Presents order summary and waits for keyword confirmation (`confirmar`, `sí`, `ok`, `dale`, etc.).
- **tracking_collect** (`src/agents/tracking_collect.py`): Collects order ID or customer details for status lookup. Single step.
- **complaint_collect** (`src/agents/complaint_collect.py`): Collects complaint details: order ref, issue description, desired resolution. Single step.
- **faq_response** (`src/agents/faq_response.py`): Answers FAQs (hours, location, payments, shipping) and serves as fallback for unknown intents.
- **execute** (`src/agents/execute.py`): Emits `execute_workflow` SSE for NestJS to trigger backend workflows. Sets `execute_confirmed=True` to prevent re-execution.

**Auto-chaining**: When a node completes all its sub-steps, the next node runs in the same turn — no extra user message needed. This is implemented via conditional edges (`_route_from_sales_collect`, `_route_from_order_summary`, `_route_from_tracking_collect`, `_route_from_complaint_collect`) that check state flags.

### Key State (`src/graphs/state.py`)

`AgentState` is a `TypedDict` persisted per `thread_id`:
- `messages`: full conversation history (LangGraph `add_messages` reducer)
- `intent`: current classified intent (`sales | tracking | complaint | faq`)
- `product_catalog`: list of `{product_id, name, description, price, stock}` from NestJS
- `user_context`: dict with `{name, email, phone, address}` from NestJS
- Sales: `sales_step` (0-3), `order_data` (dict), `sales_complete`, `order_confirmed`
- Tracking: `tracking_data` (dict), `tracking_complete`
- Complaint: `complaint_data` (dict), `complaint_complete`
- `execute_confirmed`: boolean flag preventing re-execution

### LLM Client (`src/llm.py`)

`resolve_api_key(config)` extracts the key from `config["configurable"]["openai_api_key"]` (set per-request) or falls back to `settings.openai_api_key`. Each agent node calls this independently — there is no shared client instance.

### Authentication (`src/auth/cognito.py`)

RS256 JWT validation via AWS Cognito JWKS (cached 5 min). JWKS URL and issuer are constructed automatically from `COGNITO_USER_POOL_ID` and `COGNITO_REGION`. Validates `token_use` claim (`access` or `id`) and optionally `client_id`/`aud` against `COGNITO_APP_CLIENT_ID`. Thread IDs are scoped per user (`{sub}:{thread_id}`) to prevent cross-user state leaks. API keys from the request body are stripped before emitting `node_update` SSE events.

### Database (`src/db/postgres.py`, `migrations/init.sql`)

asyncpg pool + LangGraph's `AsyncPostgresSaver`. Schema runs idempotently on startup. Tables: `checkpoints`, `checkpoint_blobs`, `checkpoint_writes` (LangGraph), `documents` (pgvector for RAG), `approval_requests` (interrupt audit log).

### Observability (`src/observability.py`)

- Prometheus metrics at `GET /metrics` — request counts, node invocations, order/tracking/complaint funnel, errors
- Optional Langfuse LLM tracing — gracefully disabled if keys not configured

## Environment

Copy `.env.example` to `.env`. Required vars:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `COGNITO_USER_POOL_ID` | AWS Cognito User Pool ID (e.g., `us-east-1_abc123`) |
| `COGNITO_REGION` | AWS region (e.g., `us-east-1`) |

Optional: `COGNITO_APP_CLIENT_ID` (audience validation), `OPENAI_API_KEY` (fallback; per-request key preferred), `LANGFUSE_*`.

## Testing

Tests use `MemorySaver` (no database needed). `conftest.py` sets dummy env vars (`DATABASE_URL`, `COGNITO_USER_POOL_ID`, `COGNITO_REGION`) before any app modules import. `asyncio_mode = "auto"` in `pyproject.toml` — no need to mark individual tests with `@pytest.mark.asyncio`.

Key test patterns: mock individual agent nodes (e.g., `patch("src.graphs.main_graph.triage_node")`), build a fresh graph with `build_graph(MemorySaver())`, stream with `graph.astream()`, and inspect chunks. Tests verify routing logic, API key security (no leaks into state/interrupts), and auto-chaining flows.

## Studio Graph

`src/graphs/studio_graph.py` is the LangGraph Studio entrypoint — uses an in-memory checkpointer and is not used in production.
