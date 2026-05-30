# verveux-agent

LangGraph multi-agent service for the Verveux/RockyBot platform. Receives chat messages from the NestJS backend via HTTP, routes them to domain-specific agent graphs, and streams responses back as Server-Sent Events (SSE).

## Overview

```
NestJS backend
      |  POST /chat/stream { agent_type: "sales", ... }
      |  Bearer <Cognito JWT>
      v
+--------------------------------------------------+
|  FastAPI  (uvicorn, port 8000)                   |
|                                                  |
|  JWT validation --> Graph Registry               |
|                       |                          |
|                  resolve agent_type               |
|                       |                          |
|             +---------+---------+                |
|             |    |    |    |                     |
|           SALES SCHOOL REST APPT                 |
|             |                                    |
|        PostgreSQL checkpointer                   |
|        (state per thread)                        |
+------------------+-------------------------------+
                   |  SSE stream
                   |  token | execute_workflow | node_update | done
                   v
            NestJS backend
```

## Multi-Agent Architecture

### Graph Registry Pattern

Each agent domain (SALES, SCHOOL, RESTAURANT, APPOINTMENTS) has its own fully isolated, pre-compiled LangGraph graph. At startup, all graphs are compiled once and cached. At request time, the `agent_type` field in the request body selects which graph handles the conversation.

```
src/graphs/
  registry.py            # AGENT_REGISTRY + get_agent_graph() dispatcher
  state.py               # Shared AgentState TypedDict
  sales_graph.py         # Full sales pipeline (triage -> sales/tracking/complaint/faq -> execute)
  school_graph.py        # Stub: triage -> faq_response
  restaurant_graph.py    # Stub: triage -> faq_response
  appointments_graph.py  # Stub: triage -> faq_response
  main_graph.py          # Backward-compat shim (delegates to sales_graph)
  studio_graph.py        # LangGraph Studio entrypoint
```

### AgentState

The `AgentState` TypedDict includes multi-agent identity fields:

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    thread_id: str

    # Multi-agent identity
    agent_type: str        # "sales" | "school" | "restaurant" | "appointments"
    capabilities: dict     # Capability contract from NestJS (read-only)
    domain_state: dict     # Generic bag for domain-specific data

    # Conversation context
    tenant_id: str
    conversation_id: str
    product_catalog: list
    knowledge: list
    user_context: dict
    contact_id: str
    contact_tags: list
    language: str
    faqs: list

    # Triage
    intent: str
    structured_intent: Optional[StructuredIntent]

    # Sales-specific fields (only used by sales graph)
    sales_phase: str
    cart: Optional[list]
    cart_confirmed: bool
    # ... additional sales, tracking, complaint fields
```

The `domain_state: dict` field is a generic bag that prevents domain-specific data from polluting shared state across agent types.

### Graph Dispatch

Unknown `agent_type` values safely fall back to SALES with a warning log.

```python
# Request: { "agent_type": "school", ... }
graph = get_agent_graph("school")  # Returns compiled school graph
# Request: { "agent_type": "unknown", ... }
graph = get_agent_graph("unknown")  # Falls back to SALES
```

### Sales Graph (Full Pipeline)

```
START
  |
  v
triage -----> sales_collect --> sales_confirm --> customer_data_collect --> order_summary --> execute --> END
  |                                                                                           ^
  |---------> tracking_collect ----------------------------------------------------------------|
  |---------> complaint_collect ---------------------------------------------------------------|
  |---------> faq_response --> END
```

Auto-chain edges advance the graph to the next phase within the same turn.

### Stub Graphs (School, Restaurant, Appointments)

Minimal `triage -> faq_response -> END` pipeline. Domain-specific nodes will be added as each agent is fleshed out.

## SSE Event Types

Events are emitted as `data: <json>\n\n` lines on the `/chat/stream` response.

| `type` | Payload fields | Description |
|---|---|---|
| `token` | `content: str` | One LLM delta token |
| `step_progress` | `step`, `total_steps`, `topic` | Collection step indicator |
| `execute_workflow` | `conversation_id`, `intent`, `order_data`, `tracking_data`, `complaint_data` | Signals NestJS to trigger backend workflows |
| `tag_contact` | `contact_id`, `tag_name` | Tag a contact |
| `create_deal` | `contact_id`, `conversation_id`, `title`, `source` | Create a CRM deal |
| `update_deal_stage` | `contact_id`, `conversation_id`, `stage_position` | Move deal stage |
| `node_update` | `node`, `data` | Graph node state (sensitive fields stripped) |
| `interrupt_detected` | `interrupt_id`, `payload` | Human approval required |
| `done` | -- | Stream finished |
| `error` | `message: str` | Agent-level error |

## HTTP API

### `POST /chat/stream`

Stream a user message through the appropriate agent graph.

**Headers:** `Authorization: Bearer <cognito_token>`

**Request body:**
```json
{
  "thread_id": "conv-uuid",
  "message": "Quiero hacer un pedido",
  "agent_type": "sales",
  "capabilities": ["crm_pipeline", "catalog", "order_management"],
  "tenant_id": "tenant-uuid",
  "conversation_id": "conv-uuid",
  "product_catalog": [],
  "user_context": {},
  "contact_id": "contact-uuid",
  "contact_tags": [],
  "language": "es",
  "rawFaqs": [],
  "prompts": {},
  "llm_provider": "openai",
  "llm_model": "gpt-4o"
}
```

Key fields:
- `agent_type` (default: `"sales"`) -- selects which compiled graph handles the request
- `capabilities` -- capability contract from NestJS, injected into state as read-only context

**Response:** `text/event-stream` -- sequence of `data: <json>` lines.

### `POST /chat/resume`

Resume a graph paused at an `interrupt()` call.

**Request body:**
```json
{
  "thread_id": "conv-uuid",
  "interrupt_id": "uuid",
  "approved": true,
  "agent_type": "sales"
}
```

### `GET /health` / `GET /healthz`

Returns `{"status": "ok"}`.

### `GET /metrics`

Prometheus metrics endpoint.

## Configuration

All settings are loaded from environment variables (or a `.env` file).

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | -- | PostgreSQL connection string |
| `COGNITO_ISSUER` | Yes | -- | Cognito issuer URL |
| `COGNITO_JWKS_URL` | Yes | -- | JWKS endpoint for token validation |
| `NESTJS_BASE_URL` | No | -- | Backend base URL for credential fetching |

## Running Locally

### With Docker Compose

```bash
docker compose up agent --build
```

### Without Docker

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
uv run uvicorn src.main:app --reload --port 8000
```

On startup the service:
1. Creates an asyncpg connection pool and runs migrations
2. Compiles all agent graphs (sales, school, restaurant, appointments) with an `AsyncPostgresSaver` checkpointer

### LangGraph Studio

```bash
docker compose up langgraph-studio --build
# Open http://localhost:2024
```

Studio compiles the sales graph with no checkpointer for visualization.

## Testing

Tests use `MemorySaver` (in-memory checkpointer) so no database is needed.

```bash
uv run pytest tests/ -v
```

## Observability

### Prometheus Metrics (GET /metrics)

| Metric | Type | Description |
|---|---|---|
| `agent_requests_total` | Counter | Total `/chat/stream` requests |
| `agent_interrupt_events_total` | Counter | Total interrupts triggered |
| `agent_node_invocations_total` | Counter | Per-node invocations (label: `node`) |
| `agent_tool_errors_total` | Counter | Per-tool errors (label: `tool`) |

## Project Layout

```
src/
  main.py                # FastAPI app, lifespan, endpoints, SSE streaming
  config.py              # Pydantic settings
  observability.py       # Prometheus counters
  auth/
    cognito.py           # JWT validation, get_current_user, scoped_thread_id
  db/
    postgres.py          # asyncpg pool init/close + migration runner
  agents/
    triage.py            # Intent classification
    sales_collect.py     # Product selection + cart management
    sales_confirm.py     # Cart confirmation
    customer_data_collect.py  # Delivery/customer data collection
    order_summary.py     # Final order summary
    execute.py           # Workflow execution trigger
    tracking_collect.py  # Order tracking flow
    complaint_collect.py # Complaint handling flow
    faq_response.py      # FAQ-based responses
    backend_client.py    # NestJS API client (credentials, etc.)
    ...
  graphs/
    registry.py          # Graph registry + dispatcher
    state.py             # AgentState TypedDict
    sales_graph.py       # Sales pipeline graph
    school_graph.py      # School agent stub
    restaurant_graph.py  # Restaurant agent stub
    appointments_graph.py # Appointments agent stub
    main_graph.py        # Backward-compat shim
    studio_graph.py      # LangGraph Studio entrypoint
  schemas/
    intent.py            # StructuredIntent model
```
