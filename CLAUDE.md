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
NestJS → POST /chat/stream (service credential + message body)
  → FastAPI (src/main.py): authenticate the calling service, scope thread_id as
    "{tenantId}:system:{conversationId}:{agentCodeName}:v{agentVersion}"
  → per-thread coalescing (_coalesced_stream): serialize runs, merge rapid message bursts
  → LangGraph graph.astream() → PostgreSQL checkpointer
  → SSE events streamed back: token | step_progress | execute_workflow | node_update | interrupt_detected | done | error
```

### Multi-message handling

WhatsApp users often split one thought across several rapid messages; NestJS forwards each as its own `POST /chat/stream`. Two layers turn a burst into ONE coherent turn:

1. **Per-thread coalescing** (`_coalesced_stream` in `src/main.py`): runs are serialized per `thread_id` with an in-process `asyncio.Lock`. A turn waits until no new fragment has arrived for `MESSAGE_SETTLE_SECONDS` (default 2s, total wait capped at `MESSAGE_SETTLE_MAX_SECONDS`, default 10s), then drains all buffered fragments into a single `HumanMessage` (texts joined with newlines, attachments concatenated, per-request FAQ payloads unioned with question-level dedupe). Superseded requests complete with an empty `done` event (`turn_usage: []`, `coalesced: true`) — NestJS treats that as "no reply": it releases the credit reservation and sends nothing. Locks are process-local; horizontal scaling requires sticky routing by thread_id. NestJS complements this with burst-aware FAQ retrieval: `AgentService.buildFaqQueryText` queries FAQs with the whole trailing inbound burst, not just the current fragment.
2. **Trailing-burst readers** (`latest_user_messages` / `latest_user_text` in `src/agents/utils.py`): every node that consumes "the user's message" reads ALL trailing consecutive human messages since the bot's last reply — never `messages[-1]` alone — so fragments that land as separate checkpoint entries are still answered together. Keyword confirmation checks (booking_confirm, school_graph question guard) test each fragment individually.

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
- **query_normalizer** (`src/agents/query_normalizer.py`): First node in every graph (`START → query_normalizer → triage`). Usually a pure-Python passthrough that stamps `original_text`; the LLM typo-correction runs only when the deterministic trigger fires (`should_normalize` in `src/graphs/shared_routing.py`: per-tenant flag `query_normalization_enabled` from the request AND env kill switch `QUERY_NORMALIZATION_ENABLED` on, text ≥ 4 chars, no `ESCALATION_KEYWORDS` hit, initial backend FAQ retrieval empty or max `score` < `FAQ_SCORE_TRIGGER_THRESHOLD`=0.15). On an applied correction (risk LOW/MEDIUM + confidence ≥ 0.7) it re-fetches FAQs once via `backend_client.search_faqs` (`GET /internal/faqs/search`, conversation-scoped) and overwrites `state["faqs"]`; it NEVER rewrites `messages` — the corrected text lives only in the write-once `normalization` provenance dict (`{enabled, model, confidence, changed_meaning_risk, reason, applied, corrected_text}`, sole writer: this node). Camila guardrails: the deterministic identity short-circuit requires the conflict to persist on both original and corrected transcripts when a normalization applied (`_identity_conflict_confirmed`), and a weak LLM `identity_conflict` (confidence < 0.75) on a normalized turn without escalation keywords is downgraded to `faq`. Observability: `query_normalization` structured log line + `query_normalizations_total{outcome}` counter.
- **greeting_response** (`src/agents/greeting_response.py`): Greets known contacts (triage intent `greeting` + contact name on file). LLM-rendered from the tenant's `{AGENT_TYPE}_GREETING` prompt (resolved via `resolve_prompt`, placeholders `{persona}`/`{role}`/`{name}`/`{language_rule}` substituted with a brace-tolerant `format_map`); falls back to the deterministic per-language templates when the LLM call fails or returns empty (fallback contributes no `turn_usage`). Shared by every agent graph.
- **execute** (`src/agents/execute.py`): Emits `execute_workflow` SSE for NestJS to trigger backend workflows. Sets `execute_confirmed=True` to prevent re-execution.

**Auto-chaining**: When a node completes all its sub-steps, the next node runs in the same turn — no extra user message needed. This is implemented via conditional edges (`_route_from_sales_collect`, `_route_from_order_summary`, `_route_from_tracking_collect`, `_route_from_complaint_collect`) that check state flags.

### Email agent (`clara`, agent type EMAIL)

Gmail → sanitize → triage → task extraction → reply draft → **human approval in the CRM** → Gmail **draft**. It never sends. Ported from the `emailAi` prototype.

- **Trigger:** the backend's email sweep (Cloud Scheduler) calls `POST /email/sync` (202, background) per mailbox and `POST /email/follow-up` per due thread; `POST /email/drafts` runs synchronously after a human approved/edited a reply. `agent_code_name` is **required** on all three (no default) and must resolve to a graph compiled with `name=EMAIL_GRAPH_NAME`, else 400.
- **Graph** (`src/graphs/clara_graph.py`, nodes in `src/agents/clara/nodes.py`): `precheck → triage → (extract_task) → draft_reply`, or `draft_follow_up` in follow-up mode. **Tool-free**: each node is pure Python or ONE `generate_structured` call. `precheck` short-circuits only on strong signals (`Auto-Submitted`, bounces, internal sender = same non-public domain as the mailbox) and fails open otherwise. Guards run after every node: parser `security_flags` force human review; `evidence_quote`/`due_date_evidence` must be literal substrings of the email.
- **Side effects** live only in `src/agents/clara/runner.py`: access token from the backend (scopes must be exactly `gmail.readonly` + `gmail.compose`, checked on every use — `src/services/gmail.py`), history-cursor sync, reports to `/api/v1/internal/email/*`. Idempotency: `alreadyIngested` before any model call, one checkpoint thread per message (`{code}:{tenant}:{mailbox}:{key}`) resumed by `run_graph_once` and **deleted** after the backend confirms, `X-Clara-Approval-Id` header so a retried draft is found instead of duplicated. A message that keeps failing is reported FAILED until the backend answers `giveUp` (3 attempts); a provider config error fails the whole run. Credentials never fall back to the platform OpenAI key.
- **Parser** (`src/services/email_parser.py`): hidden HTML, quoted history, zero-width/bidi characters (stripped before the injection regexes) and injection phrases are removed or flagged before anything reaches a model. Email text reaches prompts only inside `<<<CORREO … CORREO>>>` fences.
- **Prompts** (`src/agents/clara/prompts.py`, keys `EMAIL_*`) are mirrored in the backend's `DEFAULT_PROMPTS`; every usage row carries `prompt_key/prompt_id/prompt_version/prompt_sha` and `latency_ms` (optional `InvocationUsage` keys only this agent sets).
- **Structured output:** `ChatProvider.generate_structured` (prompt-JSON fallback); Gemini overrides it with native `response_schema` + per-node `ThinkingConfig`, folds thinking tokens into `output_tokens` (the backend bills reasoning as a subset of output) and retries once without a budget on models that cannot disable thinking.
- **Memory:** human-edited replies (last 3 per sender, keyed by approval id) in the existing `AsyncPostgresStore`, namespace `("clara_sender_prefs", tenant_id)`. Written only on a human edit, never by a model.
- **Eval:** `uv run python scripts/eval_clara.py` (real LLM, costs money) over `tests/fixtures/email/` with a local LLM-as-judge for `draft_criteria`.

### Theology agent (`ismael`, agent type THEOLOGY)

Christian theology Q&A for the MOODLE bubble, answered by **Company Brain** (the cited RAG in `yorch-tauri-backend`, one EC2 host in AWS that is **stopped when idle**).

- **Graph** (`src/graphs/ismael_graph.py`, nodes in `src/agents/ismael/nodes.py`): `ismael_triage → {ismael_survey | ismael_pending | ismael_start | greeting_response | ismael_off_topic} → END`. Triage is deterministic while a survey question is open or a job is running; otherwise ONE `generate_structured` call returns the intent and the question rewritten to stand alone (Brain never sees the conversation). No name capture — MOODLE identity is server-attested.
- **The answer never comes from the graph.** A cold Brain plus a real question outlasts the backend's 60 s turn, so `ismael_start` replies with a holding message and `rag_job.spawn` runs a background task: `brain.ensure_started` → `wait_healthy` (re-requests the start every 30 s: a host caught mid-stop refuses StartInstances) → `POST /ask` → poll `GET /ask/:id` → `POST /api/v1/internal/conversations/:id/agent-messages` (idempotent on `jobId`; the backend stores, dispatches over the widget SSE and bills `turnUsage`). `answered` is sent **verbatim** plus a References block from `citations`+`evidence` (an LLM rewrite could detach a claim from its source); `insufficient_evidence`/`off_corpus` get a ≤120-word general answer (`THEOLOGY_GENERAL`) with a notice; boot/answer timeout or a failed question gets an apology.
- **Survey (statistics only, once per contact):** on a new contact's first question `ismael_start` fires the EC2 start in the background and asks 3 fixed questions (`texts.SURVEY`: level, topic, intendedUse — option keys are statistics, never tenant-editable); the answers go to `POST /internal/contacts/:id/ismael-survey` → `profileData.ismael`, and the backend then reports `user_context.ismael_survey_done`. Numbers/labels parse deterministically; an LLM is asked only otherwise; unreadable → `no_answer`, never re-asked; a new question typed instead of an answer replaces the pending one.
- **One job per conversation:** running state in the shared `AsyncPostgresStore` (`("ismael_jobs", tenant)`, key conversationId, TTL = boot + answer timeout + 2 min) so it holds across Cloud Run instances; an in-process dict is the fallback. A recycled instance loses its job — the entry expires and the student can ask again.
- **Starting the host** (`src/services/brain.py`): metadata-server ID token (audience `BRAIN_OIDC_AUDIENCE`) → boto3 `AssumeRoleWithWebIdentity` → `ec2:StartInstances`. No AWS key anywhere. `BRAIN_START_MODE=skip` off Cloud Run. Brain auth is `X-Api-Key` (a Brain service key bound to one tenant).
- Tests: `tests/test_ismael_graph.py`.

### Key State (`src/graphs/state.py`)

`AgentState` is a `TypedDict` persisted per `thread_id`:
- `messages`: full conversation history (LangGraph `add_messages` reducer)
- `intent`: current classified intent (`sales | tracking | complaint | faq`)
- `structured_intent`: plain JSON dict (`StructuredIntent.model_dump(mode="json")`) — checkpoints must never carry custom Python types. Legacy checkpoints holding Pydantic instances are allow-listed via `JsonPlusSerializer(allowed_msgpack_modules=...)` in `main.py`; readers (`route_from_triage`, `handoff`) handle both shapes.
- `product_catalog`: list of `{product_id, name, description, price, stock}` from NestJS
- `user_context`: dict with `{name, email, phone, address}` from NestJS
- Sales: `sales_step` (0-3), `order_data` (dict), `sales_complete`, `order_confirmed`
- Tracking: `tracking_data` (dict), `tracking_complete`
- Complaint: `complaint_data` (dict), `complaint_complete`
- `execute_confirmed`: boolean flag preventing re-execution

### LLM Client (`src/llm.py`)

`resolve_api_key(config)` extracts the key from `config["configurable"]["openai_api_key"]` (set per-request) or falls back to `settings.openai_api_key`. Each agent node calls this independently — there is no shared client instance.

### Authentication (`src/auth/service_auth.py`)

Every caller is another service — the NestJS backend and its schedulers — never an end user. There is no browser or mobile client on the other end, which is why this verifies *workload* identity rather than user identity. It replaced the AWS Cognito verifier, whose JWT branch was dead on every production request.

Two accepted credentials, both handled by the single `verify_service_caller` dependency (all four authenticated routes share it):

1. **Google OIDC ID token** on `Authorization: Bearer`. RS256 against Google's JWKS (`https://www.googleapis.com/oauth2/v3/certs`, cached 5 min), issuer `https://accounts.google.com`, `aud` pinned to `SERVICE_AUTH_AUDIENCE` (this service's own Cloud Run URL), then `email_verified == true` and `email` in `SERVICE_AUTH_ALLOWED_SERVICE_ACCOUNTS`. Fails closed when the audience is set but the allowlist is empty. Cloud Run's IAM layer already validates this at the edge (`allow_unauthenticated = false`) and forwards it, so verifying here is defence in depth — which matters because the same image runs under `docker compose` with no Cloud Run in front.
2. **Shared secret** (`WEBHOOK_API_KEY`) on `X-System-Key` or `x-agent-key`, gated by `ALLOW_SHARED_SECRET_AUTH`. A *present but wrong* key does not short-circuit: during the migration the backend sends both credentials, and a stale secret must not veto a valid token.

`get_current_user` always returns the literal `"system"`. **That value is persisted data, not an auth detail** — it is the second segment of every LangGraph thread id in Postgres (`{tenantId}:system:{conversationId}:{codeName}:v{n}`), so returning a service-account email or a numeric `sub` instead would orphan every live conversation's checkpoint history. `tests/test_service_auth.py` guards it.

Rejections funnel through one helper that increments `service_auth_rejects_total{reason}` and logs `service_auth_reject`; the detail returned to the caller stays coarse. API keys from the request body are stripped before emitting `node_update` SSE events.

### Database (`src/db/postgres.py`, `migrations/init.sql`)

asyncpg pool + LangGraph's `AsyncPostgresSaver`. Schema runs idempotently on startup. Tables: `checkpoints`, `checkpoint_blobs`, `checkpoint_writes` (LangGraph), `documents` (pgvector for RAG), `approval_requests` (interrupt audit log).

### Observability (`src/observability.py`)

- Prometheus metrics at `GET /metrics` — request counts, node invocations, order/tracking/complaint funnel, errors
- Optional Langfuse LLM tracing — gracefully disabled if keys not configured

## ⚠️ `agent-migrate` (Cloud Run Job) is broken — known, not yet fixed

Seen 2026-09-29 against `vervux-platform-prod`: its last three executions all
failed (`agent-migrate-4sqmg`, `-gtvsz` on 2026-09-24, `-2b597` on 2026-09-29),
and `yorch-gcp-platform/scripts/migrate.sh yorchio` runs it after every backend
migration, so every such run ends with a failed second job.

Why, from the job's own logs:
- **It starts the server, not a migration.** `envs/prod/jobs.tf` (`module
  "agent_migrate"`) sets no `command`/`args` — unlike `rocky-migrate` /
  `yorchio-migrate` (`npx prisma migrate deploy`) — so it runs this image's
  `CMD`, which is `uvicorn src.main:app`.
- **It dies at config validation first:** the job only gets `DATABASE_URL`, and
  `Settings._require_serper_api_key` refuses to boot without `SERPER_API_KEY`
  (`Value error, SERPER_API_KEY is not set`), `exit(1)`.
- Even with the key it would be wrong: uvicorn never exits, so the job would run
  until its timeout.

Why nothing breaks: this service applies its own schema at startup —
`lifespan` → `src/db/postgres.py::run_migrations` (`migrations/init.sql`,
statement by statement) plus the checkpointer/store `setup()`. The job adds
nothing a deploy does not already do.

One statement also fails at every startup, logged as `migration_statement_failed`:
`CREATE INDEX IF NOT EXISTS approval_requests_thread_status_idx` → `must be owner
of table approval_requests`. The runtime role is not the table owner (the
migrator is), so that index is never created by the service. Harmless today;
running the migration as the migrator is what would fix it.

To fix (pick one): give the job a real one-shot entrypoint (e.g. a small
`python -m src.db.migrate` that opens the pool, calls `run_migrations` and
exits) plus the env `Settings` requires, or delete the job and the
`agent-migrate` step from `migrate.sh`.

## Environment

Copy `.env.example` to `.env`. Required vars:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `SERPER_API_KEY` | Web search for the prospecting agent — a startup validator refuses to boot without it. It checks PRESENCE only: a set-but-invalid key boots fine and then gets `403 Unauthorized` on every `/search` and `/places` call, which `_serper_post` now turns into a fatal `SerperAuthError` so the run ends FAILED instead of reporting `found: 0` (as it silently did before 2026-08-18) |

Optional: `SERVICE_AUTH_AUDIENCE` / `SERVICE_AUTH_ALLOWED_SERVICE_ACCOUNTS` / `ALLOW_SHARED_SECRET_AUTH` (see Authentication — leave the audience empty locally), `WEBHOOK_API_KEY`, `NESTJS_BASE_URL`, `OPENAI_API_KEY` (fallback; per-request key preferred), `GEMINI_*` (absent, the Gemini provider falls back to Application Default Credentials — which is the only auth Agent Platform accepts; it rejects API keys), `LANGFUSE_*`.

## Testing

Tests use `MemorySaver` (no database needed). `conftest.py` sets dummy env vars (`DATABASE_URL`, `SERPER_API_KEY`) and an empty `SERVICE_AUTH_AUDIENCE` before any app modules import, so the suite runs against the shared-secret path without reaching Google's JWKS endpoint. `asyncio_mode = "auto"` in `pyproject.toml` — no need to mark individual tests with `@pytest.mark.asyncio`.

Key test patterns: mock individual agent nodes (e.g., `patch("src.graphs.main_graph.triage_node")`), build a fresh graph with `build_graph(MemorySaver())`, stream with `graph.astream()`, and inspect chunks. Tests verify routing logic, API key security (no leaks into state/interrupts), and auto-chaining flows.

## Studio Graph

`src/graphs/studio_graph.py` is the LangGraph Studio entrypoint — uses an in-memory checkpointer and is not used in production.
