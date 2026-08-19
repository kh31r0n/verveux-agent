import json
import logging
import traceback
import uuid
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # used in lifespan
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.postgres.aio import AsyncPostgresStore  # long-term prospecting memory
from langgraph.types import Command
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

import asyncio

from .auth.service_auth import (
    get_current_user,
    scoped_thread_id,
    verify_service_caller,
)
from .agents.backend_client import (
    fetch_active_code_names,
    fetch_agent_credentials,
    fetch_in_use_code_names,
    report_enrichment_attempt,
    report_prospecting_run,
)
from .agents.prospecting_nodes import DEFAULT_LOCATION
from .services.serper import serper_call_count, start_serper_accounting
from .config import settings
from .db.postgres import (
    close_pool,
    get_pool,
    init_pool,
    run_migrations,
)
from .graphs.registry import (
    CODE_NAME_REGISTRY,
    UnknownCodeNameError,
    get_or_compile_graph,
    known_code_names,
    resolve_legacy_agent_type,
    set_checkpointer,
    set_store,
    warm_up,
)
from .observability import (
    agent_interrupt_events_total,
    agent_requests_total,
    legacy_agent_type_fallback_total,
    record_tool_error,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


async def _fetch_active_code_names_with_retry(
    max_attempts: int = 5,
    base_delay: float = 1.0,
) -> set[str]:
    """Backend may not yet be reachable when the agent boots (start ordering in
    ECS/Compose). Retry the validation call a few times before giving up."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fetch_active_code_names()
        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "active_code_names_fetch_retry",
                attempt=attempt,
                max_attempts=max_attempts,
                next_delay_seconds=delay,
                error=str(exc),
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise application DB pool (LangGraph checkpoints + RAG) and run migrations.
    pool = await init_pool()
    await run_migrations(pool)

    # Checkpoints written before 2026-07 contain StructuredIntent/IntentType
    # Pydantic objects; new code stores plain dicts. Allow-list the legacy
    # types so old threads keep deserializing after LangGraph starts blocking
    # unregistered classes. Do not extend this list — state must stay JSON-native.
    legacy_serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("src.schemas.intent", "IntentType"),
            ("src.schemas.intent", "StructuredIntent"),
        ]
    )
    async with AsyncExitStack() as stack:
        checkpointer = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(
                settings.database_url, serde=legacy_serde
            )
        )
        await checkpointer.setup()
        set_checkpointer(checkpointer)

        # Long-term store for the prospecting agent's cross-run strategy memory.
        # Separate from the checkpointer (which is per-run/per-thread). Best
        # effort: if it can't initialise, aurora falls back to stateless runs.
        try:
            store = await stack.enter_async_context(
                AsyncPostgresStore.from_conn_string(settings.database_url)
            )
            await store.setup()
            set_store(store)
            logger.info("prospecting_store_ready")
        except Exception as exc:  # noqa: BLE001 — store is optional; never block boot
            logger.error("prospecting_store_init_failed", error=str(exc))

        registry = known_code_names()

        # ── Startup validation: backend ⇄ registry ────────────────────────────
        # The source of truth for agent_code_names / channel_connections is the
        # NestJS backend's DB. We consume it via internal HTTP endpoints rather
        # than opening a second DB pool, so the agent stays decoupled from the
        # backend's schema and credentials.
        try:
            db_active = await _fetch_active_code_names_with_retry()
        except Exception as exc:
            logger.error("startup_validation_query_failed", error=str(exc))
            raise

        missing_in_registry = db_active - registry
        if missing_in_registry:
            raise SystemExit(
                "Startup validation failed: agent_code_names has active "
                f"entries not in CODE_NAME_REGISTRY: {sorted(missing_in_registry)}. "
                "Deploy a Python version that includes builders for these code names."
            )
        missing_in_db = registry - db_active
        if missing_in_db:
            logger.warning(
                "registry_codenames_not_in_db",
                code_names=sorted(missing_in_db),
            )

        # ── Warm-up: compile only the code names actually assigned somewhere ──
        try:
            in_use = await fetch_in_use_code_names()
        except Exception as exc:
            logger.warning("warm_up_query_failed", error=str(exc))
            in_use = []

        compiled = await warm_up(in_use, checkpointer)
        logger.info(
            "langgraph_ready",
            warm_compiled=compiled,
            registered=sorted(registry),
        )
        yield

    await close_pool()
    logger.info("shutdown_complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Helena Agent Service",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class PromptPayload(BaseModel):
    content: str = ""
    version: int = 0
    model_config_data: dict = {}
    is_default: bool = True
    # AiPrompt row uuid from NestJS; "" when the slot is a platform default.
    # Provenance-only — resolution reads `content` and ignores this field.
    id: str = ""

    model_config = {"populate_by_name": True}


class ChatStreamRequest(BaseModel):
    thread_id: str
    message: str
    openai_api_key: str = ""
    tenant_id: str = ""
    conversation_id: str = ""
    product_catalog: list = []
    user_context: dict = {}
    contact_id: str = ""
    contact_tags: list = []
    language: str = "en"
    prompts: dict[str, PromptPayload] = {}
    knowledge: list = []
    rawFaqs: list = []
    rawCatalog: list = []
    llm_provider: str = "openai"
    llm_model: str = ""
    anthropic_api_key: str = ""
    gemini_credentials: dict = {}
    gemini_project_id: str = ""
    gemini_location: str = ""
    # ── Multi-agent versioning ───────────────────────────────────────────────
    # `agent_code_name` is the canonical routing key (e.g. "helena"). Phase-1
    # callers may omit it; we fall back to `agent_type` via the registry shim
    # and emit `legacy_agent_type_fallback_total`. In phase 2 the empty default
    # becomes a hard 400.
    agent_code_name: str = ""
    # Custom persona the agent uses to introduce itself on this channel
    # (e.g. "Helena", "Admisiones"). Empty = use the graph's default identity.
    # Snapshotted on the conversation by the backend, so this value is stable
    # across turns even if the channel's persona changes later.
    agent_persona_name: str = ""
    agent_version: int = 1
    agent_type: str = "sales"
    capabilities: list = []
    # Per-turn idempotency key generated by NestJS. Threaded through
    # configurable and echoed back on the SSE `done` event so the backend
    # can deduplicate retries when persisting AiInvocationUsage rows.
    turn_request_id: str = ""
    # Non-text content carried by the inbound message — populated by NestJS
    # when content.type is not text/audio. Only camila currently inspects it.
    attachments: list = []
    # Per-tenant rollout flag for the query_normalizer node, read from
    # TenantSettings.queryNormalizationEnabled by the backend. Combined with
    # the fleet-wide settings.query_normalization_enabled env switch.
    query_normalization_enabled: bool = False
    # Tenant-admin CATALOG capability toggle, resolved fresh by the backend
    # every turn (AgentCapabilityPolicyService). Default True keeps behavior
    # identical for older backends that don't send it. When False the backend
    # already sent an empty product_catalog and the internal cart/order
    # endpoints 403 — this flag drives the Python-side prompt gate + degraded
    # instruction so the model never invents products, prices, or orders.
    catalog_access_enabled: bool = True
    # Business-hours flag, resolved fresh by the backend every turn
    # (WorkingHoursService via TenantSettings.agentBusinessHoursEnabled).
    # Default True keeps behavior identical for older backends that don't
    # send it. When False, graphs route every non-urgent turn to
    # faq_response, which splices the {AGENT_TYPE}_OUTSIDE_HOURS prompt.
    within_business_hours: bool = True



class ChatResumeRequest(BaseModel):
    thread_id: str
    interrupt_id: str
    approved: bool
    openai_api_key: str = ""
    tenant_id: str = ""
    conversation_id: str = ""
    agent_code_name: str = ""
    agent_version: int = 1
    agent_type: str = "sales"


class ProspectingRunRequest(BaseModel):
    """Backend-scheduler trigger for one autonomous prospecting run (aurora).

    Authenticated like every other route, by verify_service_caller — a Google
    OIDC ID token or the shared secret (here usually spelled x-agent-key).
    Credentials are NOT forwarded; the agent resolves the tenant's LLM key
    itself via fetch_agent_credentials, exactly as /chat/stream does.
    """

    tenant_id: str
    run_id: str
    run_date: str = ""
    agent_code_name: str = "aurora"
    prompts: dict[str, PromptPayload] = {}
    # Tenant-configurable targeting.
    # niche:    {key, label, search_terms: [...]}          — WHAT to search for
    # location: {country, gl?, hl?, cities: [...]}          — WHERE to search
    #
    # `niche` is REQUIRED (validated below): aurora is industry-agnostic, and a
    # built-in fallback would silently prospect an industry the tenant never
    # asked for. `location` stays optional — a country default is a reach
    # setting, not a claim about the tenant's business.
    niche: dict | None = None
    location: dict | None = None


class EnrichmentRunRequest(BaseModel):
    """Backend-scheduler trigger for one contact website enrichment (sherlock).

    Like the prospecting trigger this is a service-to-service call authenticated
    by verify_service_caller, and credentials are resolved by the agent itself
    via fetch_agent_credentials.

    `website_url` is a tenant-editable field, so every fetch derived from it goes
    through the SSRF-hardened `services.web_fetch`. `contact_country` is passed
    through untouched — the backend uses it to validate phone candidates as
    E.164, the agent never normalizes numbers itself.
    """

    tenant_id: str
    attempt_id: str
    contact_id: str
    # Empty when the contact has no website on file. The graph then discovers one
    # from `contact_name` + `contact_city`/`contact_country` and reports it back;
    # the backend writes it onto the contact.
    website_url: str = ""
    contact_country: str = ""
    contact_city: str = ""
    contact_name: str = ""
    # Output language for the generated description/strategy, from
    # TenantSettings.language. Defaults to Spanish when the backend omits it.
    language: str = "es"
    agent_code_name: str = "sherlock"
    prompts: dict[str, PromptPayload] = {}
    # The TENANT'S own commercial profile (TenantSettings.enrichmentIcp):
    # {industry, business_description, ideal_customer, disqualifiers[]}. Used to
    # judge the prospect's FIT and to write the sales strategy against what this
    # tenant actually sells — the prompts used to hardcode one offering for
    # everyone. Deliberately carries NO prices: the backend alone maps fit
    # drivers onto a price tier. Empty for older backends, which degrades to the
    # prompt's own generic wording rather than failing.
    icp: dict = {}
    # Serper geo hints from the tenant's prospecting configuration
    # ({country, gl, hl}). A prospect row carries a city but never a country, so
    # this is the only country signal website discovery has.
    discovery_location: dict = {}


# ---------------------------------------------------------------------------
# Multi-message coalescing
# ---------------------------------------------------------------------------
# WhatsApp users often split one thought across several rapid messages, and
# NestJS forwards each as its own POST /chat/stream. Without coordination the
# runs execute concurrently against the same checkpoint thread and each one
# replies to its own fragment in isolation (three greetings for three
# messages). We serialize runs per thread and coalesce: fragments that arrive
# while a turn is waiting or running are merged into the NEXT run as a single
# user turn, and their own requests complete with an empty `done` event so the
# backend releases their credit reservations and sends nothing.
#
# The locks are process-local — correct for the single-process deployment.
# If the agent is ever scaled horizontally, requests must be routed sticky by
# thread_id (or this moves to a Postgres advisory lock).


class _ThreadTurnState:
    __slots__ = ("lock", "pending", "requests")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        # Fragments not yet consumed by a graph run:
        # (message, attachments, faqs) — faqs is the per-request FAQ payload
        # NestJS retrieved for that fragment.
        self.pending: list[tuple[str, list, list]] = []
        # Number of requests currently referencing this state (for cleanup)
        self.requests = 0


_thread_turns: dict[str, _ThreadTurnState] = {}


def _checkout_turn_state(thread_id: str) -> _ThreadTurnState:
    state = _thread_turns.get(thread_id)
    if state is None:
        state = _ThreadTurnState()
        _thread_turns[thread_id] = state
    state.requests += 1
    return state


def _checkin_turn_state(thread_id: str, state: _ThreadTurnState) -> None:
    state.requests -= 1
    if state.requests <= 0 and not state.pending:
        _thread_turns.pop(thread_id, None)


async def _coalesced_stream(
    message: str,
    attachments: list,
    faqs: list,
    inputs: dict,
    config: dict,
    graph,
    agent_code_name: str,
    thread_id: str,
) -> AsyncGenerator[str, None]:
    """Serialize graph runs per thread and merge rapid message bursts.

    Wraps _stream_graph. The request that wins the thread's run slot answers
    every fragment buffered so far in one turn; superseded requests emit an
    empty `done` event (turn_usage=[]) so NestJS releases their reservation
    and dispatches no reply.
    """
    state = _checkout_turn_state(thread_id)
    state.pending.append((message, attachments, faqs))
    try:
        async with state.lock:
            # Settle window: keep waiting while new fragments are still
            # arriving, so the whole burst is answered as one turn.
            settle = settings.message_settle_seconds
            if settle > 0:
                waited = 0.0
                while waited < settings.message_settle_max_seconds:
                    seen = len(state.pending)
                    await asyncio.sleep(settle)
                    waited += settle
                    if len(state.pending) == seen:
                        break

            if not state.pending:
                # A sibling request already carried this fragment into its own
                # run and answered it.
                logger.info("chat_stream_superseded", thread_id=thread_id)
                yield _sse_event({
                    "type": "done",
                    "turn_request_id": config.get("configurable", {}).get(
                        "turn_request_id", ""
                    ),
                    "turn_usage": [],
                    "mentioned_product_ids": [],
                    "coalesced": True,
                })
                return

            fragments = state.pending[:]
            state.pending.clear()
            if len(fragments) > 1:
                logger.info(
                    "chat_stream_coalesced",
                    thread_id=thread_id,
                    fragments=len(fragments),
                )
            merged_text = "\n".join(text for text, _, _ in fragments if text)
            merged_attachments = [a for _, atts, _ in fragments for a in atts]
            # Union of every fragment's FAQ payload, deduped by question.
            # NestJS retrieves FAQs per request (querying the burst text it
            # has seen so far), so later fragments carry matches the winning
            # (first) request's payload lacks.
            merged_faqs: list = []
            seen_questions: set[str] = set()
            for _, _, faq_list in fragments:
                for faq in faq_list or []:
                    question = (faq.get("question") or "").strip().lower()
                    if question in seen_questions:
                        continue
                    if question:
                        seen_questions.add(question)
                    merged_faqs.append(faq)
            inputs["messages"] = [HumanMessage(content=merged_text)]
            inputs["attachments"] = merged_attachments
            inputs["faqs"] = merged_faqs

            async for event in _stream_graph(
                inputs, config, graph=graph, agent_code_name=agent_code_name
            ):
                yield event
    finally:
        _checkin_turn_state(thread_id, state)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


_SECRET_KEYS = {
    "openai_api_key",
    "anthropic_api_key",
    "gemini_credentials",
    "gemini_service_account_json",
    "llm_credentials",
}


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_graph(
    inputs: dict | Command,
    config: dict,
    graph=None,
    agent_code_name: str = "unknown",
) -> AsyncGenerator[str, None]:
    """Consume a graph.astream() and yield SSE-formatted strings."""
    if graph is None:
        yield _sse_event({"type": "error", "message": "Agent graph not initialised"})
        return

    # turn_usage is an operator.add channel, so the checkpointer accumulates
    # it across turns on the same thread. The done event must report only the
    # records THIS run appended — otherwise NestJS re-persists every prior
    # turn's invocations under the new turn_request_id (billing over-count).
    # Snapshot the pre-run length and slice after the run; this also keeps
    # /chat/resume correct, where a Command input can't reset any channel.
    prev_usage_count = 0
    try:
        prior = await graph.aget_state(config)
        prev_usage_count = len(prior.values.get("turn_usage") or [])
    except Exception as exc:  # noqa: BLE001 — a fresh thread has no checkpoint
        logger.debug("pre_run_state_fetch_failed", error=str(exc))

    try:
        async for chunk in graph.astream(
            inputs,
            config=config,
            stream_mode=["updates", "custom"],
            version="v2",
        ):
            chunk_type: str = chunk.get("type", "")

            if chunk_type == "custom":
                # Real-time events written by nodes via get_stream_writer()
                event_data: dict = chunk.get("data", {})
                event_kind: str = event_data.get("type", "")

                if event_kind == "token":
                    content: str = event_data.get("content", "")
                    if content:
                        yield _sse_event({"type": "token", "content": content})

                elif event_kind == "step_progress":
                    yield _sse_event({
                        "type": "step_progress",
                        "step": event_data.get("step"),
                        "total_steps": event_data.get("total_steps"),
                        "topic": event_data.get("topic", ""),
                    })

                elif event_kind == "execute_workflow":
                    yield _sse_event({
                        "type": "execute_workflow",
                        "conversation_id": event_data.get("conversation_id", ""),
                        "intent": event_data.get("intent", ""),
                        "order_data": event_data.get("order_data", {}),
                        "tracking_data": event_data.get("tracking_data", {}),
                        "complaint_data": event_data.get("complaint_data", {}),
                    })

                elif event_kind == "tag_contact":
                    yield _sse_event({
                        "type": "tag_contact",
                        "contact_id": event_data.get("contact_id", ""),
                        "tag_name": event_data.get("tag_name", ""),
                    })

                elif event_kind == "create_deal":
                    yield _sse_event({
                        "type": "create_deal",
                        "contact_id": event_data.get("contact_id", ""),
                        "conversation_id": event_data.get("conversation_id", ""),
                        "title": event_data.get("title", ""),
                        "source": event_data.get("source", "WHATSAPP"),
                    })

                elif event_kind == "update_deal_stage":
                    yield _sse_event({
                        "type": "update_deal_stage",
                        "contact_id": event_data.get("contact_id", ""),
                        "conversation_id": event_data.get("conversation_id", ""),
                        "stage_position": event_data.get("stage_position", 0),
                    })

            elif chunk_type == "updates":
                update_data: dict = chunk.get("data", {})

                # Check for interrupt
                interrupt_list = update_data.get("__interrupt__")
                if interrupt_list:
                    for interrupt_item in interrupt_list:
                        interrupt_id = str(uuid.uuid4())
                        thread_id_value: str = config["configurable"]["thread_id"]
                        payload = (
                            interrupt_item.value
                            if hasattr(interrupt_item, "value")
                            else interrupt_item
                        )

                        # Persist audit record
                        try:
                            pool = await get_pool()
                            await pool.execute(
                                """
                                INSERT INTO approval_requests (id, thread_id, payload, status)
                                VALUES ($1, $2, $3, 'pending')
                                """,
                                uuid.UUID(interrupt_id),
                                thread_id_value,
                                json.dumps(payload),
                            )
                        except Exception as db_exc:
                            logger.error(
                                "interrupt_audit_insert_failed",
                                error=str(db_exc),
                                thread_id=thread_id_value,
                            )
                            record_tool_error("audit_db", agent_code_name)

                        agent_interrupt_events_total.inc()
                        logger.info(
                            "interrupt_emitted",
                            interrupt_id=interrupt_id,
                            thread_id=thread_id_value,
                        )

                        yield _sse_event(
                            {
                                "type": "interrupt_detected",
                                "thread_id": thread_id_value,
                                "interrupt_id": interrupt_id,
                                "payload": payload,
                            }
                        )
                else:
                    # Emit regular node update (strip sensitive data)
                    for node_name, node_data in update_data.items():
                        if node_name.startswith("__"):
                            continue
                        safe_data: dict = {}
                        if isinstance(node_data, dict):
                            for k, v in node_data.items():
                                if k == "messages":
                                    safe_data["messages"] = [
                                        {
                                            "type": getattr(m, "type", "unknown"),
                                            "content": (
                                                m.content
                                                if hasattr(m, "content")
                                                else str(m)
                                            ),
                                        }
                                        for m in v
                                    ] if isinstance(v, list) else []
                                if k not in _SECRET_KEYS:
                                    try:
                                        json.dumps(v)  # verify serialisable
                                        safe_data[k] = v
                                    except (TypeError, ValueError):
                                        pass

                        yield _sse_event(
                            {"type": "node_update", "node": node_name, "data": safe_data}
                        )

        # ── Final event: emit token usage for the turn ────────────────────
        # NestJS reads turn_usage off this event and persists it as
        # AiInvocationUsage rows. turn_request_id is echoed back as-is so
        # the backend can drop duplicate runs. mentioned_product_ids is
        # consumed by NestJS to decide whether to attach a product image
        # to the outbound WhatsApp message (SALES only today).
        turn_usage: list = []
        mentioned_product_ids: list = []
        faq_used: list = []
        try:
            final_state = await graph.aget_state(config)
            all_usage = final_state.values.get("turn_usage", []) or []
            turn_usage = all_usage[prev_usage_count:]
            mentioned_product_ids = (
                final_state.values.get("mentioned_product_ids", []) or []
            )
            faq_used = final_state.values.get("faq_used") or []
        except Exception as exc:
            logger.warning("done_event_state_fetch_failed", error=str(exc))

        yield _sse_event({
            "type": "done",
            "turn_request_id": config.get("configurable", {}).get("turn_request_id", ""),
            "turn_usage": turn_usage,
            "mentioned_product_ids": mentioned_product_ids,
            "faq_used": faq_used,
        })

    except Exception as exc:
        logger.error("stream_error", error=str(exc), traceback=traceback.format_exc())
        record_tool_error("graph_stream", agent_code_name)
        yield _sse_event({"type": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
@app.get("/healthz")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# Strong references to in-flight background prospecting runs (asyncio.create_task
# only holds a weak ref, so without this the task can be GC'd mid-run).
_prospecting_tasks: set[asyncio.Task] = set()


async def _run_prospecting(
    graph, inputs: dict, config: dict, run_id: str
) -> None:
    """Run one prospecting graph to completion in the background.

    The graph's ``report`` node reports COMPLETED on success. If the run raises
    before reaching it, mark the run FAILED here so the backend's daily lock is
    released and the (bounded) usage that did happen is not lost. The backend
    reaper is the final backstop if even this report never lands.
    """
    # Install the run-scoped Serper credit counter HERE, in the parent
    # coroutine: LangGraph node tasks each get a copy of the context, so a
    # ContextVar set inside a node would never be visible to `report_node`.
    start_serper_accounting()
    try:
        await graph.ainvoke(inputs, config)
    except Exception as exc:  # noqa: BLE001 — background task must not crash the loop
        logger.error(
            "prospecting_run_failed",
            run_id=run_id,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        try:
            await report_prospecting_run(
                run_id,
                "FAILED",
                # Serper credits were spent even though the run failed, so the
                # platform cost report must still see them.
                metrics={
                    "reason": f"agent_error: {exc}",
                    "serper_calls": serper_call_count(),
                },
            )
        except Exception:  # noqa: BLE001
            logger.error("prospecting_fail_report_failed", run_id=run_id)


def prospecting_thread_id(tenant_id: str, run_id: str, run_date: str) -> str:
    """Checkpoint thread for one prospecting run — keyed on run_id, NOT run_date.

    `candidates`, `searched_queries` and `seen_urls` are `operator.add` channels:
    they accumulate and persist, and an input value cannot reset a reduced
    channel. Keying the thread by day therefore made every second run of the same
    date inherit the previous run's candidates — re-posting them to the CRM,
    re-inflating `found`/`duplicates`, and replaying stale rows extracted by an
    older agent build. Observed live: found 147 → 231 → 304 across three runs of
    one day, with an unchanging 26 create_errors.

    run_id is the right key because `retryFailedRun` re-arms the SAME run row, so
    a genuine resume still finds its checkpoint while a distinct run starts
    clean. run_date is the fallback for older backends that omit run_id.
    """
    return f"prospecting:{tenant_id}:{run_id or run_date}"


@app.post("/prospecting/run", status_code=202)
async def prospecting_run(
    req: ProspectingRunRequest,
    _auth: Annotated[dict, Depends(verify_service_caller)],
) -> dict:
    code_name = (req.agent_code_name or "aurora").strip().lower()
    try:
        graph = await get_or_compile_graph(code_name)
    except UnknownCodeNameError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Reject before claiming any work: with no niche the graph has nothing to
    # search for, and the alternative — a built-in fallback — would prospect an
    # industry the tenant never configured. The backend's scheduler skips these
    # tenants (`niche_not_configured`), so this is the contract's backstop.
    niche_terms = (req.niche or {}).get("search_terms")
    if not (req.niche or {}).get("key") or not (
        isinstance(niche_terms, list)
        and any(isinstance(t, str) and t.strip() for t in niche_terms)
    ):
        raise HTTPException(
            status_code=422,
            detail="niche requires a non-empty 'key' and at least one 'search_terms' entry",
        )

    # Resolve LLM credentials ourselves — there is no NestJS-forwarded key on a
    # scheduler-triggered run (mirrors the /chat/stream credential fetch).
    llm_provider = "openai"
    llm_model = ""
    provider_config: dict = {}
    try:
        creds = await fetch_agent_credentials(req.tenant_id)
        llm_provider = creds.get("provider", "OPENAI").lower()
        llm_model = creds.get("model") or ""
        if llm_provider == "openai":
            provider_config["openai_api_key"] = creds.get("apiKey", "")
        elif llm_provider == "anthropic":
            provider_config["anthropic_api_key"] = creds.get("apiKey", "")
        elif llm_provider == "gemini":
            provider_config["gemini_credentials"] = creds.get("geminiCredentials", {})
            provider_config["gemini_project_id"] = creds.get("geminiProjectId", "")
            provider_config["gemini_location"] = creds.get("geminiLocation", "")
    except Exception as exc:
        logger.warning(
            "prospecting_credentials_fetch_failed",
            tenant_id=req.tenant_id,
            error=str(exc),
        )
        provider_config["openai_api_key"] = settings.openai_api_key

    prompts_dict = (
        {k: v.model_dump() for k, v in req.prompts.items()} if req.prompts else {}
    )
    thread_id = prospecting_thread_id(req.tenant_id, req.run_id, req.run_date)
    config: dict = {
        "configurable": {
            "thread_id": thread_id,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "prompts": prompts_dict,
            **provider_config,
        }
    }
    # Bound the extract_and_enrich Send fan-out ONLY for Gemini, whose per-model
    # RPM quota the ~90-way parallel burst exhausts (429 RESOURCE_EXHAUSTED).
    # openai/anthropic keep LangGraph's default unbounded parallelism.
    if llm_provider == "gemini":
        config["max_concurrency"] = settings.prospecting_gemini_extract_concurrency
    # The refinement loop adds ~5 supersteps per extra iteration on top of the
    # base flow; lift the recursion limit so a high max_iterations can't trip
    # LangGraph's default of 25.
    config["recursion_limit"] = 15 + settings.prospecting_max_iterations * 8
    inputs: dict = {
        "tenant_id": req.tenant_id,
        "run_id": req.run_id,
        "run_date": req.run_date,
        "niche": req.niche,
        # Location alone keeps a default — see ProspectingRunRequest.
        "location": req.location or DEFAULT_LOCATION,
    }

    # Fire-and-forget: the run streams no reply, and the scheduler already holds
    # the daily lock, so return 202 immediately and work in the background.
    # Keep a strong reference so the event loop doesn't GC the pending task.
    task = asyncio.create_task(_run_prospecting(graph, inputs, config, req.run_id))
    _prospecting_tasks.add(task)
    task.add_done_callback(_prospecting_tasks.discard)
    logger.info(
        "prospecting_run_accepted",
        run_id=req.run_id,
        tenant_id=req.tenant_id,
        code_name=code_name,
    )
    return {"accepted": True, "run_id": req.run_id}


# Strong references to in-flight background enrichment runs (asyncio.create_task
# only holds a weak ref, so without this the task can be GC'd mid-run).
_enrichment_tasks: set[asyncio.Task] = set()


async def _run_enrichment(
    graph, inputs: dict, config: dict, attempt_id: str
) -> None:
    """Run one enrichment graph to completion in the background.

    The graph's ``report`` node reports COMPLETED/NO_RESULT on success. If the run
    raises before reaching it, mark the attempt FAILED here so the backend does
    not have to wait for its stale reaper. The reaper remains the final backstop
    if even this report never lands.
    """
    # Must be installed BEFORE ainvoke: each node runs in a child task with a
    # COPY of the context, so a counter created inside a node would be invisible
    # here and in `report_node` (see services.serper).
    start_serper_accounting()
    try:
        await graph.ainvoke(inputs, config)
    except Exception as exc:  # noqa: BLE001 — background task must not crash the loop
        logger.error(
            "enrichment_run_failed",
            attempt_id=attempt_id,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        try:
            await report_enrichment_attempt(
                attempt_id,
                "FAILED",
                error=f"agent_error: {exc}",
                # Website discovery may already have spent Serper credits before
                # the run died; the attempt row is where that is accounted for.
                metrics={"serperCalls": serper_call_count()},
            )
        except Exception:  # noqa: BLE001
            logger.error("enrichment_fail_report_failed", attempt_id=attempt_id)


@app.post("/enrichment/run", status_code=202)
async def enrichment_run(
    req: EnrichmentRunRequest,
    _auth: Annotated[dict, Depends(verify_service_caller)],
) -> dict:
    code_name = (req.agent_code_name or "sherlock").strip().lower()
    try:
        graph = await get_or_compile_graph(code_name)
    except UnknownCodeNameError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Resolve LLM credentials ourselves — there is no NestJS-forwarded key on a
    # scheduler-triggered run (mirrors the prospecting credential fetch).
    llm_provider = "openai"
    llm_model = ""
    provider_config: dict = {}
    try:
        creds = await fetch_agent_credentials(req.tenant_id)
        llm_provider = creds.get("provider", "OPENAI").lower()
        llm_model = creds.get("model") or ""
        if llm_provider == "openai":
            provider_config["openai_api_key"] = creds.get("apiKey", "")
        elif llm_provider == "anthropic":
            provider_config["anthropic_api_key"] = creds.get("apiKey", "")
        elif llm_provider == "gemini":
            provider_config["gemini_credentials"] = creds.get("geminiCredentials", {})
            provider_config["gemini_project_id"] = creds.get("geminiProjectId", "")
            provider_config["gemini_location"] = creds.get("geminiLocation", "")
    except Exception as exc:
        logger.warning(
            "enrichment_credentials_fetch_failed",
            tenant_id=req.tenant_id,
            error=str(exc),
        )
        provider_config["openai_api_key"] = settings.openai_api_key

    prompts_dict = (
        {k: v.model_dump() for k, v in req.prompts.items()} if req.prompts else {}
    )
    # One checkpoint thread per attempt. Attempts are one-shot by design, so the
    # thread is effectively single-use; keying on the attempt id means a crashed
    # run that the backend re-dispatches would resume rather than restart.
    thread_id = f"enrichment:{req.tenant_id}:{req.attempt_id}"
    config: dict = {
        "configurable": {
            "thread_id": thread_id,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "prompts": prompts_dict,
            **provider_config,
        }
    }
    # The refinement loop adds ~3 supersteps per extra iteration on top of the
    # base flow; lift the recursion limit so a high max_iterations can't trip
    # LangGraph's default of 25.
    # +2 supersteps for the discovery branch (route + node) on top of the base.
    config["recursion_limit"] = 14 + settings.sherlock_max_iterations * 6
    inputs: dict = {
        "tenant_id": req.tenant_id,
        "attempt_id": req.attempt_id,
        "contact_id": req.contact_id,
        "website_url": req.website_url,
        "contact_country": req.contact_country,
        "contact_city": req.contact_city,
        "contact_name": req.contact_name,
        "language": req.language or "es",
        "icp": req.icp or {},
        "discovery_location": req.discovery_location or {},
    }

    # Fire-and-forget: the run streams no reply and the scheduler already holds
    # the claim, so return 202 immediately and work in the background.
    task = asyncio.create_task(
        _run_enrichment(graph, inputs, config, req.attempt_id)
    )
    _enrichment_tasks.add(task)
    task.add_done_callback(_enrichment_tasks.discard)
    logger.info(
        "enrichment_run_accepted",
        attempt_id=req.attempt_id,
        tenant_id=req.tenant_id,
        contact_id=req.contact_id,
        code_name=code_name,
    )
    return {"accepted": True, "attempt_id": req.attempt_id}


@app.post("/chat/stream")
async def chat_stream(
    req: ChatStreamRequest,
    user_sub: Annotated[str, Depends(get_current_user)],
) -> StreamingResponse:
    agent_type = (req.agent_type or "").lower()
    code_name = (req.agent_code_name or "").strip().lower()
    if not code_name:
        code_name = resolve_legacy_agent_type(agent_type) or ""
        if code_name:
            legacy_agent_type_fallback_total.inc()
            logger.warning(
                "legacy_agent_type_fallback",
                agent_type=agent_type,
                resolved=code_name,
            )
    if not code_name:
        raise HTTPException(
            status_code=400,
            detail=(
                f"agent_code_name is required; agent_type={agent_type!r} is "
                f"not mappable. Known code names: {sorted(CODE_NAME_REGISTRY)}"
            ),
        )

    try:
        graph = await get_or_compile_graph(code_name)
    except UnknownCodeNameError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    thread_id = scoped_thread_id(
        req.tenant_id,
        user_sub,
        req.conversation_id,
        code_name,
        req.agent_version,
    )
    agent_requests_total.labels(agent_code_name=code_name).inc()

    # ── Resolve credentials from NestJS ──────────────────────────────────────
    # We fetch once per request. The result is injected into configurable and
    # never logged or emitted via SSE (stripped by _SECRET_KEYS).
    llm_provider = req.llm_provider or "openai"
    llm_model = req.llm_model or ""
    provider_config: dict = {}

    if req.tenant_id:
        try:
            creds = await fetch_agent_credentials(req.tenant_id)
            llm_provider = creds.get("provider", "OPENAI").lower()
            llm_model = creds.get("model") or llm_model

            if llm_provider == "openai":
                provider_config["openai_api_key"] = creds.get("apiKey", "")
            elif llm_provider == "anthropic":
                provider_config["anthropic_api_key"] = creds.get("apiKey", "")
            elif llm_provider == "gemini":
                provider_config["gemini_credentials"] = creds.get("geminiCredentials", {})
                provider_config["gemini_project_id"] = creds.get("geminiProjectId", "")
                provider_config["gemini_location"] = creds.get("geminiLocation", "")
        except Exception as exc:
            logger.warning(
                "agent_credentials_fetch_failed",
                tenant_id=req.tenant_id,
                error=str(exc),
            )
            # Fall back to per-request keys sent by the frontend (legacy path)
            provider_config = {
                "openai_api_key": req.openai_api_key,
                "anthropic_api_key": req.anthropic_api_key,
                "gemini_credentials": req.gemini_credentials,
                "gemini_project_id": req.gemini_project_id,
                "gemini_location": req.gemini_location,
            }
    else:
        # No tenantId — use legacy per-request keys
        provider_config = {
            "openai_api_key": req.openai_api_key,
            "anthropic_api_key": req.anthropic_api_key,
            "gemini_credentials": req.gemini_credentials,
            "gemini_project_id": req.gemini_project_id,
            "gemini_location": req.gemini_location,
        }

    prompts_dict = {k: v.model_dump() for k, v in req.prompts.items()} if req.prompts else {}

    config: dict = {
        "configurable": {
            "thread_id": thread_id,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "prompts": prompts_dict,
            "turn_request_id": req.turn_request_id,
            "normalization_enabled": req.query_normalization_enabled,
            **provider_config,   # injects the right keys for the resolved provider
        }
    }

    inputs: dict = {
        "messages": [HumanMessage(content=req.message)],
        "thread_id": thread_id,
        "tenant_id": req.tenant_id,
        "conversation_id": req.conversation_id,
        "agent_type": agent_type or "",
        "agent_code_name": code_name,
        "agent_persona_name": req.agent_persona_name,
        "agent_version": req.agent_version,
        "capabilities": {"agent_type": agent_type, "capabilities": req.capabilities},
        "domain_state": {},
        "product_catalog": req.product_catalog,
        # Overwritten every turn (like product_catalog) so an admin flipping the
        # CATALOG toggle takes effect on the next turn — the checkpointer can't
        # leak a stale ON value.
        "catalog_access_enabled": req.catalog_access_enabled,
        # Same per-turn overwrite rationale: business hours can flip between
        # turns (closing time), so a stale within-hours value must never leak
        # from the checkpointer.
        "within_business_hours": req.within_business_hours,
        "user_context": req.user_context,
        "contact_id": req.contact_id,
        "contact_tags": req.contact_tags,
        "language": req.language,
        "knowledge": req.knowledge,
        "faqs": [
            {
                "id": str(f.get("id") or ""),
                "question": f.get("question", ""),
                "answer": f.get("answer", ""),
                "category": f.get("category", ""),
                "priority": f.get("priority", 0),
                # Retrieval score from the backend FTS/trigram search. The
                # query_normalizer trigger reads it to decide whether the
                # initial retrieval was empty/marginal.
                "score": float(f.get("score") or 0.0),
            }
            for f in (req.rawFaqs or [])
            if isinstance(f, dict)
        ],
        "attachments": [a for a in (req.attachments or []) if isinstance(a, dict)],
        # Per-turn fields: reset on every request so a turn that skips the
        # writing node never re-reports the previous turn's values on the done
        # event (the checkpointer persists state across turns). A stale
        # mentioned_product_ids would re-attach a product image to an
        # unrelated reply; a stale faq_used would double-log FAQ usage.
        "faq_used": None,
        "mentioned_product_ids": [],
    }

    logger.info(
        "chat_stream_start",
        thread_id=thread_id,
        user_sub=user_sub,
        tenant_id=req.tenant_id,
        agent_code_name=code_name,
        agent_version=req.agent_version,
        agent_type=agent_type,
        provider=llm_provider,
        model=llm_model,
        catalog_count=len(req.product_catalog),
        catalog_access_enabled=req.catalog_access_enabled,
        faq_count=len(req.rawFaqs or []),
        # Compact prompt provenance for tenant-customised slots, keyed on the
        # row id (NOT is_default, which older backends never populate).
        custom_prompts={
            k: f"{v.id[:8]}@v{v.version}"
            for k, v in (req.prompts or {}).items()
            if v.id
        },
    )

    return StreamingResponse(
        _coalesced_stream(
            req.message,
            inputs["attachments"],
            inputs["faqs"],
            inputs,
            config,
            graph,
            code_name,
            thread_id,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/resume")
async def chat_resume(
    req: ChatResumeRequest,
    user_sub: Annotated[str, Depends(get_current_user)],
) -> StreamingResponse:
    agent_type = (req.agent_type or "").lower()
    code_name = (req.agent_code_name or "").strip().lower()
    if not code_name:
        code_name = resolve_legacy_agent_type(agent_type) or ""
        if code_name:
            legacy_agent_type_fallback_total.inc()
    if not code_name:
        raise HTTPException(
            status_code=400,
            detail=(
                f"agent_code_name is required; agent_type={agent_type!r} is "
                f"not mappable. Known code names: {sorted(CODE_NAME_REGISTRY)}"
            ),
        )

    try:
        graph = await get_or_compile_graph(code_name)
    except UnknownCodeNameError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    thread_id = scoped_thread_id(
        req.tenant_id,
        user_sub,
        req.conversation_id or req.thread_id,
        code_name,
        req.agent_version,
    )

    # Verify the interrupt belongs to this thread
    try:
        pool = await get_pool()
        row = await pool.fetchrow(
            """
            SELECT id, status FROM approval_requests
            WHERE id = $1 AND thread_id = $2
            """,
            uuid.UUID(req.interrupt_id),
            thread_id,
        )
    except Exception as db_exc:
        logger.error("resume_db_lookup_failed", error=str(db_exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error during resume lookup",
        )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interrupt not found or does not belong to this thread",
        )

    if row["status"] != "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Interrupt already resolved: {row['status']}",
        )

    # No caller-identity check here on purpose. The interrupt lookup above
    # already pins the row to this exact thread_id, and every authenticated
    # caller is the same service principal — there is no second user to defend
    # against. (The check that used to sit here compared thread_id against a
    # `{user_sub}:` prefix, which `scoped_thread_id` never produces: tenant_id
    # is the first segment, so it could not match.)

    # Mark as resolved
    resolved_status = "approved" if req.approved else "rejected"
    await pool.execute(
        """
        UPDATE approval_requests
        SET status = $1, resolved_at = NOW()
        WHERE id = $2
        """,
        resolved_status,
        uuid.UUID(req.interrupt_id),
    )

    config: dict = {
        "configurable": {
            "thread_id": thread_id,
            "openai_api_key": req.openai_api_key,
        }
    }

    logger.info(
        "chat_resume_start",
        thread_id=thread_id,
        interrupt_id=req.interrupt_id,
        approved=req.approved,
    )

    return StreamingResponse(
        _stream_graph(
            Command(resume=req.approved),
            config,
            graph=graph,
            agent_code_name=code_name,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
