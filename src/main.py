import json
import logging
import traceback
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # used in lifespan
from langgraph.types import Command
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

import asyncio

from .auth.cognito import get_current_user, scoped_thread_id
from .agents.backend_client import (
    fetch_active_code_names,
    fetch_agent_credentials,
    fetch_in_use_code_names,
)
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

    async with AsyncPostgresSaver.from_conn_string(settings.database_url) as checkpointer:
        await checkpointer.setup()
        set_checkpointer(checkpointer)

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
    vertex_credentials: dict = {}
    vertex_project_id: str = ""
    vertex_location: str = ""
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


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


_SECRET_KEYS = {
    "openai_api_key",
    "anthropic_api_key",
    "vertex_credentials",
    "vertex_service_account_json",
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
        try:
            final_state = await graph.aget_state(config)
            turn_usage = final_state.values.get("turn_usage", []) or []
            mentioned_product_ids = (
                final_state.values.get("mentioned_product_ids", []) or []
            )
        except Exception as exc:
            logger.warning("done_event_state_fetch_failed", error=str(exc))

        yield _sse_event({
            "type": "done",
            "turn_request_id": config.get("configurable", {}).get("turn_request_id", ""),
            "turn_usage": turn_usage,
            "mentioned_product_ids": mentioned_product_ids,
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
            elif llm_provider == "vertex":
                provider_config["vertex_credentials"] = creds.get("vertexCredentials", {})
                provider_config["vertex_project_id"] = creds.get("vertexProjectId", "")
                provider_config["vertex_location"] = creds.get("vertexLocation", "")
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
                "vertex_credentials": req.vertex_credentials,
                "vertex_project_id": req.vertex_project_id,
                "vertex_location": req.vertex_location,
            }
    else:
        # No tenantId — use legacy per-request keys
        provider_config = {
            "openai_api_key": req.openai_api_key,
            "anthropic_api_key": req.anthropic_api_key,
            "vertex_credentials": req.vertex_credentials,
            "vertex_project_id": req.vertex_project_id,
            "vertex_location": req.vertex_location,
        }

    prompts_dict = {k: v.model_dump() for k, v in req.prompts.items()} if req.prompts else {}

    config: dict = {
        "configurable": {
            "thread_id": thread_id,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "prompts": prompts_dict,
            "turn_request_id": req.turn_request_id,
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
        "user_context": req.user_context,
        "contact_id": req.contact_id,
        "contact_tags": req.contact_tags,
        "language": req.language,
        "knowledge": req.knowledge,
        "faqs": [
            {
                "question": f.get("question", ""),
                "answer": f.get("answer", ""),
                "category": f.get("category", ""),
                "priority": f.get("priority", 0),
            }
            for f in (req.rawFaqs or [])
            if isinstance(f, dict)
        ],
        "attachments": [a for a in (req.attachments or []) if isinstance(a, dict)],
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
        faq_count=len(req.rawFaqs or []),
    )

    return StreamingResponse(
        _stream_graph(inputs, config, graph=graph, agent_code_name=code_name),
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

    # Verify ownership: scoped thread_id already contains user_sub prefix
    expected_prefix = f"{user_sub}:"
    if not thread_id.startswith(expected_prefix):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Thread does not belong to the authenticated user",
        )

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
