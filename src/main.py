import json
import logging
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

from .auth.cognito import get_current_user, scoped_thread_id
from .config import settings
from .db.postgres import close_pool, get_pool, init_pool, run_migrations
from .graphs.main_graph import build_graph
from .observability import (
    agent_interrupt_events_total,
    agent_requests_total,
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
# Global compiled graph (initialised in lifespan)
# ---------------------------------------------------------------------------

compiled_graph = None

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global compiled_graph

    # Initialise application DB pool and run migrations
    pool = await init_pool()
    await run_migrations(pool)

    # Initialise LangGraph checkpointer and compile graph
    async with AsyncPostgresSaver.from_conn_string(settings.database_url) as checkpointer:
        await checkpointer.setup()
        compiled_graph = build_graph(checkpointer)
        logger.info("langgraph_ready")
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


class ChatStreamRequest(BaseModel):
    thread_id: str
    message: str
    openai_api_key: str
    project_id: str = ""
    conversation_id: str = ""
    product_catalog: list = []  # [{product_id, name, description, price, stock}]
    user_context: dict = {}     # {name, email, phone, address}
    contact_id: str = ""        # NestJS contact UUID
    contact_tags: list = []     # [{"id", "name", "color"}] current tags on the contact


class ChatResumeRequest(BaseModel):
    thread_id: str
    interrupt_id: str
    approved: bool
    openai_api_key: str


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_graph(
    inputs: dict | Command,
    config: dict,
) -> AsyncGenerator[str, None]:
    """Consume a graph.astream() and yield SSE-formatted strings."""
    if compiled_graph is None:
        yield _sse_event({"type": "error", "message": "Agent graph not initialised"})
        return

    try:
        async for chunk in compiled_graph.astream(
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
                            record_tool_error("audit_db")

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
                                elif k != "openai_api_key":
                                    try:
                                        json.dumps(v)  # verify serialisable
                                        safe_data[k] = v
                                    except (TypeError, ValueError):
                                        pass

                        yield _sse_event(
                            {"type": "node_update", "node": node_name, "data": safe_data}
                        )

        yield _sse_event({"type": "done"})

    except Exception as exc:
        logger.error("stream_error", error=str(exc))
        record_tool_error("graph_stream")
        yield _sse_event({"type": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
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
    if compiled_graph is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent graph not initialised")

    thread_id = scoped_thread_id(user_sub, req.thread_id)
    agent_requests_total.inc()

    config: dict = {
        "configurable": {
            "thread_id": thread_id,
            "openai_api_key": req.openai_api_key,
        }
    }

    inputs: dict = {
        "messages": [HumanMessage(content=req.message)],
        "thread_id": thread_id,
        "project_id": req.project_id,
        "conversation_id": req.conversation_id,
        "product_catalog": req.product_catalog,
        "user_context": req.user_context,
        "contact_id": req.contact_id,
        "contact_tags": req.contact_tags,
    }

    logger.info(
        "chat_stream_start",
        thread_id=thread_id,
        user_sub=user_sub,
        project_id=req.project_id,
        catalog_count=len(req.product_catalog),
    )

    return StreamingResponse(
        _stream_graph(inputs, config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat/resume")
async def chat_resume(
    req: ChatResumeRequest,
    user_sub: Annotated[str, Depends(get_current_user)],
) -> StreamingResponse:
    if compiled_graph is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent graph not initialised")

    thread_id = scoped_thread_id(user_sub, req.thread_id)

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
        _stream_graph(Command(resume=req.approved), config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
