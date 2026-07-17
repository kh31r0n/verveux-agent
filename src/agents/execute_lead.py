"""Lead submission node (veronica) — POSTs the qualified lead to NestJS.

Idempotency contract (two layers):
  1. ``lead_submitted`` state latch — a LangGraph replay of this node (or a
     stale checkpoint resume) short-circuits without any HTTP call.
  2. ``lead_submission_id`` rides as the backend's InquirySource
     idempotencyKey (unique index) — a retry that raced a successful but
     unacknowledged POST returns the original Inquiry instead of a duplicate.

The latch is set ONLY after a confirmed 2xx: a failed POST leaves it False so
the next turn retries with the SAME submission id.
"""

import structlog
from httpx import HTTPStatusError, RequestError
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..observability import record_node_invocation
from . import backend_client

logger = structlog.get_logger(__name__)

_FAILURE_TEMPLATES = {
    "es": "Disculpa, tuve un problema registrando tus datos. ¿Podrías intentarlo de nuevo en unos minutos?",
    "en": "Sorry, I had a problem saving your details. Could you try again in a few minutes?",
    "pt": "Desculpe, tive um problema ao registrar seus dados. Poderia tentar novamente em alguns minutos?",
}


async def execute_lead_node(
    state: AgentState,
    config: RunnableConfig,  # noqa: ARG001 — uniform node signature
) -> dict:
    record_node_invocation("execute_lead")
    thread_id: str = state.get("thread_id", "unknown")

    if state.get("lead_submitted"):
        logger.info("execute_lead_skipped_already_submitted", thread_id=thread_id)
        return {}

    submission_id = state.get("lead_submission_id")
    lead_data: dict = state.get("lead_data") or {}
    tenant_id = state.get("tenant_id", "")
    conversation_id = state.get("conversation_id", "")

    if not submission_id or not tenant_id or not conversation_id:
        # Defensive: the router only sends complete leads here; missing
        # context means a malformed request, not a user-visible failure.
        logger.error(
            "execute_lead_missing_context",
            thread_id=thread_id,
            has_submission_id=bool(submission_id),
            has_tenant=bool(tenant_id),
            has_conversation=bool(conversation_id),
        )
        return {}

    try:
        resp = await backend_client.submit_inquiry(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            idempotency_key=submission_id,
            lead_data=lead_data,
        )
    except (HTTPStatusError, RequestError) as exc:
        logger.error(
            "execute_lead_backend_failed",
            thread_id=thread_id,
            submission_id=submission_id,
            error=str(exc),
        )
        # lead_collect already replied this turn; append the apology so the
        # visitor knows the registration didn't go through. No latch — the
        # next turn retries with the same submission id.
        lang = state.get("language", "es")
        failure = _FAILURE_TEMPLATES.get(lang, _FAILURE_TEMPLATES["es"])
        write = get_stream_writer()
        write({"type": "token", "content": failure})
        return {"messages": [AIMessage(content=failure)]}

    logger.info(
        "execute_lead_submitted",
        thread_id=thread_id,
        submission_id=submission_id,
        inquiry_id=resp.get("inquiryId"),
        deduped=resp.get("deduped", False),
    )
    # No user-facing message on success — lead_collect's reply already
    # acknowledged ("un asesor te contactará").
    return {"lead_submitted": True}
