"""Camila handoff node — disables AI on the conversation and replies in Spanish.

Called when triage classifies an institutional intent (payment proof, document
correction, academic lookup, conflicting identity docs) OR when the inbound
message carries an attachment. The reply is a templated Spanish acknowledgement
— no LLM call — so the message is deterministic, free, and immune to
hallucination. Backend handoff (POST /internal/conversations/:id/handoff) is
fire-and-forget for resilience: even if the HTTP call fails we still emit the
goodbye message so the user is not left silent.
"""

from __future__ import annotations

import structlog
from httpx import HTTPStatusError, RequestError
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..graphs.state import AgentState
from ..observability import record_node_invocation
from . import backend_client

logger = structlog.get_logger(__name__)

_HANDOFF_REPLY = (
    "Gracias por escribirnos. En breve un asesor de la institución te "
    "contactará para continuar con tu solicitud."
)


async def handoff_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("handoff")

    reason = state.get("handoff_reason") or "agent_escalation"
    intent = state.get("intent", "")
    attachments = state.get("attachments") or []
    tenant_id = state.get("tenant_id", "")
    conversation_id = state.get("conversation_id", "")
    thread_id = state.get("thread_id", "unknown")

    intents: list[str] = []
    if intent:
        intents.append(str(intent))
    structured = state.get("structured_intent")
    secondary = getattr(structured, "secondary_intents", None) if structured else None
    if secondary is None and isinstance(structured, dict):
        secondary = structured.get("secondary_intents", [])
    for s in secondary or []:
        val = s.value if hasattr(s, "value") else str(s)
        if val and val not in intents:
            intents.append(val)

    if tenant_id and conversation_id:
        try:
            await backend_client.request_handoff(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                reason=reason,
                intents=intents,
                has_attachments=bool(attachments),
            )
        except (HTTPStatusError, RequestError) as exc:
            # Non-fatal: emit the user-facing message even if the backend
            # flip failed — the human inbox will still see the message,
            # the AI may answer the next turn but the operator can flip
            # aiEnabled manually.
            logger.error(
                "handoff_backend_call_failed",
                thread_id=thread_id,
                conversation_id=conversation_id,
                error=str(exc),
            )
    else:
        logger.warning(
            "handoff_missing_ids",
            thread_id=thread_id,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )

    logger.info(
        "handoff_emitted",
        thread_id=thread_id,
        conversation_id=conversation_id,
        reason=reason,
        intents=intents,
        has_attachments=bool(attachments),
    )

    return {
        "messages": [AIMessage(content=_HANDOFF_REPLY)],
        "handoff_reason": reason,
    }
