"""Camila — academic secretary agent graph.

Topology:

    START → camila_triage
              ├── name_capture  (only when no name on file yet)
              ├── handoff       (institutional intent OR any attachment)
              └── faq_response  (default — RAG-constrained, reused from sofia)

`name_capture` is terminal for its turn (we wait for the user to actually
answer). `handoff` is permanently terminal — it disables AI on the
conversation via /internal/conversations/:id/handoff. `faq_response`
follows the same defensive RAG pattern as the existing school agent.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..agents.camila_triage import camila_triage_node
from ..agents.faq_response import faq_response_node
from ..agents.handoff import handoff_node
from ..agents.name_capture import name_capture_node
from ..schemas.intent import IntentType
from .state import AgentState

_HANDOFF_INTENT_VALUES = {
    IntentType.PAYMENT_PROOF.value,
    IntentType.CORRECTION_REQUEST.value,
    IntentType.ACADEMIC_LOOKUP.value,
    IntentType.IDENTITY_CONFLICT.value,
}


def _has_name(state: AgentState) -> bool:
    """True when we already know the contact's name (latched flag OR a
    non-empty `user_context.name` provided by the backend snapshot)."""
    if state.get("school_name_captured"):
        return True
    ctx = state.get("user_context") or {}
    name = (ctx.get("name") or "").strip() if isinstance(ctx, dict) else ""
    return bool(name)


def _route_from_triage(
    state: AgentState,
) -> Literal["name_capture", "handoff", "faq_response"]:
    # Attachments always escalate — receipts, certificates, ID photos.
    if state.get("attachments"):
        return "handoff"

    intent = str(state.get("intent") or "").lower()
    if intent in _HANDOFF_INTENT_VALUES:
        return "handoff"

    if not _has_name(state):
        return "name_capture"

    return "faq_response"


def build_camila_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("camila_triage", camila_triage_node)
    graph.add_node("name_capture", name_capture_node)
    graph.add_node("handoff", handoff_node)
    graph.add_node("faq_response", faq_response_node)

    graph.add_edge(START, "camila_triage")
    graph.add_conditional_edges(
        "camila_triage",
        _route_from_triage,
        {
            "name_capture": "name_capture",
            "handoff": "handoff",
            "faq_response": "faq_response",
        },
    )
    graph.add_edge("name_capture", END)
    graph.add_edge("handoff", END)
    graph.add_edge("faq_response", END)

    return graph.compile(checkpointer=checkpointer)
