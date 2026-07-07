"""Camila — academic secretary agent graph.

Topology:

    START → camila_triage
              ├── name_capture       (only when no name on file yet)
              ├── handoff            (institutional intent OR any attachment)
              ├── greeting_response  (greeting + name known — deterministic, no LLM)
              └── faq_response       (default — RAG-constrained, reused from sofia)

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
from ..agents.greeting_response import greeting_response_node
from ..agents.handoff import handoff_node
from ..agents.name_capture import name_capture_node
from ..agents.query_normalizer import query_normalizer_node
from ..schemas.intent import IntentType
from .shared_routing import should_capture_name
from .state import AgentState

_HANDOFF_INTENT_VALUES = {
    IntentType.PAYMENT_PROOF.value,
    IntentType.CORRECTION_REQUEST.value,
    IntentType.ACADEMIC_LOOKUP.value,
    IntentType.IDENTITY_CONFLICT.value,
}


def _route_from_triage(
    state: AgentState,
) -> Literal["name_capture", "handoff", "greeting_response", "faq_response"]:
    # Attachments always escalate — receipts, certificates, ID photos.
    if state.get("attachments"):
        return "handoff"

    intent = str(state.get("intent") or "").lower()
    if intent in _HANDOFF_INTENT_VALUES:
        return "handoff"

    if should_capture_name(state, intent):
        return "name_capture"

    if intent == IntentType.GREETING.value:
        return "greeting_response"

    return "faq_response"


def _route_from_name_capture(
    state: AgentState,
) -> Literal["handoff", "greeting_response", "faq_response", "__end__"]:
    """End when the node already replied; otherwise the customer's message
    carries a real request (refusal-with-question / implicit deferral) —
    re-route it so it gets answered in this turn. Cannot loop: the node
    always latches a name or a deferral before this router runs."""
    if state.get("name_capture_reply_sent"):
        return END
    route = _route_from_triage(state)
    return "faq_response" if route == "name_capture" else route


def build_camila_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("camila_triage", camila_triage_node)
    graph.add_node("name_capture", name_capture_node)
    graph.add_node("handoff", handoff_node)
    graph.add_node("greeting_response", greeting_response_node)
    graph.add_node("faq_response", faq_response_node)

    graph.add_node("query_normalizer", query_normalizer_node)
    graph.add_edge(START, "query_normalizer")
    graph.add_edge("query_normalizer", "camila_triage")
    graph.add_conditional_edges(
        "camila_triage",
        _route_from_triage,
        {
            "name_capture": "name_capture",
            "handoff": "handoff",
            "greeting_response": "greeting_response",
            "faq_response": "faq_response",
        },
    )
    graph.add_conditional_edges(
        "name_capture",
        _route_from_name_capture,
        {
            "handoff": "handoff",
            "greeting_response": "greeting_response",
            "faq_response": "faq_response",
            END: END,
        },
    )
    graph.add_edge("handoff", END)
    graph.add_edge("greeting_response", END)
    graph.add_edge("faq_response", END)

    return graph.compile(checkpointer=checkpointer)
