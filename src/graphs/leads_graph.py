"""Leads agent graph (veronica) — web-widget lead capture & qualification.

Topology::

    START → query_normalizer → triage → {
        greeting_response  (greeting + name known),
        name_capture       (unknown visitor, non-urgent intent),
        lead_collect → execute_lead  (auto-chain when fullName + email +
                                      serviceInterest complete),
        faq_response,
    } → END

The graph is deliberately small: veronica answers FAQs and qualifies leads —
no carts, orders, or bookings. Once a lead was submitted, further messages
re-triage normally (usually to faq).
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..agents.execute_lead import execute_lead_node
from ..agents.faq_response import faq_response_node
from ..agents.greeting_response import greeting_response_node, has_contact_name
from ..agents.lead_collect import lead_collect_node
from ..agents.name_capture import name_capture_node
from ..agents.query_normalizer import query_normalizer_node
from ..agents.triage import triage_node
from .shared_routing import should_capture_name, should_restrict_to_faq
from .state import AgentState

# ── Routing helpers ───────────────────────────────────────────────────────────


def _in_lead_flow(state: AgentState) -> bool:
    """True when qualification already has traction — a mid-flow turn must
    never be hijacked by the name ask (the collect node handles names itself)."""
    return bool(state.get("lead_data")) and not state.get("lead_submitted")


def _route_from_triage(
    state: AgentState,
) -> Literal[
    "greeting_response",
    "name_capture",
    "lead_collect",
    "faq_response",
    "execute_lead",
]:
    # Outside business hours: FAQ-only, including a mid-qualification flow
    # (immediate cutoff — lead_data survives in the checkpoint and the flow
    # resumes when hours reopen).
    if should_restrict_to_faq(state):
        return "faq_response"
    if state.get("intent") == "greeting" and has_contact_name(state):
        return "greeting_response"
    # New visitors introduce themselves before a fresh flow starts; the
    # captured name doubles as the lead's fullName seed.
    if not _in_lead_flow(state) and should_capture_name(state):
        return "name_capture"

    intent = state.get("intent") or "faq"
    if intent == "lead_capture":
        if state.get("lead_submitted"):
            # Already registered — treat follow-ups as general questions.
            return "faq_response"
        if state.get("lead_collection_complete") and state.get(
            "lead_submission_id"
        ):
            # A previous turn completed collection but the submission failed —
            # retry with the same submission id before anything else.
            return "execute_lead"
        return "lead_collect"
    return "faq_response"


def _route_from_name_capture(state: AgentState):
    """End when the node already replied; otherwise answer the request that
    came with the refusal/name in this same turn. Cannot loop back: the node
    always latches a name or a deferral before this router runs."""
    if state.get("name_capture_reply_sent"):
        return END
    route = _route_from_triage(state)
    return "faq_response" if route == "name_capture" else route


def _route_from_lead_collect(
    state: AgentState,
) -> Literal["execute_lead", "__end__"]:
    """Auto-chain into submission the moment the mandatory set completes."""
    if state.get("lead_collection_complete") and not state.get("lead_submitted"):
        return "execute_lead"
    return END


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_leads_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("lead_collect", lead_collect_node)
    graph.add_node("faq_response", faq_response_node)
    graph.add_node("greeting_response", greeting_response_node)
    graph.add_node("name_capture", name_capture_node)
    graph.add_node("execute_lead", execute_lead_node)

    graph.add_node("query_normalizer", query_normalizer_node)
    graph.add_edge(START, "query_normalizer")
    graph.add_edge("query_normalizer", "triage")

    graph.add_conditional_edges(
        "triage",
        _route_from_triage,
        {
            "greeting_response": "greeting_response",
            "name_capture": "name_capture",
            "lead_collect": "lead_collect",
            "faq_response": "faq_response",
            "execute_lead": "execute_lead",
        },
    )

    graph.add_conditional_edges(
        "name_capture",
        _route_from_name_capture,
        {
            "greeting_response": "greeting_response",
            "lead_collect": "lead_collect",
            "faq_response": "faq_response",
            "execute_lead": "execute_lead",
            END: END,
        },
    )

    graph.add_conditional_edges(
        "lead_collect",
        _route_from_lead_collect,
        {"execute_lead": "execute_lead", END: END},
    )

    graph.add_edge("execute_lead", END)
    graph.add_edge("faq_response", END)
    graph.add_edge("greeting_response", END)

    return graph.compile(checkpointer=checkpointer)
