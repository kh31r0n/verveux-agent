"""Appointments agent graph — availability, booking flow, FAQ, escalation."""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..agents.availability_check import availability_check_node
from ..agents.booking_collect import booking_collect_node
from ..agents.booking_confirm import booking_confirm_node
from ..agents.escalation import escalation_node
from ..agents.faq_response import faq_response_node
from ..agents.triage import triage_node
from .state import AgentState


def _route_from_triage(
    state: AgentState,
) -> Literal[
    "availability_check",
    "booking_collect",
    "faq_response",
    "escalation",
    "__end__",
]:
    intent = state.get("intent", "")
    if intent == "availability":
        return "availability_check"
    if intent == "booking":
        return "booking_collect"
    if intent == "escalation":
        return "escalation"
    # Default: faq
    return "faq_response"


def _route_from_booking_collect(
    state: AgentState,
) -> Literal["booking_confirm", "__end__"]:
    if state.get("booking_complete"):
        return "booking_confirm"
    return END


def build_appointments_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("availability_check", availability_check_node)
    graph.add_node("booking_collect", booking_collect_node)
    graph.add_node("booking_confirm", booking_confirm_node)
    graph.add_node("faq_response", faq_response_node)
    graph.add_node("escalation", escalation_node)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        _route_from_triage,
        {
            "availability_check": "availability_check",
            "booking_collect": "booking_collect",
            "faq_response": "faq_response",
            "escalation": "escalation",
            END: END,
        },
    )
    graph.add_edge("availability_check", END)
    graph.add_conditional_edges(
        "booking_collect",
        _route_from_booking_collect,
        {"booking_confirm": "booking_confirm", END: END},
    )
    graph.add_edge("booking_confirm", END)
    graph.add_edge("faq_response", END)
    graph.add_edge("escalation", END)

    return graph.compile(checkpointer=checkpointer)
