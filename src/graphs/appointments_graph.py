"""Appointments agent graph — stub.

Minimal triage → faq_response pipeline. Domain-specific nodes (booking,
rescheduling, cancellation) will be added as the appointments agent is fleshed out.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..agents.faq_response import faq_response_node
from ..agents.triage import triage_node
from .state import AgentState


def _route_from_triage(state: AgentState) -> Literal["faq_response", "__end__"]:
    intent = state.get("intent", "")
    if intent == "faq":
        return "faq_response"
    # All other intents fall through to FAQ for now
    return "faq_response"


def build_appointments_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("faq_response", faq_response_node)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        _route_from_triage,
        {"faq_response": "faq_response", END: END},
    )
    graph.add_edge("faq_response", END)

    return graph.compile(checkpointer=checkpointer)
