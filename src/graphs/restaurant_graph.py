"""Restaurant agent graph — menu inquiry, order flow, complaints, FAQ, escalation."""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..agents.escalation import escalation_node
from ..agents.faq_response import faq_response_node
from ..agents.menu_inquiry import menu_inquiry_node
from ..agents.restaurant_order_collect import restaurant_order_collect_node
from ..agents.restaurant_order_summary import restaurant_order_summary_node
from ..agents.triage import triage_node
from .state import AgentState

# Re-use the complaint_collect node from the sales flow
from ..agents.complaint_collect import complaint_collect_node


def _route_from_triage(
    state: AgentState,
) -> Literal[
    "menu_inquiry",
    "order_collect",
    "complaint_collect",
    "faq_response",
    "escalation",
    "__end__",
]:
    intent = state.get("intent", "")
    if intent == "menu_inquiry":
        return "menu_inquiry"
    if intent in ("order", "sales"):
        return "order_collect"
    if intent == "complaint":
        return "complaint_collect"
    if intent == "escalation":
        return "escalation"
    # Default: faq
    return "faq_response"


def _route_from_order_collect(
    state: AgentState,
) -> Literal["order_summary", "__end__"]:
    if state.get("restaurant_order_complete"):
        return "order_summary"
    return END


def build_restaurant_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("menu_inquiry", menu_inquiry_node)
    graph.add_node("order_collect", restaurant_order_collect_node)
    graph.add_node("order_summary", restaurant_order_summary_node)
    graph.add_node("complaint_collect", complaint_collect_node)
    graph.add_node("faq_response", faq_response_node)
    graph.add_node("escalation", escalation_node)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        _route_from_triage,
        {
            "menu_inquiry": "menu_inquiry",
            "order_collect": "order_collect",
            "complaint_collect": "complaint_collect",
            "faq_response": "faq_response",
            "escalation": "escalation",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "order_collect",
        _route_from_order_collect,
        {"order_summary": "order_summary", END: END},
    )
    graph.add_edge("menu_inquiry", END)
    graph.add_edge("order_summary", END)
    graph.add_edge("complaint_collect", END)
    graph.add_edge("faq_response", END)
    graph.add_edge("escalation", END)

    return graph.compile(checkpointer=checkpointer)
