from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..agents.complaint_collect import complaint_collect_node
from ..agents.execute import execute_node
from ..agents.faq_response import faq_response_node
from ..agents.order_history import order_history_node
from ..agents.order_summary import order_summary_node
from ..agents.sales_collect import sales_collect_node
from ..agents.tracking_collect import tracking_collect_node
from ..agents.triage import route_from_triage, triage_node
from .state import AgentState


def _route_from_sales_collect(state: AgentState) -> Literal["order_summary", "__end__"]:
    """After sales collection: if all done, proceed to order summary immediately."""
    if state.get("sales_complete"):
        return "order_summary"
    return END


def _route_from_order_summary(state: AgentState) -> Literal["execute", "__end__"]:
    """After order summary: if confirmed, proceed to execute."""
    if state.get("order_confirmed"):
        return "execute"
    return END


def _route_from_tracking_collect(state: AgentState) -> Literal["execute", "__end__"]:
    """After tracking collection: if complete, proceed to execute."""
    if state.get("tracking_complete"):
        return "execute"
    return END


def _route_from_complaint_collect(state: AgentState) -> Literal["execute", "__end__"]:
    """After complaint collection: if complete, proceed to execute."""
    if state.get("complaint_complete"):
        return "execute"
    return END


def build_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("sales_collect", sales_collect_node)
    graph.add_node("order_summary", order_summary_node)
    graph.add_node("tracking_collect", tracking_collect_node)
    graph.add_node("complaint_collect", complaint_collect_node)
    graph.add_node("order_history", order_history_node)
    graph.add_node("faq_response", faq_response_node)
    graph.add_node("execute", execute_node)

    graph.add_edge(START, "triage")

    graph.add_conditional_edges(
        "triage",
        route_from_triage,
        {
            "sales_collect": "sales_collect",
            "order_summary": "order_summary",
            "tracking_collect": "tracking_collect",
            "complaint_collect": "complaint_collect",
            "order_history": "order_history",
            "faq_response": "faq_response",
            "execute": "execute",
        },
    )

    # Auto-advance when a phase completes within the same turn
    graph.add_conditional_edges(
        "sales_collect",
        _route_from_sales_collect,
        {"order_summary": "order_summary", END: END},
    )
    graph.add_conditional_edges(
        "order_summary",
        _route_from_order_summary,
        {"execute": "execute", END: END},
    )
    graph.add_conditional_edges(
        "tracking_collect",
        _route_from_tracking_collect,
        {"execute": "execute", END: END},
    )
    graph.add_conditional_edges(
        "complaint_collect",
        _route_from_complaint_collect,
        {"execute": "execute", END: END},
    )

    graph.add_edge("execute", END)
    graph.add_edge("order_history", END)
    graph.add_edge("faq_response", END)

    return graph.compile(checkpointer=checkpointer)
