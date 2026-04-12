from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..agents.complaint_collect import complaint_collect_node
from ..agents.customer_data_collect import customer_data_collect_node
from ..agents.execute import execute_node
from ..agents.faq_response import faq_response_node
from ..agents.order_summary import order_summary_node
from ..agents.sales_collect import sales_collect_node
from ..agents.sales_confirm import sales_confirm_node
from ..agents.tracking_collect import tracking_collect_node
from ..agents.triage import route_from_triage, triage_node
from .state import AgentState

# ── Sales routing helpers ─────────────────────────────────────────────────────


def _route_from_sales_collect(
    state: AgentState,
) -> Literal["sales_confirm", "__end__"]:
    """After PRODUCT_SELECTION: advance to confirmation when phase is complete."""
    if state.get("product_selection_complete"):
        return "sales_confirm"
    return END


def _route_from_sales_confirm(
    state: AgentState,
) -> Literal["customer_data_collect", "sales_collect", "__end__"]:
    """
    After PRODUCT_CONFIRMATION:
      - Confirmed → customer_data_collect (auto-chain in same turn)
      - Rejected  → sales_collect (let user edit the cart)
      - Unclear   → END (wait for explicit yes/no)
    """
    if state.get("cart_confirmed") is True:
        return "customer_data_collect"
    if state.get("cart_confirmed") is False:
        return "sales_collect"
    return END


def _route_from_customer_data_collect(
    state: AgentState,
) -> Literal["order_summary", "__end__"]:
    """After CUSTOMER_DATA: advance to final summary when all required fields are present."""
    if state.get("customer_data_complete"):
        return "order_summary"
    return END


def _route_from_order_summary(
    state: AgentState,
) -> Literal["execute", "__end__"]:
    """After final order summary: execute when the user gives final confirmation."""
    if state.get("order_confirmed"):
        return "execute"
    return END


# ── Non-sales routing helpers ─────────────────────────────────────────────────


def _route_from_tracking_collect(
    state: AgentState,
) -> Literal["execute", "__end__"]:
    if state.get("tracking_complete"):
        return "execute"
    return END


def _route_from_complaint_collect(
    state: AgentState,
) -> Literal["execute", "__end__"]:
    if state.get("complaint_complete"):
        return "execute"
    return END


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_graph(checkpointer):
    graph = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("triage", triage_node)

    # Sales pipeline (new phase-based)
    graph.add_node("sales_collect", sales_collect_node)
    graph.add_node("sales_confirm", sales_confirm_node)
    graph.add_node("customer_data_collect", customer_data_collect_node)
    graph.add_node("order_summary", order_summary_node)

    # Other intents
    graph.add_node("tracking_collect", tracking_collect_node)
    graph.add_node("complaint_collect", complaint_collect_node)
    graph.add_node("faq_response", faq_response_node)

    # Execution
    graph.add_node("execute", execute_node)

    # ── Entry edge ────────────────────────────────────────────────────────────
    graph.add_edge(START, "triage")

    # ── Triage routing ────────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "triage",
        route_from_triage,
        {
            "sales_collect": "sales_collect",
            "sales_confirm": "sales_confirm",
            "customer_data_collect": "customer_data_collect",
            "order_summary": "order_summary",
            "tracking_collect": "tracking_collect",
            "complaint_collect": "complaint_collect",
            "faq_response": "faq_response",
            "execute": "execute",
        },
    )

    # ── Sales pipeline edges (auto-chain within same turn) ────────────────────
    graph.add_conditional_edges(
        "sales_collect",
        _route_from_sales_collect,
        {"sales_confirm": "sales_confirm", END: END},
    )
    graph.add_conditional_edges(
        "sales_confirm",
        _route_from_sales_confirm,
        {
            "customer_data_collect": "customer_data_collect",
            "sales_collect": "sales_collect",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "customer_data_collect",
        _route_from_customer_data_collect,
        {"order_summary": "order_summary", END: END},
    )
    graph.add_conditional_edges(
        "order_summary",
        _route_from_order_summary,
        {"execute": "execute", END: END},
    )

    # ── Tracking / complaint pipeline edges ───────────────────────────────────
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

    # ── Terminal edges ────────────────────────────────────────────────────────
    graph.add_edge("execute", END)
    graph.add_edge("faq_response", END)

    return graph.compile(checkpointer=checkpointer)
