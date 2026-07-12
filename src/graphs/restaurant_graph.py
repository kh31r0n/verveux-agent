"""Restaurant agent graph — menu inquiry, order flow, complaints, FAQ, escalation.

Order lifecycle (mirrors the sales pipeline, restaurant-flavoured)::

    order_collect → order_summary → END          (summary asks "confirmar")
                        │  next turn (restaurant_phase == "confirmation")
                        ▼
    triage ─▶ restaurant_confirm ──▶ execute → END   (checkout → backend Order)
                        └──▶ order_collect            (modify: same-turn re-collect)

`greeting_response` answers greetings from already-identified contacts with a
deterministic template (no LLM call); everything unmapped still falls back to
`faq_response`.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..agents.escalation import escalation_node
from ..agents.execute import execute_node
from ..agents.faq_response import faq_response_node
from ..agents.greeting_response import greeting_response_node, has_contact_name
from ..agents.menu_inquiry import menu_inquiry_node
from ..agents.name_capture import name_capture_node
from ..agents.query_normalizer import query_normalizer_node
from ..agents.restaurant_confirm import restaurant_confirm_node
from ..agents.restaurant_order_collect import restaurant_order_collect_node
from ..agents.restaurant_order_summary import restaurant_order_summary_node
from ..agents.triage import triage_node
from .shared_routing import should_capture_name
from .state import AgentState

# Re-use the complaint_collect node from the sales flow
from ..agents.complaint_collect import complaint_collect_node


def _order_in_progress(state: AgentState) -> bool:
    """True when an order has traction — a mid-order turn must never be
    hijacked by the name ask."""
    return bool(state.get("cart") or state.get("restaurant_order_data"))


def _route_from_triage(
    state: AgentState,
) -> Literal[
    "menu_inquiry",
    "order_collect",
    "restaurant_confirm",
    "execute",
    "complaint_collect",
    "greeting_response",
    "name_capture",
    "faq_response",
    "escalation",
    "__end__",
]:
    intent = state.get("intent", "")
    if intent == "complaint":
        return "complaint_collect"
    if intent == "escalation":
        return "escalation"
    # New users introduce themselves before a fresh flow starts; an
    # in-progress order and urgent intents (handled above) skip the ask.
    if not _order_in_progress(state) and should_capture_name(state, intent):
        return "name_capture"
    if intent in ("order", "sales"):
        # Post-checkout: a new "order" turn starts a fresh order (the collect
        # node resets cart + flags when execute_confirmed is set).
        if state.get("execute_confirmed"):
            return "order_collect"
        # Crash/retry resume: confirmed but checkout never ran.
        if state.get("restaurant_order_confirmed"):
            return "execute"
        # The customer is replying to the order summary.
        if state.get("restaurant_phase") == "confirmation":
            return "restaurant_confirm"
        return "order_collect"
    if intent == "menu_inquiry":
        return "menu_inquiry"
    if intent == "greeting" and has_contact_name(state):
        return "greeting_response"
    # Default: faq
    return "faq_response"


def _route_from_name_capture(state: AgentState):
    """End when the node already replied; otherwise answer the request that
    came with the refusal/deferral in this same turn. Cannot loop back: the
    node always latches a name or a deferral before this router runs."""
    if state.get("name_capture_reply_sent"):
        return END
    route = _route_from_triage(state)
    return "faq_response" if route == "name_capture" else route


def _route_from_order_collect(
    state: AgentState,
) -> Literal["order_summary", "__end__"]:
    if state.get("restaurant_order_complete"):
        return "order_summary"
    return END


def _route_from_confirm(
    state: AgentState,
) -> Literal["execute", "order_collect", "__end__"]:
    if state.get("restaurant_order_confirmed"):
        return "execute"
    # modify: re-enter collection in the same turn so the edit request is
    # answered (the confirm node already flipped the phase back to "collect").
    if state.get("restaurant_phase") == "collect":
        return "order_collect"
    # unclear: the node already re-asked.
    return END


def build_restaurant_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("menu_inquiry", menu_inquiry_node)
    graph.add_node("order_collect", restaurant_order_collect_node)
    graph.add_node("order_summary", restaurant_order_summary_node)
    graph.add_node("restaurant_confirm", restaurant_confirm_node)
    graph.add_node("execute", execute_node)
    graph.add_node("complaint_collect", complaint_collect_node)
    graph.add_node("greeting_response", greeting_response_node)
    graph.add_node("name_capture", name_capture_node)
    graph.add_node("faq_response", faq_response_node)
    graph.add_node("escalation", escalation_node)

    graph.add_node("query_normalizer", query_normalizer_node)
    graph.add_edge(START, "query_normalizer")
    graph.add_edge("query_normalizer", "triage")
    graph.add_conditional_edges(
        "triage",
        _route_from_triage,
        {
            "menu_inquiry": "menu_inquiry",
            "order_collect": "order_collect",
            "restaurant_confirm": "restaurant_confirm",
            "execute": "execute",
            "complaint_collect": "complaint_collect",
            "greeting_response": "greeting_response",
            "name_capture": "name_capture",
            "faq_response": "faq_response",
            "escalation": "escalation",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "name_capture",
        _route_from_name_capture,
        {
            "menu_inquiry": "menu_inquiry",
            "order_collect": "order_collect",
            "restaurant_confirm": "restaurant_confirm",
            "execute": "execute",
            "complaint_collect": "complaint_collect",
            "greeting_response": "greeting_response",
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
    graph.add_conditional_edges(
        "restaurant_confirm",
        _route_from_confirm,
        {"execute": "execute", "order_collect": "order_collect", END: END},
    )
    graph.add_edge("menu_inquiry", END)
    graph.add_edge("order_summary", END)
    graph.add_edge("execute", END)
    graph.add_edge("complaint_collect", END)
    graph.add_edge("greeting_response", END)
    graph.add_edge("faq_response", END)
    graph.add_edge("escalation", END)

    return graph.compile(checkpointer=checkpointer)
