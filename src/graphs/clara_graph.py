"""Email agent graph (code name ``clara``).

    START ─(mode)─┬─ inbound ──► precheck ─┬─ short-circuit ──► END
                  │                        └─► triage ─┬─ noise / fyi ──► END
                  │                                    ├─ action_required ──► extract_task ─┐
                  │                                    └─ customer_conversation ────────────┴─► draft_reply ─► END
                  └─ follow_up ─► draft_follow_up ─► END

Tool-free: nodes only classify, extract or draft. The graph is compiled with
``name=EMAIL_GRAPH_NAME`` so the /email/* endpoints can refuse a code name that
points at a non-email graph without reaching into the registry.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, TypedDict

from langgraph.graph import END, START, StateGraph

from ..agents.clara.nodes import (
    draft_follow_up_node,
    draft_reply_node,
    extract_task_node,
    precheck_node,
    route_after_precheck,
    route_after_triage,
    route_mode,
    triage_node,
)
from ..usage import InvocationUsage

EMAIL_GRAPH_NAME = "email_agent"


class ClaraState(TypedDict, total=False):
    """JSON-native only: every value is a plain dict/list/str from model_dump(mode="json")."""

    mode: str  # "inbound" | "follow_up"
    agent_code_name: str
    tenant_id: str
    mailbox_id: str
    email: dict
    sender_context: dict
    sender_prefs: dict
    follow_up_context: dict
    precheck: dict
    triage: dict
    task: dict
    reply_draft: dict
    turn_usage: Annotated[List[InvocationUsage], operator.add]


def build_clara_graph(checkpointer):
    graph = StateGraph(ClaraState)
    graph.add_node("precheck", precheck_node)
    graph.add_node("triage", triage_node)
    graph.add_node("extract_task", extract_task_node)
    graph.add_node("draft_reply", draft_reply_node)
    graph.add_node("draft_follow_up", draft_follow_up_node)

    graph.add_conditional_edges(
        START, route_mode, {"precheck": "precheck", "draft_follow_up": "draft_follow_up"}
    )
    graph.add_conditional_edges(
        "precheck", route_after_precheck, {"end": END, "triage": "triage"}
    )
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {"end": END, "extract_task": "extract_task", "draft_reply": "draft_reply"},
    )
    graph.add_edge("extract_task", "draft_reply")
    graph.add_edge("draft_reply", END)
    graph.add_edge("draft_follow_up", END)
    return graph.compile(checkpointer=checkpointer, name=EMAIL_GRAPH_NAME)
