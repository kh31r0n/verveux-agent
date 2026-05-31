"""School agent graph — admissions, course inquiry, schedule inquiry, FAQ, escalation."""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..agents.admissions_collect import admissions_collect_node
from ..agents.course_inquiry import course_inquiry_node
from ..agents.escalation import escalation_node
from ..agents.faq_response import faq_response_node
from ..agents.schedule_inquiry import schedule_inquiry_node
from ..agents.triage import triage_node
from .state import AgentState


def _route_from_triage(
    state: AgentState,
) -> Literal[
    "admissions_collect",
    "course_inquiry",
    "schedule_inquiry",
    "faq_response",
    "escalation",
    "__end__",
]:
    intent = state.get("intent", "")
    if intent == "admissions":
        return "admissions_collect"
    if intent == "course_inquiry":
        return "course_inquiry"
    if intent == "schedule_inquiry":
        return "schedule_inquiry"
    if intent == "escalation":
        return "escalation"
    # Default: faq
    return "faq_response"


def _route_from_admissions(state: AgentState) -> Literal["__end__"]:
    return END


def build_school_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("admissions_collect", admissions_collect_node)
    graph.add_node("course_inquiry", course_inquiry_node)
    graph.add_node("schedule_inquiry", schedule_inquiry_node)
    graph.add_node("faq_response", faq_response_node)
    graph.add_node("escalation", escalation_node)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        _route_from_triage,
        {
            "admissions_collect": "admissions_collect",
            "course_inquiry": "course_inquiry",
            "schedule_inquiry": "schedule_inquiry",
            "faq_response": "faq_response",
            "escalation": "escalation",
            END: END,
        },
    )
    graph.add_edge("admissions_collect", END)
    graph.add_edge("course_inquiry", END)
    graph.add_edge("schedule_inquiry", END)
    graph.add_edge("faq_response", END)
    graph.add_edge("escalation", END)

    return graph.compile(checkpointer=checkpointer)
