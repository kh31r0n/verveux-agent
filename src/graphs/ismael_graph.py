"""Ismael — Christian theology Q&A over Company Brain (Moodle channel).

Topology:

    START → ismael_triage
              ├── ismael_survey          (a statistics question is waiting for its answer)
              ├── ismael_pending         (a Brain job is still answering this conversation)
              ├── ismael_start           (a new theology question)
              ├── ismael_moodle_support  (how to use Moodle / a technical problem / errands)
              ├── ismael_faq             (an institution FAQ answers it — before Brain or support)
              ├── ismael_teacher         (contact a teacher — a multi-turn flow, see teacher.py)
              │     ├── ismael_start / ismael_moodle_support / ismael_faq  (accepted help)
              │     └── ismael_triage    (the reply was another question: flow dropped)
              ├── greeting_response      (just a greeting — shared node, THEOLOGY_GREETING)
              └── ismael_off_topic       (anything else)

Every branch ends the turn (the teacher flow may first hand over to another node). The answer itself is not produced by the graph: a
background job (src/agents/ismael/rag_job.py) waits for Brain — booting its
host from stopped if needed — and delivers the answer out of band, because a
cold Brain plus a real question outlasts the backend's 60 s turn timeout.

No name capture: the MOODLE channel's identity is server-attested, so the name
always arrives in user_context.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..agents.greeting_response import greeting_response_node
from ..agents.ismael.nodes import (
    ismael_off_topic_node,
    ismael_pending_node,
    ismael_start_node,
    ismael_survey_node,
    ismael_triage_node,
)
from ..agents.ismael.faq import ismael_faq_node
from ..agents.ismael.support import ismael_moodle_support_node
from ..agents.ismael.teacher import ismael_teacher_node
from .state import AgentState

_ROUTES = {
    "survey": "ismael_survey",
    "pending": "ismael_pending",
    "start": "ismael_start",
    "moodle_support": "ismael_moodle_support",
    "faq": "ismael_faq",
    "teacher": "ismael_teacher",
    "greeting": "greeting_response",
    "off_topic": "ismael_off_topic",
}

# Where the teacher flow may hand the turn over; anything else ends it.
_TEACHER_HANDOFFS = {
    "start": "ismael_start",
    "moodle_support": "ismael_moodle_support",
    "faq": "ismael_faq",
    "retriage": "ismael_triage",
}


def _route_from_triage(
    state: AgentState,
) -> Literal[
    "ismael_survey",
    "ismael_pending",
    "ismael_start",
    "ismael_moodle_support",
    "ismael_faq",
    "ismael_teacher",
    "greeting_response",
    "ismael_off_topic",
]:
    return _ROUTES.get(str(state.get("ismael_route") or ""), "ismael_start")  # type: ignore[return-value]


def _route_from_teacher(state: AgentState) -> str:
    return _TEACHER_HANDOFFS.get(str(state.get("ismael_route") or ""), END)


def build_ismael_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("ismael_triage", ismael_triage_node)
    graph.add_node("ismael_survey", ismael_survey_node)
    graph.add_node("ismael_pending", ismael_pending_node)
    graph.add_node("ismael_start", ismael_start_node)
    graph.add_node("ismael_moodle_support", ismael_moodle_support_node)
    graph.add_node("ismael_faq", ismael_faq_node)
    graph.add_node("ismael_teacher", ismael_teacher_node)
    graph.add_node("greeting_response", greeting_response_node)
    graph.add_node("ismael_off_topic", ismael_off_topic_node)

    graph.add_edge(START, "ismael_triage")
    graph.add_conditional_edges(
        "ismael_triage", _route_from_triage, {node: node for node in _ROUTES.values()}
    )
    graph.add_conditional_edges(
        "ismael_teacher",
        _route_from_teacher,
        {**{node: node for node in _TEACHER_HANDOFFS.values()}, END: END},
    )
    for node in _ROUTES.values():
        if node != "ismael_teacher":
            graph.add_edge(node, END)

    return graph.compile(checkpointer=checkpointer)
