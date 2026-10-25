"""Pablo — conversational assessment inside a Moodle activity (mod_vervuxassess).

Topology:

    START → pablo_triage                (on the backend's turn kind, never on student text)
              ├── pablo_next_cycle      (start / continue: compose and present a question)
              ├── pablo_receive_answer  (answer: transcribe, record, acknowledge)
              ├── pablo_timeout         (the backend closed a cycle or the attempt)
              └── pablo_error           (a turn this attempt does not expect)

Every branch ends the turn. The timer is paused for all of them; the backend
restarts it on the ``cycle_presented`` event. The UX survey is a form owned by
the backend and the plugin, and scoring + feedback run afterwards as a
background job (``POST /assessment/score``), so no turn ever scores.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..agents.pablo.nodes import (
    pablo_error_node,
    pablo_next_cycle_node,
    pablo_receive_answer_node,
    pablo_timeout_node,
    pablo_triage_node,
)
from .state import AgentState

_ROUTES = {
    "next_cycle": "pablo_next_cycle",
    "answer": "pablo_receive_answer",
    "timeout": "pablo_timeout",
    "unexpected": "pablo_error",
}


def _route_from_triage(state: AgentState) -> str:
    return _ROUTES.get(str(state.get("pablo_route") or ""), "pablo_error")


def build_pablo_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("pablo_triage", pablo_triage_node)
    graph.add_node("pablo_next_cycle", pablo_next_cycle_node)
    graph.add_node("pablo_receive_answer", pablo_receive_answer_node)
    graph.add_node("pablo_timeout", pablo_timeout_node)
    graph.add_node("pablo_error", pablo_error_node)

    graph.add_edge(START, "pablo_triage")
    graph.add_conditional_edges(
        "pablo_triage", _route_from_triage, {node: node for node in set(_ROUTES.values())}
    )
    for node in set(_ROUTES.values()):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=checkpointer)
