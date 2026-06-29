"""Appointments agent graph — restructured lifecycle (Phase 3).

Topology::

    START → triage → {
        appointment_collect → availability_lookup →
            reservation_propose → confirmation,
        appointment_cancel,
        appointment_reschedule → availability_lookup →
            reservation_propose → reschedule_execute,
        faq_response,
        escalation,
    } → END

The same ``availability_lookup → reservation_propose`` pair serves both new
bookings and reschedules; the downstream node (``confirmation`` vs
``reschedule_execute``) is selected by the ``reschedule_source_id`` flag on
``AgentState``.

Slot-conflict recovery: if the backend rejects a hold with ``code:SLOT_TAKEN``
the lifecycle nodes set ``slot_conflict=True``. Conditional edges route back
to ``availability_lookup`` so the user can pick a fresh slot without
restarting the conversation.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..agents.appointments.appointment_cancel import appointment_cancel_node
from ..agents.appointments.appointment_collect import appointment_collect_node
from ..agents.appointments.appointment_reschedule import (
    appointment_reschedule_node,
    reschedule_execute_node,
)
from ..agents.appointments.availability_lookup import availability_lookup_node
from ..agents.appointments.confirmation import confirmation_node
from ..agents.appointments.reservation_propose import reservation_propose_node
from ..agents.escalation import escalation_node
from ..agents.faq_response import faq_response_node
from ..agents.triage import triage_node
from .state import AgentState


def _route_from_triage(
    state: AgentState,
) -> Literal[
    "appointment_collect",
    "appointment_cancel",
    "appointment_reschedule",
    "faq_response",
    "escalation",
    "__end__",
]:
    intent = state.get("intent", "")
    booking_intent = state.get("booking_intent")

    # In-progress flow takes precedence over a fresh intent so a follow-up
    # message ("la 2 por favor", "sí confirmo") doesn't get re-triaged.
    if booking_intent == "book":
        return "appointment_collect"
    if booking_intent == "cancel":
        return "appointment_cancel"
    if booking_intent == "reschedule":
        return "appointment_reschedule"

    if intent == "booking":
        return "appointment_collect"
    if intent == "appointment_cancel":
        return "appointment_cancel"
    if intent == "appointment_reschedule":
        return "appointment_reschedule"
    if intent == "escalation":
        return "escalation"
    return "faq_response"


def _route_from_appointment_collect(
    state: AgentState,
) -> Literal["availability_lookup", "__end__"]:
    if state.get("booking_complete"):
        return "availability_lookup"
    return END


def _route_from_availability_lookup(
    state: AgentState,
) -> Literal["reservation_propose", "__end__"]:
    if state.get("candidate_slots"):
        return "reservation_propose"
    return END


def _route_from_reservation_propose(
    state: AgentState,
) -> Literal["confirmation", "reschedule_execute", "__end__"]:
    if not state.get("chosen_slot"):
        return END
    if state.get("reschedule_source_id"):
        return "reschedule_execute"
    return "confirmation"


def _route_from_confirmation(
    state: AgentState,
) -> Literal["availability_lookup", "__end__"]:
    if state.get("slot_conflict"):
        return "availability_lookup"
    return END


def _route_from_reschedule_execute(
    state: AgentState,
) -> Literal["availability_lookup", "__end__"]:
    if state.get("slot_conflict"):
        return "availability_lookup"
    return END


def _route_from_appointment_reschedule(
    state: AgentState,
) -> Literal["availability_lookup", "__end__"]:
    if state.get("reschedule_source_id"):
        return "availability_lookup"
    return END


def build_appointments_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("appointment_collect", appointment_collect_node)
    graph.add_node("availability_lookup", availability_lookup_node)
    graph.add_node("reservation_propose", reservation_propose_node)
    graph.add_node("confirmation", confirmation_node)
    graph.add_node("appointment_cancel", appointment_cancel_node)
    graph.add_node("appointment_reschedule", appointment_reschedule_node)
    graph.add_node("reschedule_execute", reschedule_execute_node)
    graph.add_node("faq_response", faq_response_node)
    graph.add_node("escalation", escalation_node)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        _route_from_triage,
        {
            "appointment_collect": "appointment_collect",
            "appointment_cancel": "appointment_cancel",
            "appointment_reschedule": "appointment_reschedule",
            "faq_response": "faq_response",
            "escalation": "escalation",
            END: END,
        },
    )

    graph.add_conditional_edges(
        "appointment_collect",
        _route_from_appointment_collect,
        {"availability_lookup": "availability_lookup", END: END},
    )
    graph.add_conditional_edges(
        "availability_lookup",
        _route_from_availability_lookup,
        {"reservation_propose": "reservation_propose", END: END},
    )
    graph.add_conditional_edges(
        "reservation_propose",
        _route_from_reservation_propose,
        {
            "confirmation": "confirmation",
            "reschedule_execute": "reschedule_execute",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "confirmation",
        _route_from_confirmation,
        {"availability_lookup": "availability_lookup", END: END},
    )
    graph.add_conditional_edges(
        "reschedule_execute",
        _route_from_reschedule_execute,
        {"availability_lookup": "availability_lookup", END: END},
    )

    graph.add_conditional_edges(
        "appointment_reschedule",
        _route_from_appointment_reschedule,
        {"availability_lookup": "availability_lookup", END: END},
    )

    graph.add_edge("appointment_cancel", END)
    graph.add_edge("faq_response", END)
    graph.add_edge("escalation", END)

    return graph.compile(checkpointer=checkpointer)
