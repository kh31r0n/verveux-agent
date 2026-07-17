"""Appointments agent graph — restructured lifecycle (Phase 3).

Topology::

    START → triage → {
        appointment_collect → availability_lookup →
            reservation_propose → confirmation,
        appointment_cancel,
        appointment_reschedule → availability_lookup →
            reservation_propose → reschedule_execute,
        greeting_response  (greeting + name known — deterministic, no LLM),
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
from ..agents.greeting_response import greeting_response_node, has_contact_name
from ..agents.name_capture import name_capture_node
from ..agents.query_normalizer import query_normalizer_node
from ..agents.triage import triage_node
from .shared_routing import should_capture_name, should_restrict_to_faq
from .state import AgentState


def _route_from_triage(
    state: AgentState,
) -> Literal[
    "appointment_collect",
    "appointment_cancel",
    "appointment_reschedule",
    "greeting_response",
    "name_capture",
    "faq_response",
    "escalation",
    "__end__",
]:
    intent = state.get("intent", "")
    booking_intent = state.get("booking_intent")

    # Outside business hours: FAQ-only, including in-progress flows
    # (immediate cutoff — booking state survives in the checkpoint and the
    # flow resumes when hours reopen). appointment_cancel and escalation are
    # URGENT_INTENTS, so they fall through to their branches below.
    if should_restrict_to_faq(state, intent):
        return "faq_response"

    # In-progress flow takes precedence over a fresh intent so a follow-up
    # message ("la 2 por favor", "sí confirmo") doesn't get re-triaged.
    if booking_intent == "book":
        return "appointment_collect"
    if booking_intent == "cancel":
        return "appointment_cancel"
    if booking_intent == "reschedule":
        return "appointment_reschedule"

    if intent == "appointment_cancel":
        return "appointment_cancel"
    if intent == "escalation":
        return "escalation"
    # New users introduce themselves before a fresh flow starts; urgent
    # intents (cancel, escalation — above) and in-progress flows skip it.
    if should_capture_name(state, intent):
        return "name_capture"

    if intent == "booking":
        return "appointment_collect"
    if intent == "appointment_reschedule":
        return "appointment_reschedule"
    if intent == "greeting" and has_contact_name(state):
        return "greeting_response"
    return "faq_response"


def _route_from_name_capture(state: AgentState):
    """End when the node already replied; otherwise answer the request that
    came with the refusal/deferral in this same turn. Cannot loop back: the
    node always latches a name or a deferral before this router runs."""
    if state.get("name_capture_reply_sent"):
        return END
    route = _route_from_triage(state)
    return "faq_response" if route == "name_capture" else route


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
            "appointment_collect": "appointment_collect",
            "appointment_cancel": "appointment_cancel",
            "appointment_reschedule": "appointment_reschedule",
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
            "appointment_collect": "appointment_collect",
            "appointment_cancel": "appointment_cancel",
            "appointment_reschedule": "appointment_reschedule",
            "greeting_response": "greeting_response",
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
    graph.add_edge("greeting_response", END)
    graph.add_edge("faq_response", END)
    graph.add_edge("escalation", END)

    return graph.compile(checkpointer=checkpointer)
