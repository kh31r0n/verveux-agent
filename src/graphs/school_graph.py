"""School agent graph — admissions, course inquiry, schedule inquiry, FAQ, escalation.

`greeting_response` answers greetings from already-identified contacts with a
deterministic template (no LLM call); everything unmapped still falls back to
`faq_response`.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..agents.admissions_collect import admissions_collect_node
from ..agents.course_inquiry import course_inquiry_node
from ..agents.escalation import escalation_node
from ..agents.faq_response import faq_response_node
from ..agents.greeting_response import greeting_response_node, has_contact_name
from ..agents.name_capture import name_capture_node
from ..agents.query_normalizer import query_normalizer_node
from ..agents.schedule_inquiry import schedule_inquiry_node
from ..agents.triage import triage_node
from ..agents.utils import latest_user_messages
from .shared_routing import should_capture_name
from .state import AgentState

_QUESTION_STARTERS = (
    "qué ", "que ", "cuál", "cual", "cuándo", "cuando ", "cuánto", "cuanto",
    "dónde", "donde ", "cómo", "como ", "por qué", "por que", "quién", "quien ",
    "hay ", "tienen", "tienes", "existen", "puedo ", "se puede", "necesito saber",
    "what ", "which ", "when ", "how ", "where ", "why ", "who ", "is there",
    "are there", "do you", "can i",
)


def _looks_like_question(text: str) -> bool:
    """Heuristic: detect interrogative user messages (Spanish + English)."""
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("¿") or stripped.endswith("?"):
        return True
    lowered = stripped.lower()
    return any(lowered.startswith(starter) for starter in _QUESTION_STARTERS)


def _route_from_triage(
    state: AgentState,
) -> Literal[
    "admissions_collect",
    "course_inquiry",
    "schedule_inquiry",
    "greeting_response",
    "name_capture",
    "faq_response",
    "escalation",
    "__end__",
]:
    intent = state.get("intent", "")
    if intent == "escalation":
        return "escalation"
    # New users introduce themselves before a fresh flow starts; an
    # in-progress admissions flow and urgent intents skip the ask.
    if not state.get("admissions_data") and should_capture_name(state, intent):
        return "name_capture"
    if intent == "admissions":
        # Guard: if user is asking an informational question and no admissions
        # data has been collected yet, answer via FAQ before starting the
        # data-collection flow. The question may be any fragment of a rapid
        # multi-message burst, so check each trailing user message.
        if not state.get("admissions_data") and any(
            _looks_like_question(fragment) for fragment in latest_user_messages(state)
        ):
            return "faq_response"
        return "admissions_collect"
    if intent == "course_inquiry":
        return "course_inquiry"
    if intent == "schedule_inquiry":
        return "schedule_inquiry"
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


def _route_from_admissions(state: AgentState) -> Literal["__end__"]:
    return END


def build_school_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("admissions_collect", admissions_collect_node)
    graph.add_node("course_inquiry", course_inquiry_node)
    graph.add_node("schedule_inquiry", schedule_inquiry_node)
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
            "admissions_collect": "admissions_collect",
            "course_inquiry": "course_inquiry",
            "schedule_inquiry": "schedule_inquiry",
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
            "admissions_collect": "admissions_collect",
            "course_inquiry": "course_inquiry",
            "schedule_inquiry": "schedule_inquiry",
            "greeting_response": "greeting_response",
            "faq_response": "faq_response",
            "escalation": "escalation",
            END: END,
        },
    )
    graph.add_edge("admissions_collect", END)
    graph.add_edge("course_inquiry", END)
    graph.add_edge("schedule_inquiry", END)
    graph.add_edge("greeting_response", END)
    graph.add_edge("faq_response", END)
    graph.add_edge("escalation", END)

    return graph.compile(checkpointer=checkpointer)
