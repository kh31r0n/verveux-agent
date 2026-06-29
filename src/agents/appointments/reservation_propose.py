"""reservation_propose — parse the user's slot choice and pause for confirmation.

Uses LangGraph's ``interrupt()`` so the graph blocks here, the FastAPI layer
emits an ``interrupt_detected`` SSE event, and the user's next inbound
WhatsApp message resumes the graph via ``Command(resume=user_reply)``.

When the user confirms, ``confirmation`` runs in the resumed turn and creates
the hold. When the user picks a different slot, this node re-runs and the
graph loops back through availability_lookup if needed.
"""

import re

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from ...graphs.state import AgentState
from ...observability import record_node_invocation

logger = structlog.get_logger(__name__)


def _last_user_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages") or []):
        if getattr(msg, "type", "") == "human":
            return msg.content
    return ""


def _parse_choice(text: str, slots: list[dict]) -> dict | None:
    """Pick a slot based on the user's reply.

    Strategies, in order:
      1. A leading integer 1..N picks slots[N-1].
      2. Any time substring HH:MM that uniquely matches a slot's start time.
    Returns None if no unambiguous match.
    """
    if not text or not slots:
        return None

    digits = re.match(r"\s*(\d{1,2})\b", text)
    if digits:
        idx = int(digits.group(1)) - 1
        if 0 <= idx < len(slots):
            return slots[idx]

    time_match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if time_match:
        hh = int(time_match.group(1))
        mm = int(time_match.group(2))
        candidates = [
            s
            for s in slots
            if isinstance(s.get("startsAt"), str)
            and f"T{hh:02d}:{mm:02d}" in s["startsAt"]
        ]
        if len(candidates) == 1:
            return candidates[0]
    return None


async def reservation_propose_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("reservation_propose")

    slots = state.get("candidate_slots") or []
    if not slots:
        return {
            "chosen_slot": None,
            "messages": [
                AIMessage(
                    content="No tengo slots para proponer; déjame buscar disponibilidad nuevamente."
                )
            ],
        }

    last_user_msg = _last_user_message(state)
    choice = _parse_choice(last_user_msg, slots) if last_user_msg else None

    if not choice:
        # Couldn't parse — ask again. No LLM call needed; deterministic prompt.
        return {
            "chosen_slot": None,
            "messages": [
                AIMessage(
                    content=(
                        "Para confirmar tu cita responde con el número del horario que prefieres "
                        f"(del 1 al {len(slots)}) o escribe la hora (ej. 14:00)."
                    )
                )
            ],
        }

    starts = choice.get("startsAt", "")
    ends = choice.get("endsAt", "")
    type_name = state.get("appointment_type_name") or "tu cita"

    # Block here for explicit confirmation. The payload reaches the
    # `interrupt_detected` SSE event so the channel can display it to the
    # user; the resumed reply is appended to messages on the next turn and
    # `confirmation` reads it from state.
    user_reply: str = interrupt(
        {
            "type": "appointment_confirmation",
            "appointment_type": type_name,
            "starts_at": starts,
            "ends_at": ends,
            "resources": choice.get("resources", []),
            "prompt": (
                f"Para confirmar tu reserva de {type_name} el {starts}, "
                "responde 'sí' o escribe otra opción."
            ),
        }
    )

    confirmed = bool(
        user_reply
        and any(token in user_reply.lower() for token in ("si", "sí", "ok", "dale", "confirmo", "confirmar"))
    )
    if not confirmed:
        return {
            "chosen_slot": None,
            "messages": [
                AIMessage(
                    content="Sin problema — dime qué otro horario te conviene o pídeme buscar nuevamente."
                )
            ],
        }

    return {
        "chosen_slot": choice,
        "messages": [
            AIMessage(content=f"Perfecto — voy a confirmar {type_name} para {starts}.")
        ],
    }
