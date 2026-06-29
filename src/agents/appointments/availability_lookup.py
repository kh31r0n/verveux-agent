"""availability_lookup — call /internal/appointments/availability and present slots.

Computes a default search window (now → +14 days) and asks the backend for the
top N slots. Falls through to ``reservation_propose`` in the same turn so the
user receives one cohesive reply.
"""

from datetime import datetime, timedelta, timezone

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ...graphs.state import AgentState
from ...providers.registry import get_provider, resolve_model
from ...observability import record_node_invocation
from ...usage import make_usage_record
from ..backend_client import search_appointment_availability
from ..utils import language_instruction, resolve_persona, resolve_prompt

logger = structlog.get_logger(__name__)

_DEFAULT_HORIZON_DAYS = 14
_DEFAULT_LIMIT = 6

_CONVERSATIONAL_PROMPT = """Eres {persona}, una asistente de agendamiento por WhatsApp. {language_rule}

Te entrego una lista numerada de slots disponibles para una cita de tipo "{appointment_type_name}".
Tu tarea: presentar los slots al usuario y pedirle que elija uno.

Reglas:
- Conversa con calidez y brevedad.
- Numera las opciones del 1 al N, con fecha y hora local del slot.
- Pide al usuario que responda con el número o con el horario.
- Si la lista está vacía, discúlpate y sugiere intentar otro día o tipo.
- No inventes horarios; usa solamente los provistos.

Slots disponibles:
{slot_lines}
"""


def _format_slot_label(slot: dict) -> str:
    starts = slot.get("startsAt", "")
    try:
        dt = datetime.fromisoformat(starts.replace("Z", "+00:00"))
        return dt.strftime("%a %d %b · %H:%M UTC")
    except Exception:
        return starts


async def availability_lookup_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("availability_lookup")

    tenant_id = state.get("tenant_id", "")
    appointment_type_id = state.get("appointment_type_id")
    appointment_type_name = state.get("appointment_type_name") or "tu cita"

    turn_usage: list = []

    if not appointment_type_id:
        # Defensive: routing should not reach here without a type, but if it
        # does we punt back to collection.
        return {
            "candidate_slots": [],
            "messages": [
                AIMessage(
                    content="Aún necesito saber qué tipo de cita quieres reservar. ¿Cuál te interesa?"
                )
            ],
        }

    now = datetime.now(timezone.utc).replace(microsecond=0)
    horizon = now + timedelta(days=_DEFAULT_HORIZON_DAYS)
    try:
        result = await search_appointment_availability(
            tenant_id=tenant_id,
            appointment_type_id=appointment_type_id,
            from_iso=now.isoformat(),
            to_iso=horizon.isoformat(),
            limit=_DEFAULT_LIMIT,
        )
    except Exception as exc:
        logger.error("availability_search_failed", error=str(exc))
        return {
            "candidate_slots": [],
            "messages": [
                AIMessage(
                    content="No pude consultar disponibilidad en este momento. Inténtalo de nuevo en unos minutos."
                )
            ],
        }

    slots: list[dict] = result.get("slots") if isinstance(result, dict) else []
    slot_lines = (
        "\n".join(
            f"{i + 1}. {_format_slot_label(s)} (recursos: {len(s.get('resources', []))})"
            for i, s in enumerate(slots)
        )
        if slots
        else "(sin slots en el rango actual)"
    )

    provider = get_provider(config)
    model = resolve_model(config)
    conv_prompt = resolve_prompt(
        config, "APPOINTMENT_AVAILABILITY", _CONVERSATIONAL_PROMPT, state
    )
    system_content = conv_prompt.format(
        persona=resolve_persona(state, "Marco"),
        language_rule=language_instruction(state.get("language", "es")),
        appointment_type_name=appointment_type_name,
        slot_lines=slot_lines,
    )
    history = [
        {
            "role": "user" if getattr(m, "type", "") == "human" else "assistant",
            "content": m.content,
        }
        for m in (state.get("messages") or [])[-4:]
        if hasattr(m, "content") and m.content
    ]
    messages_payload = [{"role": "system", "content": system_content}] + history
    write = get_stream_writer()
    full_response = ""
    async for chunk in provider.stream_chat(model=model, messages=messages_payload):
        write({"type": "token", "content": chunk})
        full_response += chunk
    turn_usage.append(
        make_usage_record(
            node="availability_lookup.reply", provider=provider, model=model
        )
    )

    return {
        "candidate_slots": slots,
        "chosen_slot": None,
        "slot_conflict": False,
        "messages": [AIMessage(content=full_response)],
        "turn_usage": turn_usage,
    }
