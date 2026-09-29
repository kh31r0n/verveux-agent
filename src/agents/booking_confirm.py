"""Appointments booking confirmation node — confirms booking details."""

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from ..usage import make_usage_record
from .utils import language_instruction, latest_user_messages, resolve_prompt, format_user_context

logger = structlog.get_logger(__name__)

_CONFIRMATION_SYSTEM_PROMPT = """Eres Helena, una asistente de reservaciones por WhatsApp.

Has recopilado toda la información de la cita del cliente.
Genera un resumen claro de la reservación con:

- **Cliente**: nombre y teléfono
- **Servicio**: tipo de servicio solicitado
- **Fecha y hora**: cuándo es la cita
- **Notas**: cualquier información adicional

Después del resumen, pregunta:
"¿Los datos de tu cita son correctos? Responde **confirmar** para reservar, o dime qué ajustar."

{language_rule}"""


async def booking_confirm_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("booking_confirm")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="booking_confirm_node",
        metadata={"thread_id": thread_id, "node": "booking_confirm"},
    )

    booking_data = state.get("booking_data") or {}

    lang_rule = language_instruction(state.get("language", "en"))
    prompt = resolve_prompt(config, "BOOKING_CONFIRMATION", _CONFIRMATION_SYSTEM_PROMPT, state)
    system_content = prompt.format(language_rule=lang_rule)

    # Add booking data context
    booking_info = "\n\nDatos de la reservación:"
    if booking_data.get("customer_name"):
        booking_info += f"\n- Cliente: {booking_data['customer_name']}"
    if booking_data.get("customer_phone"):
        booking_info += f"\n- Teléfono: {booking_data['customer_phone']}"
    if booking_data.get("service_type"):
        booking_info += f"\n- Servicio: {booking_data['service_type']}"
    if booking_data.get("preferred_date"):
        booking_info += f"\n- Fecha: {booking_data['preferred_date']}"
    if booking_data.get("preferred_time"):
        booking_info += f"\n- Hora: {booking_data['preferred_time']}"
    if booking_data.get("notes"):
        booking_info += f"\n- Notas: {booking_data['notes']}"

    system_content += booking_info
    system_content += format_user_context(state)

    history = [
        {"role": "user" if getattr(m, "type", "") == "human" else "assistant", "content": m.content}
        for m in state["messages"][-6:]
        if hasattr(m, "content") and m.content
    ]
    messages_payload = [{"role": "system", "content": system_content}] + history

    generation = trace.generation(name="booking_confirm_llm", model=model, input={"messages": messages_payload})
    write = get_stream_writer()
    full_response = ""
    async for chunk in provider.stream_chat(model=model, messages=messages_payload):
        write({"type": "token", "content": chunk})
        full_response += chunk
    generation.end(output=full_response)

    # Check if user confirmed — the confirmation keyword may be any fragment
    # of a multi-message burst ("mmm ok" / "confirmar"), so check each one.
    confirmed = any(
        fragment.lower().strip() in ("confirmar", "sí", "si", "yes", "confirm")
        for fragment in latest_user_messages(state)
    )

    return {
        "messages": [AIMessage(content=full_response)],
        "booking_confirmed": confirmed,
        "turn_usage": [
            make_usage_record(node="booking_confirm", provider=provider, model=model)
        ],
    }
