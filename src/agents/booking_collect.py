"""Appointments booking collection node — two-stage extraction + conversation."""

import json

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from ..usage import make_usage_record
from .utils import language_instruction, resolve_persona, resolve_prompt, format_user_context

logger = structlog.get_logger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """Eres un asistente de extracción de datos de reservaciones.

Extrae del mensaje del usuario la siguiente información:
- customer_name: Nombre del cliente
- customer_phone: Teléfono de contacto
- preferred_date: Fecha preferida (formato YYYY-MM-DD si es posible)
- preferred_time: Hora preferida (formato HH:MM)
- service_type: Tipo de servicio o cita
- notes: Notas adicionales

Devuelve un objeto JSON con los campos encontrados. Omite campos no mencionados.
Responde SOLO con el objeto JSON — sin markdown, sin explicación."""

_CONVERSATIONAL_SYSTEM_PROMPT = """Eres {persona}, una asistente de reservaciones por WhatsApp.
Eres amable y eficiente. {language_rule}

Tu tarea: recopilar la información necesaria para completar una reservación.
Necesitas:
- Nombre del cliente (customer_name)
- Teléfono de contacto (customer_phone)
- Fecha preferida (preferred_date)
- Hora preferida (preferred_time)
- Tipo de servicio (service_type)
- Notas adicionales (notes) — opcional

Campos ya recopilados: {collected_fields}
Campos faltantes: {missing_fields}

Reglas:
- Sé concisa y amigable — es WhatsApp.
- Si ya tienes toda la información, confirma los datos.
- NO devuelvas JSON — esta es una respuesta en lenguaje natural."""

_REQUIRED_FIELDS = [
    "customer_name", "customer_phone",
    "preferred_date", "preferred_time", "service_type",
]


async def booking_collect_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("booking_collect")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="booking_collect_node",
        metadata={"thread_id": thread_id, "node": "booking_collect"},
    )

    booking_data: dict = state.get("booking_data") or {}
    turn_usage: list = []

    # ── Stage 1: Extraction ──────────────────────────────────────────────────
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", "") == "human":
            last_user_msg = msg.content
            break

    if last_user_msg:
        extraction_prompt = resolve_prompt(
            config, "BOOKING_EXTRACTION", _EXTRACTION_SYSTEM_PROMPT, state
        )
        extraction_messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": last_user_msg},
        ]

        gen = trace.generation(name="booking_extraction", model=model, input={"messages": extraction_messages})
        extracted_text = ""
        async for chunk in provider.stream_chat(model=model, messages=extraction_messages):
            extracted_text += chunk
        turn_usage.append(
            make_usage_record(
                node="booking_collect.extraction", provider=provider, model=model,
            )
        )
        gen.end(output=extracted_text)

        try:
            extracted = json.loads(extracted_text)
            booking_data.update(extracted)
        except (json.JSONDecodeError, TypeError):
            pass

    # ── Stage 2: Check completion ────────────────────────────────────────────
    collected = [f for f in _REQUIRED_FIELDS if booking_data.get(f)]
    missing = [f for f in _REQUIRED_FIELDS if not booking_data.get(f)]
    is_complete = len(missing) == 0

    # ── Stage 3: Conversational response ─────────────────────────────────────
    lang_rule = language_instruction(state.get("language", "en"))
    conv_prompt = resolve_prompt(
        config, "BOOKING_CONVERSATIONAL", _CONVERSATIONAL_SYSTEM_PROMPT, state
    )
    system_content = conv_prompt.format(
        persona=resolve_persona(state, "Helena"),
        language_rule=lang_rule,
        collected_fields=", ".join(f"{f}={booking_data[f]}" for f in collected),
        missing_fields=", ".join(missing) if missing else "Ninguno — todos recopilados",
    )
    system_content += format_user_context(state)

    history = [
        {"role": "user" if getattr(m, "type", "") == "human" else "assistant", "content": m.content}
        for m in state["messages"][-6:]
        if hasattr(m, "content") and m.content
    ]
    messages_payload = [{"role": "system", "content": system_content}] + history

    gen2 = trace.generation(name="booking_conversational", model=model, input={"messages": messages_payload})
    write = get_stream_writer()
    full_response = ""
    async for chunk in provider.stream_chat(model=model, messages=messages_payload):
        write({"type": "token", "content": chunk})
        full_response += chunk
    turn_usage.append(
        make_usage_record(node="booking_collect.reply", provider=provider, model=model)
    )
    gen2.end(output=full_response)

    return {
        "messages": [AIMessage(content=full_response)],
        "booking_data": booking_data,
        "booking_complete": is_complete,
        "turn_usage": turn_usage,
    }
