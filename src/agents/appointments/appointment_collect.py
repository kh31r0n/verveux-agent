"""appointment_collect — pick the AppointmentType and collect required customer fields.

Two-stage pattern (extraction → conversational reply) consistent with
sales_collect / booking_collect. Talks to /internal/appointments/types to
discover the tenant's catalogue without trusting the LLM to invent types.
"""

import json

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ...graphs.state import AgentState
from ...json_utils import strip_json_fences
from ...providers.registry import get_provider, resolve_model
from ...observability import record_node_invocation
from ...usage import make_usage_record
from ..backend_client import list_appointment_types
from ..utils import (
    format_user_context,
    language_instruction,
    resolve_persona,
    resolve_prompt,
)

logger = structlog.get_logger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """Eres un asistente de extracción de datos para agendamientos.

Te entrego (1) el catálogo de tipos de cita disponibles, (2) los campos requeridos del tipo seleccionado si ya está elegido, y (3) los datos ya recopilados. Tu tarea:

1. Si aún no se ha seleccionado un tipo de cita, identifica si el mensaje del usuario menciona uno por nombre o palabra clave.
2. Extrae cualquier valor que el usuario haya provisto para los campos requeridos.

Devuelve SOLO un objeto JSON con esta forma (omite campos no detectados):
{{"appointment_type_id": "<uuid>", "fields": {{"<key>": "<value>", ...}}}}

Catálogo:
{type_catalog}

Tipo seleccionado: {selected_type}
Campos requeridos:
{required_fields}
Datos ya recopilados:
{collected}
"""

_CONVERSATIONAL_SYSTEM_PROMPT = """Eres {persona}, una asistente de agendamiento por WhatsApp. {language_rule}

Tu misión: ayudar al usuario a (1) elegir un tipo de cita y (2) entregar los datos requeridos.

Reglas:
- Sé cálida, breve y directa. Esto es WhatsApp.
- NUNCA inventes tipos de cita ni campos. Usa solamente lo del catálogo.
- Si aún no hay tipo seleccionado, ofrece la lista de opciones (nombres) y pregunta cuál.
- Si hay tipo pero faltan campos, pídelos uno o dos a la vez en orden razonable.
- Si tienes todo, confirma que avanzarás a buscar disponibilidad.
- No respondas en JSON; usa lenguaje natural.

Catálogo disponible:
{type_catalog}

Tipo seleccionado: {selected_type}
Campos requeridos:
{required_fields}
Datos ya recopilados:
{collected}
Campos faltantes:
{missing}
"""


def _format_catalog(types: list[dict]) -> str:
    if not types:
        return "(sin tipos configurados)"
    return "\n".join(
        f"- id={t.get('id')} | name={t.get('name')} | duración={t.get('durationMinutes')}min"
        for t in types
    )


def _format_required_fields(fields: list[dict]) -> str:
    if not fields:
        return "(ninguno)"
    return "\n".join(
        f"- {f.get('key')} ({f.get('label')}, type={f.get('type')}, required={f.get('required')})"
        for f in fields
    )


def _format_collected(data: dict) -> str:
    if not data:
        return "(vacío)"
    return ", ".join(f"{k}={v}" for k, v in data.items())


def _last_user_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages") or []):
        if getattr(msg, "type", "") == "human":
            return msg.content
    return ""


def _autoselect_sole_type(
    types: list[dict], appointment_type_id: str | None
) -> str | None:
    """Single-service tenants: adopt the only configured type.

    The extraction stage sees only the user's latest message, so a bare
    confirmation ("sí, agéndala") to a one-option offer can't resolve a type.
    Without a structurally selected type ``booking_complete`` stays False and
    the turn ends silently — even though the conversational reply (which does
    see history) confidently promised to search availability. Auto-selecting
    the sole type closes that gap.
    """
    if appointment_type_id:
        return appointment_type_id
    if len(types) == 1:
        return types[0].get("id")
    return None


async def appointment_collect_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("appointment_collect")

    tenant_id = state.get("tenant_id", "")
    turn_usage: list = []
    provider = get_provider(config)
    model = resolve_model(config)

    # Load catalogue once per turn. The backend is the source of truth.
    types: list[dict] = []
    try:
        types = await list_appointment_types(tenant_id)
    except Exception as exc:
        logger.warning("appointment_types_fetch_failed", error=str(exc))

    prior_type_id = state.get("appointment_type_id")
    appointment_type_id = _autoselect_sole_type(types, prior_type_id)
    selected_type: dict | None = (
        next((t for t in types if t.get("id") == appointment_type_id), None)
        if appointment_type_id
        else None
    )

    required_fields: list[dict] = state.get("required_customer_fields") or []
    collected: dict = dict(state.get("collected_customer_data") or {})

    # A freshly auto-selected sole type carries no prior field state — load its
    # required fields so the completion check below is accurate.
    if selected_type and appointment_type_id != prior_type_id:
        required_fields = selected_type.get("requiredCustomerFields") or []
        collected = {}

    # ── Stage 1: extraction from the user's latest message ────────────────────
    last_user_msg = _last_user_message(state)
    if last_user_msg:
        extraction_prompt = resolve_prompt(
            config,
            "APPOINTMENT_EXTRACTION",
            _EXTRACTION_SYSTEM_PROMPT,
            state,
        )
        system_content = extraction_prompt.format(
            type_catalog=_format_catalog(types),
            selected_type=(
                f"{selected_type.get('name')} ({selected_type.get('id')})"
                if selected_type
                else "(ninguno aún)"
            ),
            required_fields=_format_required_fields(required_fields),
            collected=_format_collected(collected),
        )
        # Include recent turns so a bare confirmation ("sí, agéndala") can be
        # resolved against the type the assistant just offered — the extraction
        # LLM otherwise sees only the standalone latest message.
        recent_history = [
            {
                "role": "user" if getattr(m, "type", "") == "human" else "assistant",
                "content": m.content,
            }
            for m in (state.get("messages") or [])[-6:-1]
            if hasattr(m, "content") and m.content
        ]
        extraction_messages = (
            [{"role": "system", "content": system_content}]
            + recent_history
            + [{"role": "user", "content": last_user_msg}]
        )
        extracted_text = ""
        async for chunk in provider.stream_chat(
            model=model, messages=extraction_messages
        ):
            extracted_text += chunk
        turn_usage.append(
            make_usage_record(
                node="appointment_collect.extraction",
                provider=provider,
                model=model,
            )
        )

        try:
            parsed = json.loads(strip_json_fences(extracted_text))
        except (json.JSONDecodeError, ValueError):
            logger.warning("appointment_collect_extraction_parse_failed", raw=extracted_text[:200])
            parsed = {}

        new_type_id = parsed.get("appointment_type_id")
        if new_type_id and any(t.get("id") == new_type_id for t in types):
            appointment_type_id = new_type_id
            selected_type = next(
                (t for t in types if t.get("id") == new_type_id), None
            )
            if selected_type:
                required_fields = selected_type.get("requiredCustomerFields") or []
                collected = {}

        fields = parsed.get("fields") or {}
        if isinstance(fields, dict):
            valid_keys = {f.get("key") for f in required_fields}
            for key, value in fields.items():
                if key in valid_keys and value:
                    collected[key] = value

    # ── Completion check ──────────────────────────────────────────────────────
    missing = [
        f.get("key")
        for f in required_fields
        if f.get("required") and not collected.get(f.get("key"))
    ]
    is_complete = bool(appointment_type_id) and not missing

    # ── Stage 2: conversational reply ─────────────────────────────────────────
    conv_prompt = resolve_prompt(
        config,
        "APPOINTMENT_CONVERSATIONAL",
        _CONVERSATIONAL_SYSTEM_PROMPT,
        state,
    )
    system_content = conv_prompt.format(
        persona=resolve_persona(state, "Marco"),
        language_rule=language_instruction(state.get("language", "es")),
        type_catalog=_format_catalog(types),
        selected_type=(
            selected_type.get("name") if selected_type else "(ninguno aún)"
        ),
        required_fields=_format_required_fields(required_fields),
        collected=_format_collected(collected),
        missing=", ".join(missing) if missing else "(ninguno — listo para buscar)",
    )
    system_content += format_user_context(state)

    history = [
        {
            "role": "user" if getattr(m, "type", "") == "human" else "assistant",
            "content": m.content,
        }
        for m in (state.get("messages") or [])[-6:]
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
            node="appointment_collect.reply", provider=provider, model=model
        )
    )

    update: dict = {
        "messages": [AIMessage(content=full_response)],
        "booking_intent": state.get("booking_intent") or "book",
        "appointment_type_id": appointment_type_id,
        "appointment_type_name": (
            selected_type.get("name") if selected_type else None
        ),
        "appointment_type_duration_min": (
            selected_type.get("durationMinutes") if selected_type else None
        ),
        "required_customer_fields": required_fields,
        "collected_customer_data": collected,
        "booking_complete": is_complete,
        "turn_usage": turn_usage,
    }
    return update
