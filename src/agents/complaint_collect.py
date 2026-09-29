
import json

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..json_utils import strip_json_fences
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from ..usage import make_usage_record
from .utils import format_user_context, language_instruction, latest_user_text, resolve_persona, resolve_prompt
from ..services.command_bus import command_bus_client, CommandResult

logger = structlog.get_logger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """Eres un asistente de extracción de datos de quejas/reclamos.

Extrae del mensaje del usuario la siguiente información:
- order_ref: Número de pedido o referencia relacionada
- issue_description: Descripción del problema o queja
- desired_resolution: Lo que el cliente espera como solución (reembolso, reemplazo, etc.)

Devuelve un objeto JSON con los campos encontrados. Omite campos no mencionados.
Responde SOLO con el objeto JSON — sin markdown, sin explicación.
"""

_CONVERSATIONAL_SYSTEM_PROMPT = """Eres {persona}, una asistente de atención al cliente por WhatsApp.
El cliente tiene una queja o reclamo.

Tu tarea: recopilar la información necesaria para procesar la queja de forma empática y profesional.
Necesitas:
- Referencia del pedido (order_ref) — número de pedido u orden
- Descripción del problema (issue_description) — qué pasó
- Resolución deseada (desired_resolution) — qué espera el cliente

Campos ya recopilados: {collected_fields}
Campos faltantes: {missing_fields}

Reglas:
- Sé empática y comprensiva — el cliente tiene un problema.
- Si ya tienes toda la información, confirma y dile que vas a registrar y procesar su queja.
- Sé concisa — es WhatsApp.
- {language_rule}
- NO devuelvas JSON.
"""


async def complaint_collect_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("complaint_collect")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")
    complaint_data: dict = dict(state.get("complaint_data") or {})

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="complaint_collect",
        metadata={"thread_id": thread_id, "node": "complaint_collect"},
    )

    write = get_stream_writer()
    write({"type": "step_progress", "step": 1, "total_steps": 1, "topic": "Registro de queja"})

    has_new_message = bool(state["messages"]) and getattr(state["messages"][-1], "type", "") == "human"
    # The whole trailing burst of user messages — complaints are typically
    # narrated across several rapid WhatsApp messages.
    user_turn_text = latest_user_text(state)
    turn_usage: list = []

    agent_type = (state.get("agent_type") or "sales").upper()
    complaint_ext_key = f"RESTAURANT_COMPLAINT_EXTRACTION" if agent_type == "RESTAURANT" else "COMPLAINT_EXTRACTION"
    complaint_conv_key = f"RESTAURANT_COMPLAINT_CONVERSATIONAL" if agent_type == "RESTAURANT" else "COMPLAINT_CONVERSATIONAL"

    if has_new_message:
        complaint_ext_prompt = resolve_prompt(config, complaint_ext_key, _EXTRACTION_SYSTEM_PROMPT, state)
        extraction_messages = [
            {"role": "system", "content": complaint_ext_prompt},
            {"role": "user", "content": user_turn_text},
        ]

        extraction_gen = trace.generation(
            name="complaint_extraction_llm",
            model=model,
            input={"messages": extraction_messages},
        )

        extraction_stream = provider.stream_chat(
            model=model,
            messages=extraction_messages,
        )

        extraction_raw = ""
        async for chunk in extraction_stream:
            extraction_raw += chunk
        turn_usage.append(
            make_usage_record(
                node="complaint_collect.extraction", provider=provider, model=model,
            )
        )

        extraction_gen.end(output=extraction_raw)

        try:
            extracted = json.loads(strip_json_fences(extraction_raw))
            if isinstance(extracted, dict):
                complaint_data.update({k: v for k, v in extracted.items() if v})
        except (json.JSONDecodeError, ValueError):
            logger.warning("complaint_extraction_failed", raw=extraction_raw[:200])

    # Check completeness — need at least issue_description + one of order_ref or desired_resolution
    required_fields = ["order_ref", "issue_description", "desired_resolution"]
    missing_fields = [f for f in required_fields if not complaint_data.get(f)]
    complaint_complete = len(missing_fields) == 0

    collected_summary = ", ".join(f"{k}={repr(v)}" for k, v in complaint_data.items()) or "ninguno"
    missing_summary = ", ".join(missing_fields) if missing_fields else "todos respondidos"
    user_ctx_str = format_user_context(state)

    if complaint_complete:
        result = await command_bus_client.send_command(
            command="create_ticket",
            tenant_id=state["tenant_id"],
            contact_id=state["contact_id"],
            conversation_id=state["conversation_id"],
            payload=complaint_data
        )

        if result.success:
            conv_prompt = (
                "Tienes toda la información necesaria para procesar la queja del cliente. "
                "Confirma los datos empáticamente y dile que vas a registrar y procesar su queja. "
                f"Incluye el ID del ticket en el mensaje: {result.data.get('id')}."
                "Sé breve y comprensiva."
            ) + user_ctx_str
            conv_messages = [{"role": "system", "content": conv_prompt}]
            return {
                "messages": [AIMessage(content="")],
                "complaint_complete": True,
                "execute_confirmed": True,
                "complaint_data": result.data,
                "turn_usage": turn_usage,
            }
        else:
            conv_prompt = (
                "Hubo un problema al registrar la queja. "
                f"Error: {result.error['message']}. "
                "Informa al usuario del problema y pregunta si quiere intentar de nuevo."
            ) + user_ctx_str
            conv_messages = [{"role": "system", "content": conv_prompt}]
    else:
        complaint_conv_prompt = resolve_prompt(config, complaint_conv_key, _CONVERSATIONAL_SYSTEM_PROMPT, state)
        conv_messages = [
            {
                "role": "system",
                "content": complaint_conv_prompt.format(
                    persona=resolve_persona(state, "Helena"),
                    collected_fields=collected_summary,
                    missing_fields=missing_summary,
                    language_rule=language_instruction(state.get("language", "en")),
                ) + user_ctx_str,
            },
            {
                "role": "user",
                "content": (
                    user_turn_text
                    if has_new_message
                    else "Tengo un problema con mi pedido"
                ),
            },
        ]

    conv_gen = trace.generation(
        name="complaint_conversational_llm",
        model=model,
        input={"messages": conv_messages},
    )

    conv_stream = provider.stream_chat(
        model=model,
        messages=conv_messages,
    )

    full_response = ""
    async for chunk in conv_stream:
        write({"type": "token", "content": chunk})
        full_response += chunk
    turn_usage.append(
        make_usage_record(
            node="complaint_collect.reply", provider=provider, model=model,
        )
    )

    conv_gen.end(output=full_response)

    return {
        "messages": [AIMessage(content=full_response)],
        "complaint_data": complaint_data,
        "complaint_complete": complaint_complete,
        "turn_usage": turn_usage,
    }

