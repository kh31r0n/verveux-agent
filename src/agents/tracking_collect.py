import json

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from ..usage import make_usage_record
from .utils import format_user_context, language_instruction, resolve_persona, resolve_prompt

logger = structlog.get_logger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """Eres un asistente de extracción de datos de seguimiento de pedidos.

Extrae del mensaje del usuario la siguiente información:
- order_id: Número o ID del pedido
- customer_name: Nombre del cliente (si lo proporciona)
- customer_phone: Teléfono del cliente (si lo proporciona)

Devuelve un objeto JSON con los campos encontrados. Omite campos no mencionados.
Responde SOLO con el objeto JSON — sin markdown, sin explicación.
"""

_CONVERSATIONAL_SYSTEM_PROMPT = """Eres {persona}, una asistente de atención al cliente por WhatsApp.
El cliente quiere rastrear un pedido.

Tu tarea: pedir al cliente la información necesaria para buscar su pedido.
Necesitas al menos uno de estos datos:
- Número de pedido (order_id)
- Nombre del cliente (customer_name)
- Teléfono del cliente (customer_phone)

Campos ya recopilados: {collected_fields}
Campos faltantes: {missing_info}

Reglas:
- Si ya tienes al menos el número de pedido O (nombre + teléfono), di que tienes suficiente info.
- Sé concisa y amigable — es WhatsApp.
- {language_rule}
- NO devuelvas JSON.
"""


async def tracking_collect_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("tracking_collect")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")
    tracking_data: dict = dict(state.get("tracking_data") or {})

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="tracking_collect",
        metadata={"thread_id": thread_id, "node": "tracking_collect"},
    )

    write = get_stream_writer()
    write({"type": "step_progress", "step": 1, "total_steps": 1, "topic": "Rastreo de pedido"})

    has_new_message = bool(state["messages"]) and getattr(state["messages"][-1], "type", "") == "human"
    turn_usage: list = []

    if has_new_message:
        tracking_ext_prompt = resolve_prompt(config, "TRACKING_EXTRACTION", _EXTRACTION_SYSTEM_PROMPT, state)
        extraction_messages = [
            {"role": "system", "content": tracking_ext_prompt},
            {"role": "user", "content": state["messages"][-1].content},
        ]

        extraction_gen = trace.generation(
            name="tracking_extraction_llm",
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
                node="tracking_collect.extraction", provider=provider, model=model,
            )
        )

        extraction_gen.end(output=extraction_raw)

        try:
            extracted = json.loads(extraction_raw.strip())
            if isinstance(extracted, dict):
                tracking_data.update({k: v for k, v in extracted.items() if v})
        except (json.JSONDecodeError, ValueError):
            logger.warning("tracking_extraction_failed", raw=extraction_raw[:200])

    # Determine if we have enough info
    has_order_id = bool(tracking_data.get("order_id"))
    has_name_phone = bool(tracking_data.get("customer_name")) and bool(tracking_data.get("customer_phone"))
    tracking_complete = has_order_id or has_name_phone

    # Generate conversational response
    collected_summary = ", ".join(f"{k}={repr(v)}" for k, v in tracking_data.items()) or "ninguno"
    missing_info = []
    if not has_order_id:
        missing_info.append("order_id")
    if not tracking_data.get("customer_name"):
        missing_info.append("customer_name")
    if not tracking_data.get("customer_phone"):
        missing_info.append("customer_phone")

    user_ctx_str = format_user_context(state)

    if tracking_complete:
        conv_prompt = (
            "Tienes suficiente información para buscar el pedido del cliente. "
            "Confirma los datos que tienes y dile que vas a buscar su pedido. "
            "Sé breve."
        ) + user_ctx_str
        conv_messages = [{"role": "system", "content": conv_prompt}]
    else:
        tracking_conv_prompt = resolve_prompt(config, "TRACKING_CONVERSATIONAL", _CONVERSATIONAL_SYSTEM_PROMPT, state)
        conv_messages = [
            {
                "role": "system",
                "content": tracking_conv_prompt.format(
                    persona=resolve_persona(state, "Helena"),
                    collected_fields=collected_summary,
                    missing_info=", ".join(missing_info),
                    language_rule=language_instruction(state.get("language", "en")),
                ) + user_ctx_str,
            },
            {
                "role": "user",
                "content": (
                    state["messages"][-1].content
                    if has_new_message
                    else "Quiero rastrear mi pedido"
                ),
            },
        ]

    conv_gen = trace.generation(
        name="tracking_conversational_llm",
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
            node="tracking_collect.reply", provider=provider, model=model,
        )
    )

    conv_gen.end(output=full_response)

    return {
        "messages": [AIMessage(content=full_response)],
        "tracking_data": tracking_data,
        "tracking_complete": tracking_complete,
        "turn_usage": turn_usage,
    }
