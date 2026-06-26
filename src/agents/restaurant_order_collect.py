"""Restaurant order collection node — two-stage extraction + conversation."""

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

_EXTRACTION_SYSTEM_PROMPT = """Eres un asistente de extracción de datos de pedidos de restaurante.

Extrae del mensaje del usuario la siguiente información:
- items: Lista de platillos [{"name": str, "quantity": int, "notes": str}]
- service_type: "delivery" o "pickup"
- delivery_address: Dirección de entrega (si aplica)
- special_notes: Notas especiales generales del pedido

Para los items, usa los nombres exactos del menú/catálogo si coinciden.
Devuelve un objeto JSON con los campos encontrados. Omite campos no mencionados.
Responde SOLO con el objeto JSON — sin markdown, sin explicación."""

_CONVERSATIONAL_SYSTEM_PROMPT = """Eres {persona}, una asistente de pedidos por WhatsApp para un restaurante.
Eres amable y eficiente. {language_rule}

Tu tarea: ayudar al cliente a armar su pedido de comida.
Necesitas recopilar:
- Platillos seleccionados con cantidad (items)
- Tipo de servicio: para llevar o a domicilio (service_type)
- Si es a domicilio: dirección de entrega (delivery_address)

Campos ya recopilados: {collected_fields}
Campos faltantes: {missing_fields}

{catalog_info}

Reglas:
- Sé amigable y sugiere complementos (bebidas, postres) naturalmente.
- Confirma cada item agregado.
- Si ya tienes todos los datos, presenta un resumen del pedido.
- NO devuelvas JSON — esta es una respuesta en lenguaje natural."""

_REQUIRED_FIELDS = ["items", "service_type"]


def _format_catalog(state: AgentState) -> str:
    catalog = state.get("product_catalog") or []
    if not catalog:
        return ""
    items = []
    for item in catalog[:30]:
        line = f"- {item.get('name', 'Item')}"
        if item.get("price"):
            line += f" — ${item['price']}"
        items.append(line)
    return "Menú disponible:\n" + "\n".join(items)


async def restaurant_order_collect_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("restaurant_order_collect")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="restaurant_order_collect_node",
        metadata={"thread_id": thread_id, "node": "restaurant_order_collect"},
    )

    order_data: dict = state.get("restaurant_order_data") or {}
    turn_usage: list = []

    # ── Stage 1: Extraction ──────────────────────────────────────────────────
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", "") == "human":
            last_user_msg = msg.content
            break

    if last_user_msg:
        extraction_prompt = resolve_prompt(
            config, "RESTAURANT_ORDER_EXTRACTION", _EXTRACTION_SYSTEM_PROMPT, state
        )
        extraction_messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": last_user_msg},
        ]

        gen = trace.generation(name="restaurant_order_extraction", model=model, input={"messages": extraction_messages})
        extracted_text = ""
        async for chunk in provider.stream_chat(model=model, messages=extraction_messages):
            extracted_text += chunk
        turn_usage.append(
            make_usage_record(
                node="restaurant_order_collect.extraction",
                provider=provider,
                model=model,
            )
        )
        gen.end(output=extracted_text)

        try:
            extracted = json.loads(extracted_text)
            # Merge items lists
            if "items" in extracted and "items" in order_data:
                order_data["items"].extend(extracted["items"])
                del extracted["items"]
            order_data.update(extracted)
        except (json.JSONDecodeError, TypeError):
            pass

    # ── Stage 2: Check completion ────────────────────────────────────────────
    collected = [f for f in _REQUIRED_FIELDS if order_data.get(f)]
    missing = [f for f in _REQUIRED_FIELDS if not order_data.get(f)]

    # If delivery, also require address
    if order_data.get("service_type") == "delivery" and not order_data.get("delivery_address"):
        missing.append("delivery_address")

    is_complete = len(missing) == 0

    # ── Stage 3: Conversational response ─────────────────────────────────────
    lang_rule = language_instruction(state.get("language", "en"))
    conv_prompt = resolve_prompt(
        config, "RESTAURANT_ORDER_CONVERSATIONAL", _CONVERSATIONAL_SYSTEM_PROMPT, state
    )

    collected_str = ", ".join(f"{f}={order_data.get(f)}" for f in collected)
    system_content = conv_prompt.format(
        persona=resolve_persona(state, "Helena"),
        language_rule=lang_rule,
        collected_fields=collected_str or "Ninguno aún",
        missing_fields=", ".join(missing) if missing else "Ninguno — pedido completo",
        catalog_info=_format_catalog(state),
    )
    system_content += format_user_context(state)

    history = [
        {"role": "user" if getattr(m, "type", "") == "human" else "assistant", "content": m.content}
        for m in state["messages"][-6:]
        if hasattr(m, "content") and m.content
    ]
    messages_payload = [{"role": "system", "content": system_content}] + history

    gen2 = trace.generation(name="restaurant_order_conversational", model=model, input={"messages": messages_payload})
    write = get_stream_writer()
    full_response = ""
    async for chunk in provider.stream_chat(model=model, messages=messages_payload):
        write({"type": "token", "content": chunk})
        full_response += chunk
    turn_usage.append(
        make_usage_record(
            node="restaurant_order_collect.reply", provider=provider, model=model,
        )
    )
    gen2.end(output=full_response)

    return {
        "messages": [AIMessage(content=full_response)],
        "restaurant_order_data": order_data,
        "restaurant_order_complete": is_complete,
        "turn_usage": turn_usage,
    }
