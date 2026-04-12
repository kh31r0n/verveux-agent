"""
customer_data_collect — CUSTOMER_DATA phase.

Responsibility:
  Collect the delivery and payment information needed to fulfil the order.
  This is a single conversational step (no sub-steps), replacing the
  former sales_collect steps 1 and 3.

Required fields (must all be present before advancing):
    customer_name, delivery_address

Optional fields (collected opportunistically):
    customer_phone, customer_email, delivery_date_preference, payment_method

Phase transition:
  Sets customer_data_complete = True and sales_phase = "payment" once all
  required fields are confirmed.  Uses user_context to pre-populate fields
  the NestJS backend already knows about, reducing unnecessary questions.
"""

import json

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation
from ..services.cart import CartService
from .utils import format_user_context, language_instruction

logger = structlog.get_logger(__name__)

_REQUIRED_FIELDS = ["customer_name", "delivery_address"]
_ALL_FIELDS = [
    "customer_name",
    "customer_phone",
    "customer_email",
    "delivery_address",
    "delivery_date_preference",
    "payment_method",
]

_FIELD_LABELS = {
    "customer_name": "Nombre completo",
    "customer_phone": "Teléfono de contacto",
    "customer_email": "Correo electrónico",
    "delivery_address": "Dirección de entrega",
    "delivery_date_preference": "Fecha preferida de entrega",
    "payment_method": "Método de pago (efectivo, transferencia, tarjeta)",
}

# ── Extraction prompt ─────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM_PROMPT = """Eres un extractor de datos de entrega para un pedido de compra.

Extrae del mensaje del usuario los siguientes campos si están presentes:
- customer_name: nombre completo del cliente
- customer_phone: número de teléfono
- customer_email: correo electrónico
- delivery_address: dirección de entrega completa
- delivery_date_preference: preferencia de fecha/horario de entrega
- payment_method: método de pago mencionado

Devuelve SOLO un objeto JSON con los campos encontrados.
Omite campos que NO fueron mencionados.
Sin markdown, sin explicación.
"""

# ── Conversational prompt ─────────────────────────────────────────────────────

_CONV_SYSTEM_PROMPT = """Eres Helena, asistente de ventas por WhatsApp. {language_rule}

El cliente ya confirmó su carrito y ahora necesitas sus datos de entrega.

Carrito confirmado:
{cart_summary}

Campos ya recopilados: {collected}
Campos faltantes: {missing}

Instrucción: {instruction}

Reglas:
- Sé amigable y muy concisa — es WhatsApp.
- Si ya tienes todos los campos requeridos, confirma y dile que presentarás el resumen final.
- Pregunta máximo 2 campos por mensaje.
- NO devuelvas JSON.
"""


async def customer_data_collect_node(state: AgentState, config: RunnableConfig) -> dict:
    record_node_invocation("customer_data_collect")

    api_key = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id = state.get("thread_id", "unknown")
    lang = state.get("language", "es")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="customer_data_collect",
        metadata={"thread_id": thread_id, "node": "customer_data_collect"},
    )

    write = get_stream_writer()
    write({"type": "step_progress", "step": 1, "total_steps": 1, "topic": "Datos de entrega"})

    # ── Seed order_data from user_context if provided by NestJS ──────────────
    order_data: dict = dict(state.get("order_data") or {})
    user_ctx: dict = state.get("user_context") or {}

    # Pre-populate from user_context without overwriting anything already collected
    prefill_map = {
        "customer_name": user_ctx.get("name"),
        "customer_phone": user_ctx.get("phone"),
        "customer_email": user_ctx.get("email"),
        "delivery_address": user_ctx.get("address"),
    }
    for field, value in prefill_map.items():
        if value and not order_data.get(field):
            order_data[field] = value

    has_new_message = (
        bool(state["messages"])
        and getattr(state["messages"][-1], "type", "") == "human"
    )

    # ── Extract fields from user message ─────────────────────────────────────
    if has_new_message:
        extraction_messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": state["messages"][-1].content},
        ]

        extraction_gen = trace.generation(
            name="customer_data_extraction_llm",
            model="gpt-5.4-nano",
            input={"messages": extraction_messages},
        )

        extraction_stream = await client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=extraction_messages,
            stream=True,
        )
        extraction_raw = ""
        async for chunk in extraction_stream:
            delta = chunk.choices[0].delta.content if chunk.choices else ""
            if delta:
                extraction_raw += delta
        extraction_gen.end(output=extraction_raw)

        try:
            extracted = json.loads(extraction_raw.strip())
            if isinstance(extracted, dict):
                order_data.update({k: v for k, v in extracted.items() if v})
        except (json.JSONDecodeError, ValueError):
            logger.warning("customer_data_extraction_parse_failed", raw=extraction_raw[:200])

    # ── Check completeness ────────────────────────────────────────────────────
    missing_required = [f for f in _REQUIRED_FIELDS if not order_data.get(f)]
    missing_all = [f for f in _ALL_FIELDS if not order_data.get(f)]
    customer_data_complete = len(missing_required) == 0

    # ── Build conversational response ─────────────────────────────────────────
    cart: list = state.get("cart") or []
    cart_summary = CartService.format_cart(cart)

    collected_str = (
        ", ".join(f"{_FIELD_LABELS.get(k, k)}: {v}" for k, v in order_data.items() if v)
        or "ninguno aún"
    )
    missing_str = (
        ", ".join(_FIELD_LABELS.get(f, f) for f in missing_all)
        if missing_all
        else "todos recopilados"
    )

    if customer_data_complete:
        instruction = (
            "Tienes todos los datos requeridos para el pedido. "
            "Confirma brevemente los datos de entrega y dile al usuario "
            "que a continuación verá el resumen final para confirmar."
        )
    elif not has_new_message:
        instruction = (
            "Primera vez en este paso. Saluda al usuario por su nombre si lo tienes "
            "y pide amablemente los datos de entrega que faltan. "
            "Empieza por el nombre y la dirección si no los tienes."
        )
    else:
        next_missing = missing_required[:2] if missing_required else missing_all[:2]
        fields_to_ask = " y ".join(_FIELD_LABELS.get(f, f) for f in next_missing)
        instruction = f"Pregunta por: {fields_to_ask}."

    conv_messages = [
        {
            "role": "system",
            "content": _CONV_SYSTEM_PROMPT.format(
                language_rule=language_instruction(lang),
                cart_summary=cart_summary,
                collected=collected_str,
                missing=missing_str,
                instruction=instruction,
            ) + format_user_context(state),
        },
        {
            "role": "user",
            "content": (
                state["messages"][-1].content
                if has_new_message
                else "Continuar con los datos de entrega"
            ),
        },
    ]

    conv_gen = trace.generation(
        name="customer_data_conv_llm",
        model="gpt-5.4-nano",
        input={"messages": conv_messages},
    )

    conv_stream = await client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=conv_messages,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_response = ""
    prompt_tokens = 0
    completion_tokens = 0

    async for chunk in conv_stream:
        delta = chunk.choices[0].delta.content if chunk.choices else ""
        if delta:
            write({"type": "token", "content": delta})
            full_response += delta
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    conv_gen.end(output=full_response, usage={"input": prompt_tokens, "output": completion_tokens})

    logger.info(
        "customer_data_collect_done",
        thread_id=thread_id,
        complete=customer_data_complete,
        missing_required=missing_required,
    )

    update: dict = {
        "messages": [AIMessage(content=full_response)],
        "order_data": order_data,
        "customer_data_complete": customer_data_complete,
    }
    if customer_data_complete:
        update["sales_phase"] = "payment"

    return update
