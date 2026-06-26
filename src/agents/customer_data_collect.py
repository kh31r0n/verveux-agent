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

Double-response guard:
  When routing auto-chains from sales_confirm → customer_data_collect in the
  same streaming turn, the last message in history is already an AI message
  (sales_confirm's response). In that case we must NOT emit another response
  or the user sees two AI messages in a single turn. The guard checks:
    - last message is AI
    - cart_confirmed == True
    - customer_data_complete == False (we haven't finished yet)
  If all three hold, return {} immediately and let the next user turn trigger
  the actual data collection.
"""

import json

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from ..services.cart import CartService
from ..usage import make_usage_record
from .utils import format_user_context, language_instruction, resolve_persona

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

_CONV_SYSTEM_PROMPT = """Eres {persona}, asistente de ventas por WhatsApp. {language_rule}

El cliente ya confirmó su carrito y ahora necesitas sus datos de entrega.
Ve directo al punto — no saludes ni uses frases de transición como "¡Perfecto!" o "¡Genial!".

Carrito confirmado:
{cart_summary}

Campos ya recopilados: {collected}
Campos faltantes: {missing}

Instrucción: {instruction}

Reglas:
- Sé directo y muy conciso — es WhatsApp.
- NO repitas el saludo ni la confirmación del carrito.
- Si ya tienes todos los campos requeridos, confirma y dile que presentarás el resumen final.
- Pregunta máximo 2 campos por mensaje.
- NO devuelvas JSON.
"""


async def customer_data_collect_node(state: AgentState, config: RunnableConfig) -> dict:
    record_node_invocation("customer_data_collect")

    thread_id = state.get("thread_id", "unknown")

    # ── Double-response guard ─────────────────────────────────────────────────
    #
    # When sales_confirm confirms the cart (cart_confirmed=True) and routes
    # here in the same streaming turn, the last message is already an AI
    # message from sales_confirm. Emitting another response here would produce
    # two AI messages in a single turn, confusing the user.
    #
    # We detect this by checking:
    #   1. last message is an AI message (sales_confirm already responded)
    #   2. cart was just confirmed (we're entering this node for the first time)
    #   3. customer data is not yet complete (we still need to collect it)
    #
    # In this case we return {} to skip the response. The next user message
    # will re-enter this node with has_new_message=True and respond normally.
    last_msg = state["messages"][-1] if state.get("messages") else None
    last_is_ai = last_msg is not None and getattr(last_msg, "type", "") == "ai"
    cart_just_confirmed = state.get("cart_confirmed", False)
    already_collecting = state.get("customer_data_complete", False)

    if last_is_ai and cart_just_confirmed and not already_collecting:
        logger.info(
            "customer_data_collect_guard_skipped",
            thread_id=thread_id,
            reason="auto_chain_from_sales_confirm",
        )
        return {}

    provider = get_provider(config)
    model = resolve_model(config)
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
    turn_usage: list = []

    # ── Extract fields from user message ─────────────────────────────────────
    if has_new_message:
        extraction_messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": state["messages"][-1].content},
        ]

        extraction_gen = trace.generation(
            name="customer_data_extraction_llm",
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
                node="customer_data_collect.extraction", provider=provider, model=model,
            )
        )
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
            "Primera vez en este paso. Pide directamente los datos de entrega "
            "que faltan. Empieza por el nombre y la dirección si no los tienes. "
            "No uses frases de bienvenida ni transición."
        )
    else:
        next_missing = missing_required[:2] if missing_required else missing_all[:2]
        fields_to_ask = " y ".join(_FIELD_LABELS.get(f, f) for f in next_missing)
        instruction = f"Pregunta por: {fields_to_ask}."

    conv_messages = [
        {
            "role": "system",
            "content": _CONV_SYSTEM_PROMPT.format(
                persona=resolve_persona(state, "Helena"),
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
            node="customer_data_collect.reply", provider=provider, model=model,
        )
    )
    conv_gen.end(output=full_response)

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
        "turn_usage": turn_usage,
    }
    if customer_data_complete:
        update["sales_phase"] = "payment"

    return update