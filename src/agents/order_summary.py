import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from .utils import language_instruction, resolve_prompt
from .backend_client import get_or_create_cart

logger = structlog.get_logger(__name__)

_SUMMARY_SYSTEM_PROMPT = """Eres Helena, una asistente de ventas por WhatsApp.

El backend ha calculado el resumen del carrito del cliente.
Presenta el resumen de forma clara y amigable con:

- **Productos** (lista de items con cantidad y precio unitario)
- **Total** (usa exactamente el grandTotal que se te proporciona — nunca recalcules)

Después del resumen, pregunta al cliente si desea confirmar:
"¿El pedido está correcto? Responde **confirmar** para enviarlo, o dime qué necesitas corregir."

{language_rule}
"""

_CORRECTION_SYSTEM_PROMPT = """Eres Helena, una asistente de ventas por WhatsApp.

El cliente quiere corregir algo en su pedido. Su solicitud es:
"{correction}"

Datos actuales del carrito:
{cart_summary}

Reconoce la corrección y dile que actualizarás el carrito. Luego pide confirmación nuevamente.

{language_rule}
"""

_CONFIRM_KEYWORDS = {"confirmar", "confirm", "yes", "sí", "si", "ok", "okay", "enviar", "dale", "listo", "perfecto"}


def _format_cart_for_llm(cart: dict) -> str:
    lines = []
    for item in cart.get("items", []):
        lines.append(
            f"- {item['productName']} x{item['quantity']} — ${item['unitPrice']:.2f} c/u = ${item['lineTotal']:.2f}"
        )
    lines.append(f"\n**Total: ${cart.get('grandTotal', 0):.2f} {cart.get('currency', 'USD')}**")
    return "\n".join(lines)


def _build_cart_from_state(state_cart: list) -> dict:
    """
    Convert the AgentState cart list into the same dict shape that the
    backend returns, so _format_cart_for_llm works identically for both.
    """
    items = [
        {
            "productName": item["name"],
            "quantity": item["qty"],
            "unitPrice": float(item["price"]),
            "lineTotal": round(item["qty"] * float(item["price"]), 2),
        }
        for item in state_cart
        if isinstance(item, dict)
    ]
    grand_total = round(sum(i["lineTotal"] for i in items), 2)
    return {
        "items": items,
        "grandTotal": grand_total,
        "currency": "USD",
    }


async def order_summary_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("order_summary")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")
    contact_id: str = state.get("contact_id", "")
    conversation_id: str = state.get("conversation_id", "")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="order_summary",
        metadata={"thread_id": thread_id, "node": "order_summary"},
    )

    write = get_stream_writer()
    lang_rule = language_instruction(state.get("language", "en"))
    write({"type": "step_progress", "step": 4, "total_steps": 4, "topic": "Resumen del pedido"})

    # ── Fetch backend cart (source of truth for totals) ───────────────────────
    cart: dict = {}
    backend_fetch_ok = False

    if contact_id:
        try:
            cart = await get_or_create_cart(
                contact_id=contact_id,
                conversation_id=conversation_id or None,
            )
            backend_fetch_ok = True
            logger.info(
                "order_summary_cart_fetched",
                thread_id=thread_id,
                item_count=len(cart.get("items", [])),
                grand_total=cart.get("grandTotal"),
            )
        except Exception as exc:
            logger.warning(
                "order_summary_cart_fetch_failed",
                thread_id=thread_id,
                error=str(exc),
            )

    # ── Fallback: use in-memory state cart when backend is empty or failed ────
    #
    # This handles the case where backend sync silently failed in sales_collect
    # (upsert_cart_item threw but was swallowed), leaving the backend with an
    # empty cart while AgentState.cart has the correct items.
    #
    if not cart.get("items"):
        state_cart: list = state.get("cart") or []
        if state_cart:
            cart = _build_cart_from_state(state_cart)
            logger.warning(
                "order_summary_state_cart_fallback",
                thread_id=thread_id,
                item_count=len(state_cart),
                grand_total=cart.get("grandTotal"),
                backend_fetch_ok=backend_fetch_ok,
                reason="backend_cart_empty",
            )
        else:
            # Both backend and state are empty — nothing to summarise.
            logger.error(
                "order_summary_no_cart_anywhere",
                thread_id=thread_id,
                backend_fetch_ok=backend_fetch_ok,
            )

    # ── Determine user intent ──────────────────────────────────────────────────
    has_new_message = bool(state["messages"]) and getattr(state["messages"][-1], "type", "") == "human"
    order_confirmed = False

    if has_new_message:
        user_text = (state["messages"][-1].content or "").strip().lower()
        if any(kw in user_text for kw in _CONFIRM_KEYWORDS):
            order_confirmed = True
            logger.info("order_confirmed_by_user", thread_id=thread_id)

    cart_summary_str = _format_cart_for_llm(cart) if cart.get("items") else "(carrito no disponible)"

    # ── Build LLM prompt ──────────────────────────────────────────────────────
    if order_confirmed:
        messages_payload = [
            {
                "role": "system",
                "content": (
                    "El cliente ha confirmado su pedido. "
                    "Agradece brevemente y dile que estás procesando su pedido. "
                    "Sé concisa."
                ),
            }
        ]
    elif has_new_message and not any(
        kw in (state["messages"][-1].content or "").strip().lower() for kw in _CONFIRM_KEYWORDS
    ):
        correction_text = state["messages"][-1].content or ""
        correction_prompt = resolve_prompt(config, "ORDER_CORRECTION", _CORRECTION_SYSTEM_PROMPT)
        messages_payload = [
            {
                "role": "system",
                "content": correction_prompt.format(
                    correction=correction_text,
                    cart_summary=cart_summary_str,
                    language_rule=lang_rule,
                ),
            }
        ]
    else:
        summary_prompt = resolve_prompt(config, "ORDER_SUMMARY", _SUMMARY_SYSTEM_PROMPT)
        messages_payload = [
            {
                "role": "system",
                "content": summary_prompt.format(language_rule=lang_rule),
            },
            {
                "role": "user",
                "content": f"Carrito actual:\n\n{cart_summary_str}",
            },
        ]

    gen = trace.generation(
        name="order_summary_llm",
        model=model,
        input={"messages": messages_payload},
    )

    stream = provider.stream_chat(
        model=model,
        messages=messages_payload,
    )

    full_response = ""
    async for chunk in stream:
        write({"type": "token", "content": chunk})
        full_response += chunk

    gen.end(output=full_response)

    return {
        "messages": [AIMessage(content=full_response)],
        "current_cart": cart,
        "order_confirmed": order_confirmed,
    }