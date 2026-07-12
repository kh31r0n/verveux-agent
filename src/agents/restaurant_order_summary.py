"""Restaurant order summary node — presents the order and asks for confirmation.

Renders from the synced backend cart (source of truth for totals) with a
state-cart fallback, mirroring the sales order_summary. Latches
`restaurant_phase = "confirmation"` so the customer's reply resumes at
restaurant_confirm on the next turn.
"""

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from ..usage import make_usage_record
from .backend_client import get_or_create_cart
from .order_summary import _build_cart_from_state, _format_cart_for_llm
from .utils import (
    format_user_context,
    language_instruction,
    resolve_persona,
    resolve_prompt,
)

logger = structlog.get_logger(__name__)

_SUMMARY_SYSTEM_PROMPT = """Eres {persona}, una asistente de pedidos por WhatsApp para un restaurante.
{language_rule}

Has recopilado toda la información del pedido del cliente.
Genera un resumen claro y apetitoso del pedido con:

- **Platillos** (nombre, cantidad, notas especiales)
- **Tipo de servicio** (para llevar o a domicilio)
- **Dirección** (si es a domicilio)
- **Total**

Después del resumen, pregunta EXACTAMENTE:
"¿Tu pedido está correcto? Responde **confirmar** para enviarlo, o dime qué cambiar."

Sé breve — es WhatsApp. No inventes platillos ni precios."""


def _format_order_details(order_data: dict, cart: dict) -> str:
    """Order-level fields + per-item notes appended to the LLM context."""
    lines = [_format_cart_for_llm(cart) if cart.get("items") else "(carrito no disponible)"]

    notes_lines = []
    for item in cart.get("items", []):
        # Backend cart items expose `attributes.agentNotes`; state-cart items
        # built by _build_cart_from_state carry no notes (they live on the
        # state cart list itself, already synced to the backend).
        attrs = item.get("attributes") or {}
        if isinstance(attrs, dict) and attrs.get("agentNotes"):
            notes_lines.append(f"  - {item.get('productName', '')}: {attrs['agentNotes']}")
    if notes_lines:
        lines.append("Notas de preparación:")
        lines.extend(notes_lines)

    service_type = order_data.get("service_type")
    if service_type:
        label = "a domicilio" if service_type == "delivery" else "para llevar"
        lines.append(f"Tipo de servicio: {label}")
    if service_type == "delivery" and order_data.get("delivery_address"):
        lines.append(f"Dirección: {order_data['delivery_address']}")
    if order_data.get("special_notes"):
        lines.append(f"Notas del pedido: {order_data['special_notes']}")

    return "\n".join(lines)


async def restaurant_order_summary_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("restaurant_order_summary")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")
    contact_id: str = state.get("contact_id", "")
    conversation_id: str = state.get("conversation_id", "")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="restaurant_order_summary_node",
        metadata={"thread_id": thread_id, "node": "restaurant_order_summary"},
    )

    order_data = state.get("restaurant_order_data") or {}

    # ── Fetch backend cart (source of truth for totals) ───────────────────────
    cart: dict = {}
    if contact_id and not state.get("backend_cart_sync_failed"):
        try:
            cart = await get_or_create_cart(
                contact_id=contact_id,
                conversation_id=conversation_id or None,
            )
        except Exception as exc:
            logger.warning(
                "restaurant_summary_cart_fetch_failed",
                thread_id=thread_id,
                error=str(exc),
            )

    # Fallback: the in-memory state cart when backend fetch failed or is empty.
    if not cart.get("items"):
        state_cart: list = state.get("cart") or []
        if state_cart:
            cart = _build_cart_from_state(state_cart)
            logger.warning(
                "restaurant_summary_state_cart_fallback",
                thread_id=thread_id,
                item_count=len(state_cart),
            )
        else:
            logger.error("restaurant_summary_no_cart_anywhere", thread_id=thread_id)

    lang_rule = language_instruction(state.get("language", "en"))
    prompt = resolve_prompt(config, "RESTAURANT_ORDER_SUMMARY", _SUMMARY_SYSTEM_PROMPT, state)
    system_content = prompt.format(
        persona=resolve_persona(state, "Giulia"),
        language_rule=lang_rule,
    )
    system_content += f"\n\nDatos del pedido recopilados:\n{_format_order_details(order_data, cart)}"
    system_content += format_user_context(state)

    history = [
        {"role": "user" if getattr(m, "type", "") == "human" else "assistant", "content": m.content}
        for m in state["messages"][-6:]
        if hasattr(m, "content") and m.content
    ]
    messages_payload = [{"role": "system", "content": system_content}] + history

    generation = trace.generation(name="restaurant_order_summary_llm", model=model, input={"messages": messages_payload})
    write = get_stream_writer()
    full_response = ""
    async for chunk in provider.stream_chat(model=model, messages=messages_payload):
        write({"type": "token", "content": chunk})
        full_response += chunk
    generation.end(output=full_response)

    return {
        "messages": [AIMessage(content=full_response)],
        # Latch: the next turn's "confirmar"/"cámbialo" resumes at restaurant_confirm.
        "restaurant_phase": "confirmation",
        "turn_usage": [
            make_usage_record(
                node="restaurant_order_summary", provider=provider, model=model,
            )
        ],
    }
