"""Restaurant order summary node — generates order summary and asks for confirmation."""

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from ..usage import make_usage_record
from .utils import language_instruction, resolve_prompt, format_user_context

logger = structlog.get_logger(__name__)

_SUMMARY_SYSTEM_PROMPT = """Eres Helena, una asistente de pedidos por WhatsApp para un restaurante.

Has recopilado toda la información del pedido del cliente.
Genera un resumen claro y apetitoso del pedido con:

- **Platillos** (nombre, cantidad, notas especiales)
- **Tipo de servicio** (para llevar o a domicilio)
- **Dirección** (si es a domicilio)
- **Total estimado** (calcula con precios del catálogo si están disponibles)

Después del resumen, pregunta:
"¿Tu pedido está correcto? Responde **confirmar** para enviarlo, o dime qué cambiar."

{language_rule}"""


async def restaurant_order_summary_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("restaurant_order_summary")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="restaurant_order_summary_node",
        metadata={"thread_id": thread_id, "node": "restaurant_order_summary"},
    )

    order_data = state.get("restaurant_order_data") or {}
    catalog = state.get("product_catalog") or []

    lang_rule = language_instruction(state.get("language", "en"))
    prompt = resolve_prompt(config, "RESTAURANT_ORDER_SUMMARY", _SUMMARY_SYSTEM_PROMPT)
    system_content = prompt.format(language_rule=lang_rule)

    # Add order data context
    order_info = f"\n\nDatos del pedido recopilados:\n{_format_order_data(order_data, catalog)}"
    system_content += order_info
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
        "turn_usage": [
            make_usage_record(
                node="restaurant_order_summary", provider=provider, model=model,
            )
        ],
    }


def _format_order_data(order_data: dict, catalog: list) -> str:
    """Format order data for the LLM context."""
    lines = []

    items = order_data.get("items") or []
    if items:
        lines.append("Platillos:")
        catalog_map = {c.get("name", "").lower(): c for c in catalog}
        total = 0.0
        for item in items:
            name = item.get("name", "Unknown")
            qty = item.get("quantity", 1)
            notes = item.get("notes", "")
            line = f"  - {qty}x {name}"
            if notes:
                line += f" ({notes})"
            # Try to find price
            cat_item = catalog_map.get(name.lower())
            if cat_item and cat_item.get("price"):
                price = float(cat_item["price"]) * qty
                line += f" — ${price:.2f}"
                total += price
            lines.append(line)
        if total > 0:
            lines.append(f"  Total estimado: ${total:.2f}")

    if order_data.get("service_type"):
        lines.append(f"Tipo de servicio: {order_data['service_type']}")
    if order_data.get("delivery_address"):
        lines.append(f"Dirección: {order_data['delivery_address']}")
    if order_data.get("special_notes"):
        lines.append(f"Notas: {order_data['special_notes']}")

    return "\n".join(lines) if lines else "Sin datos de pedido"
