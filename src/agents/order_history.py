import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from .utils import language_instruction, resolve_persona
from .backend_client import get_order_history

logger = structlog.get_logger(__name__)

_HISTORY_SYSTEM_PROMPT = """Eres {persona}, una asistente de atención al cliente por WhatsApp.

El cliente quiere ver sus pedidos anteriores.
Se te proporcionará la lista de pedidos recientes del backend.

Tu tarea:
- Presenta los pedidos de forma clara y legible para WhatsApp.
- Para cada pedido muestra: fecha, estado, productos y total.
- Si no hay pedidos, díselo amablemente.
- Sé concisa — es una conversación por WhatsApp.

{language_rule}
"""


def _format_orders(orders: list) -> str:
    if not orders:
        return "No se encontraron pedidos anteriores."

    lines = []
    for i, order in enumerate(orders, 1):
        from datetime import datetime
        date_str = ""
        if order.get("checkoutAt"):
            try:
                dt = datetime.fromisoformat(order["checkoutAt"].replace("Z", "+00:00"))
                date_str = dt.strftime("%d/%m/%Y")
            except Exception:
                date_str = order["checkoutAt"][:10]

        lines.append(f"*Pedido {i}* — {date_str}")
        lines.append(f"Estado: {order.get('status', 'N/A')}")
        for item in order.get("items", []):
            lines.append(f"  • {item['productName']} x{item['quantity']} — ${item['lineTotal']:.2f}")
        lines.append(f"Total: ${order.get('grandTotal', 0):.2f} {order.get('currency', 'USD')}")
        lines.append("")

    return "\n".join(lines).strip()


async def order_history_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("order_history")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")
    contact_id: str = state.get("contact_id", "")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="order_history",
        metadata={"thread_id": thread_id, "node": "order_history"},
    )

    write = get_stream_writer()
    lang_rule = language_instruction(state.get("language", "en"))

    orders = []
    if contact_id:
        try:
            orders = await get_order_history(contact_id=contact_id, limit=5)
        except Exception as exc:
            logger.warning("order_history_fetch_failed", thread_id=thread_id, error=str(exc))

    history_text = _format_orders(orders)

    messages_payload = [
        {
            "role": "system",
            "content": _HISTORY_SYSTEM_PROMPT.format(
                persona=resolve_persona(state, "Helena"),
                language_rule=lang_rule,
            ),
        },
        {
            "role": "user",
            "content": f"Pedidos del cliente:\n\n{history_text}",
        },
    ]

    gen = trace.generation(
        name="order_history_llm",
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

    return {"messages": [AIMessage(content=full_response)]}
