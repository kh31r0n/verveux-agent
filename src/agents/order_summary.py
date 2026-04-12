import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
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


async def order_summary_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("order_summary")

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
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

    # Fetch backend cart (source of truth for totals)
    cart: dict = {}
    if contact_id:
        try:
            cart = await get_or_create_cart(
                contact_id=contact_id,
                conversation_id=conversation_id or None,
            )
        except Exception as exc:
            logger.warning("order_summary_cart_fetch_failed", thread_id=thread_id, error=str(exc))

    has_new_message = bool(state["messages"]) and getattr(state["messages"][-1], "type", "") == "human"
    order_confirmed = False

    if has_new_message:
        user_text = (state["messages"][-1].content or "").strip().lower()
        if any(kw in user_text for kw in _CONFIRM_KEYWORDS):
            order_confirmed = True
            logger.info("order_confirmed_by_user", thread_id=thread_id)

    cart_summary_str = _format_cart_for_llm(cart) if cart else "(carrito no disponible)"

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
        model="gpt-5.4-nano",
        input={"messages": messages_payload},
    )

    stream = await client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=messages_payload,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_response = ""
    prompt_tokens = 0
    completion_tokens = 0

    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else ""
        if delta:
            write({"type": "token", "content": delta})
            full_response += delta
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    gen.end(output=full_response, usage={"input": prompt_tokens, "output": completion_tokens})

    return {
        "messages": [AIMessage(content=full_response)],
        "current_cart": cart,
        "order_confirmed": order_confirmed,
    }
