import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation

logger = structlog.get_logger(__name__)

_SUMMARY_SYSTEM_PROMPT = """Eres Helena, una asistente de ventas por WhatsApp.

Has recopilado toda la información del pedido del cliente.
Genera un resumen claro y profesional del pedido con las siguientes secciones:

- **Datos del cliente** (nombre, teléfono, email)
- **Productos** (lista de items con cantidad y notas)
- **Entrega** (dirección, fecha preferida, método de pago)
- **Total estimado** (si tienes precios del catálogo, calcula el total)

Después del resumen, pregunta al cliente si desea confirmar:
"¿El pedido está correcto? Responde **confirmar** para enviarlo, o dime qué necesitas corregir."

Responde en español.
"""

_CORRECTION_SYSTEM_PROMPT = """Eres Helena, una asistente de ventas por WhatsApp.

El cliente quiere corregir algo en su pedido. Su solicitud de corrección es:
"{correction}"

Datos actuales del pedido:
{order_data}

Reconoce la corrección, confirma lo que actualizaste y presenta un resumen breve actualizado.
Luego pregunta nuevamente si desea confirmar el pedido.

Responde en español.
"""

_CONFIRM_KEYWORDS = {"confirmar", "confirm", "yes", "sí", "si", "ok", "okay", "enviar", "dale", "listo", "perfecto"}


async def order_summary_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("order_summary")

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id: str = state.get("thread_id", "unknown")
    order_data: dict = dict(state.get("order_data") or {})

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="order_summary",
        metadata={"thread_id": thread_id, "node": "order_summary"},
    )

    write = get_stream_writer()
    write({"type": "step_progress", "step": 4, "total_steps": 4, "topic": "Resumen del pedido"})

    has_new_message = bool(state["messages"]) and getattr(state["messages"][-1], "type", "") == "human"
    order_confirmed = False

    if has_new_message:
        user_text = (state["messages"][-1].content or "").strip().lower()
        if any(kw in user_text for kw in _CONFIRM_KEYWORDS):
            order_confirmed = True
            logger.info("order_confirmed_by_user", thread_id=thread_id)

    order_data_str = "\n".join(f"- **{k}**: {v}" for k, v in order_data.items())

    # Include catalog for price calculation
    catalog = state.get("product_catalog") or []
    catalog_str = ""
    if catalog:
        catalog_str = "\n\nCatálogo de precios:\n" + "\n".join(
            f"- {p.get('name', 'N/A')}: ${p.get('price', 'N/A')}"
            for p in catalog
        )

    if order_confirmed:
        intro_messages = [
            {
                "role": "system",
                "content": (
                    "El cliente ha confirmado su pedido. "
                    "Agradece brevemente y dile que estás procesando su pedido. "
                    "Sé concisa."
                ),
            }
        ]
    elif has_new_message and not any(kw in (state["messages"][-1].content or "").strip().lower() for kw in _CONFIRM_KEYWORDS):
        # User sent a correction
        correction_text = state["messages"][-1].content or ""
        intro_messages = [
            {
                "role": "system",
                "content": _CORRECTION_SYSTEM_PROMPT.format(
                    correction=correction_text,
                    order_data=order_data_str,
                ),
            }
        ]
    else:
        # First visit — generate full summary
        intro_messages = [
            {
                "role": "system",
                "content": _SUMMARY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Datos del pedido:\n\n{order_data_str}{catalog_str}",
            },
        ]

    gen = trace.generation(
        name="order_summary_llm",
        model="gpt-5.4-nano",
        input={"messages": intro_messages},
    )

    stream = await client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=intro_messages,
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
        "order_data": order_data,
        "order_confirmed": order_confirmed,
    }
