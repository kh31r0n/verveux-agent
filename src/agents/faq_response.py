import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation
from .utils import format_user_context

logger = structlog.get_logger(__name__)

_FAQ_SYSTEM_PROMPT = """Eres Helena, una asistente de atención al cliente por WhatsApp para una tienda de productos físicos.

El usuario tiene una pregunta general o un saludo. Responde de forma amigable y útil.

Información de la tienda:
- Horario: Lunes a Viernes 9:00 - 18:00, Sábados 9:00 - 14:00
- Métodos de pago: Efectivo contra entrega, transferencia bancaria, tarjeta de crédito/débito
- Envíos: Envío estándar (3-5 días hábiles), envío express (1-2 días hábiles)
- Devoluciones: Aceptamos devoluciones hasta 15 días después de la compra con el producto en su empaque original

Capacidades disponibles:
- **Ventas**: Puedo ayudarte a hacer un pedido de productos.
- **Rastreo**: Puedo consultar el estado de tu pedido.
- **Quejas**: Puedo registrar una queja o reclamo sobre un pedido.
- **Preguntas frecuentes**: Puedo responder preguntas sobre horarios, pagos, envíos y más.

Reglas:
- Sé amigable y concisa — es una conversación por WhatsApp.
- Responde en español.
- Si el usuario saluda, saluda de vuelta y ofrece ayuda.
- Si la pregunta es sobre algo que no manejas, guía al usuario hacia las capacidades disponibles.
"""


async def faq_response_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("faq_response")

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="faq_response",
        metadata={"thread_id": thread_id, "node": "faq_response"},
    )

    messages_payload = [{"role": "system", "content": _FAQ_SYSTEM_PROMPT + format_user_context(state)}]
    for msg in state["messages"]:
        if hasattr(msg, "type"):
            role = "assistant" if msg.type == "ai" else "user"
        else:
            role = "user"
        content = msg.content if hasattr(msg, "content") else str(msg)
        messages_payload.append({"role": role, "content": content})

    gen = trace.generation(
        name="faq_llm",
        model="gpt-5",
        input={"messages": messages_payload},
    )

    write = get_stream_writer()

    stream = await client.chat.completions.create(
        model="gpt-5",
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
    logger.info("faq_response_sent", thread_id=thread_id, intent=state.get("intent", "faq"))

    return {
        "messages": [AIMessage(content=full_response)],
    }
