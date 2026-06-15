import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from ..usage import make_usage_record
from .utils import format_user_context, language_instruction, resolve_prompt

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
- {language_rule}
- Si el usuario saluda, saluda de vuelta y ofrece ayuda.
- Si la pregunta es sobre algo que no manejas, guía al usuario hacia las capacidades disponibles.
"""

_FAQ_KNOWLEDGE_BLOCK = """
Preguntas frecuentes de la tienda (PRIORIZA estas respuestas si el usuario pregunta algo similar):
{faq_lines}

INSTRUCCIÓN IMPORTANTE: Si la pregunta del usuario coincide con alguna de las preguntas frecuentes anteriores, usa exactamente la respuesta provista. No inventes información adicional.
"""


def _format_faqs_for_prompt(faqs: list) -> str:
    """Format FAQ list as a structured Q&A block for the LLM system prompt."""
    if not faqs:
        return ""
    lines = []
    # Sort by priority descending so most important FAQs appear first
    sorted_faqs = sorted(faqs, key=lambda f: f.get("priority", 0), reverse=True)
    for faq in sorted_faqs:
        q = faq.get("question", "").strip()
        a = faq.get("answer", "").strip()
        cat = faq.get("category", "")
        if q and a:
            cat_prefix = f"[{cat}] " if cat else ""
            lines.append(f"P: {cat_prefix}{q}\nR: {a}")
    if not lines:
        return ""
    return _FAQ_KNOWLEDGE_BLOCK.format(faq_lines="\n\n".join(lines))


async def faq_response_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("faq_response")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="faq_response",
        metadata={"thread_id": thread_id, "node": "faq_response"},
    )

    lang_rule = language_instruction(state.get("language", "en"))
    agent_type = (state.get("agent_type") or "sales").upper()
    faq_key = f"{agent_type}_FAQ"
    faq_prompt = resolve_prompt(config, faq_key, _FAQ_SYSTEM_PROMPT)

    # ── Inject FAQ knowledge from per-request FAQs ────────────────────────────
    faqs: list = state.get("faqs") or []
    faq_knowledge_block = _format_faqs_for_prompt(faqs)

    # ── Inject product catalog so the LLM can answer product-specific questions
    catalog = state.get("product_catalog") or []
    catalog_block = ""
    if catalog:
        lines = ["\n\nCatálogo de productos disponibles:"]
        for p in catalog:
            line = (
                f"- **{p.get('name', 'N/A')}**: {p.get('description', '')} "
                f"— ${p.get('price', 'N/A')} (stock: {p.get('stock', 'N/A')})"
            )
            lines.append(line)
        catalog_block = "\n".join(lines)

    system_content = (
        faq_prompt.format(language_rule=lang_rule)
        + faq_knowledge_block
        + catalog_block
        + format_user_context(state)
    )

    messages_payload = [{"role": "system", "content": system_content}]
    for msg in state["messages"]:
        if hasattr(msg, "type"):
            role = "assistant" if msg.type == "ai" else "user"
        else:
            role = "user"
        content = msg.content if hasattr(msg, "content") else str(msg)
        messages_payload.append({"role": role, "content": content})

    gen = trace.generation(
        name="faq_llm",
        model=model,
        input={"messages": messages_payload},
    )

    write = get_stream_writer()

    stream = provider.stream_chat(
        model=model,
        messages=messages_payload,
    )

    full_response = ""
    async for chunk in stream:
        write({"type": "token", "content": chunk})
        full_response += chunk

    gen.end(output=full_response)
    logger.info(
        "faq_response_sent",
        thread_id=thread_id,
        intent=state.get("intent", "faq"),
        faqs_injected=len(faqs),
    )

    return {
        "messages": [AIMessage(content=full_response)],
        "turn_usage": [
            make_usage_record(node="faq_response", provider=provider, model=model)
        ],
    }