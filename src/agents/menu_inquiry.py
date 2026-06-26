"""Restaurant menu inquiry node — answers questions about menu items."""

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

_MENU_INQUIRY_SYSTEM_PROMPT = """Eres Helena, una asistente de restaurante por WhatsApp.

El usuario tiene una pregunta sobre el menú, platillos, ingredientes, precios o disponibilidad.
Responde de forma clara y apetitosa usando el catálogo de productos proporcionado.

{language_rule}

Reglas:
- Usa la información del catálogo/menú proporcionado para responder.
- Si preguntan por ingredientes o alérgenos que no conoces, indica que pueden consultar con el restaurante.
- Puedes sugerir platillos populares o complementos si es apropiado.
- Sé concisa y amigable — es WhatsApp.
- Si el usuario quiere ordenar, indícale que puedes tomar su pedido."""


def _format_menu(state: AgentState) -> str:
    """Build a menu block from product catalog."""
    catalog = state.get("product_catalog") or []
    if not catalog:
        return ""

    items = []
    for item in catalog[:30]:
        line = f"- {item.get('name', 'Item')}"
        if item.get("price"):
            line += f" — ${item['price']}"
        if item.get("description"):
            line += f" ({item['description']})"
        items.append(line)

    return "\n\nMenú disponible:\n" + "\n".join(items)


async def menu_inquiry_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("menu_inquiry")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="menu_inquiry_node",
        metadata={"thread_id": thread_id, "node": "menu_inquiry"},
    )

    lang_rule = language_instruction(state.get("language", "en"))
    prompt = resolve_prompt(config, "MENU_INQUIRY", _MENU_INQUIRY_SYSTEM_PROMPT, state)
    system_content = prompt.format(language_rule=lang_rule)
    system_content += _format_menu(state)
    system_content += format_user_context(state)

    faqs = state.get("faqs") or []
    if faqs:
        faq_lines = [f"- P: {f['question']}\n  R: {f['answer']}" for f in faqs[:5]]
        system_content += "\n\nPreguntas frecuentes:\n" + "\n".join(faq_lines)

    history = [
        {"role": "user" if getattr(m, "type", "") == "human" else "assistant", "content": m.content}
        for m in state["messages"][-6:]
        if hasattr(m, "content") and m.content
    ]
    messages_payload = [{"role": "system", "content": system_content}] + history

    generation = trace.generation(name="menu_inquiry_llm", model=model, input={"messages": messages_payload})
    write = get_stream_writer()
    full_response = ""
    async for chunk in provider.stream_chat(model=model, messages=messages_payload):
        write({"type": "token", "content": chunk})
        full_response += chunk
    generation.end(output=full_response)

    return {
        "messages": [AIMessage(content=full_response)],
        "turn_usage": [
            make_usage_record(node="menu_inquiry", provider=provider, model=model)
        ],
    }
