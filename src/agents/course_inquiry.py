"""School course inquiry node — answers questions about courses and programs."""

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from .utils import language_instruction, resolve_prompt, format_user_context

logger = structlog.get_logger(__name__)

_COURSE_INQUIRY_SYSTEM_PROMPT = """Eres Helena, una asistente académica por WhatsApp para una institución educativa.

El usuario tiene una pregunta sobre cursos, programas o oferta académica.
Responde de forma clara y útil usando la información disponible.

{language_rule}

Reglas:
- Si tienes información específica en el catálogo o conocimiento proporcionado, úsala.
- Si no tienes la información exacta, indica que consultarás y que alguien se pondrá en contacto.
- Sé concisa y amigable — es WhatsApp.
- Puedes sugerir cursos relacionados si es apropiado."""


def _format_knowledge(state: AgentState) -> str:
    """Build a knowledge block from FAQs and product catalog for courses."""
    parts = []

    faqs = state.get("faqs") or []
    if faqs:
        faq_lines = [f"- P: {f['question']}\n  R: {f['answer']}" for f in faqs[:10]]
        parts.append("Preguntas frecuentes:\n" + "\n".join(faq_lines))

    catalog = state.get("product_catalog") or []
    if catalog:
        items = [f"- {c.get('name', '')}: {c.get('description', '')}" for c in catalog[:20]]
        parts.append("Catálogo de cursos/programas:\n" + "\n".join(items))

    return "\n\n" + "\n\n".join(parts) if parts else ""


async def course_inquiry_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("course_inquiry")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="course_inquiry_node",
        metadata={"thread_id": thread_id, "node": "course_inquiry"},
    )

    lang_rule = language_instruction(state.get("language", "en"))
    prompt = resolve_prompt(config, "COURSE_INQUIRY", _COURSE_INQUIRY_SYSTEM_PROMPT)
    system_content = prompt.format(language_rule=lang_rule)
    system_content += _format_knowledge(state)
    system_content += format_user_context(state)

    history = [
        {"role": "user" if getattr(m, "type", "") == "human" else "assistant", "content": m.content}
        for m in state["messages"][-6:]
        if hasattr(m, "content") and m.content
    ]
    messages_payload = [{"role": "system", "content": system_content}] + history

    generation = trace.generation(name="course_inquiry_llm", model=model, input={"messages": messages_payload})
    write = get_stream_writer()
    full_response = ""
    async for chunk in provider.stream_chat(model=model, messages=messages_payload):
        write({"type": "token", "content": chunk})
        full_response += chunk
    generation.end(output=full_response)

    return {"messages": [AIMessage(content=full_response)]}
