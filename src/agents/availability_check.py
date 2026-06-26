"""Appointments availability check node — presents available slots."""

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

_AVAILABILITY_CHECK_SYSTEM_PROMPT = """Eres Helena, una asistente de citas y reservaciones por WhatsApp.

El usuario quiere consultar disponibilidad de horarios para una cita.
Tu tarea: presentar los horarios disponibles y ayudar al usuario a elegir uno.

{language_rule}

Reglas:
- Presenta los horarios disponibles de forma clara y organizada.
- Si necesitas saber el tipo de servicio o profesional preferido, pregunta primero.
- Si no tienes horarios específicos en el contexto, indica los horarios generales de atención.
- Sé concisa y amigable — es WhatsApp.
- Una vez que el usuario elija un horario, confirma que procederás con la reservación."""


def _format_availability_knowledge(state: AgentState) -> str:
    """Build knowledge block from FAQs and catalog relevant to availability."""
    parts = []

    faqs = state.get("faqs") or []
    if faqs:
        faq_lines = [f"- P: {f['question']}\n  R: {f['answer']}" for f in faqs[:8]]
        parts.append("Información disponible:\n" + "\n".join(faq_lines))

    catalog = state.get("product_catalog") or []
    if catalog:
        items = [f"- {c.get('name', '')}: {c.get('description', '')}" for c in catalog[:15]]
        parts.append("Servicios disponibles:\n" + "\n".join(items))

    return "\n\n" + "\n\n".join(parts) if parts else ""


async def availability_check_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("availability_check")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="availability_check_node",
        metadata={"thread_id": thread_id, "node": "availability_check"},
    )

    lang_rule = language_instruction(state.get("language", "en"))
    prompt = resolve_prompt(config, "AVAILABILITY_CHECK", _AVAILABILITY_CHECK_SYSTEM_PROMPT, state)
    system_content = prompt.format(language_rule=lang_rule)
    system_content += _format_availability_knowledge(state)
    system_content += format_user_context(state)

    history = [
        {"role": "user" if getattr(m, "type", "") == "human" else "assistant", "content": m.content}
        for m in state["messages"][-6:]
        if hasattr(m, "content") and m.content
    ]
    messages_payload = [{"role": "system", "content": system_content}] + history

    generation = trace.generation(name="availability_check_llm", model=model, input={"messages": messages_payload})
    write = get_stream_writer()
    full_response = ""
    async for chunk in provider.stream_chat(model=model, messages=messages_payload):
        write({"type": "token", "content": chunk})
        full_response += chunk
    generation.end(output=full_response)

    return {
        "messages": [AIMessage(content=full_response)],
        "turn_usage": [
            make_usage_record(node="availability_check", provider=provider, model=model)
        ],
    }
