"""School schedule inquiry node — answers questions about schedules and timetables."""

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

_SCHEDULE_INQUIRY_SYSTEM_PROMPT = """Eres Helena, una asistente académica por WhatsApp para una institución educativa.

El usuario tiene una pregunta sobre horarios, calendarios o programación académica.
Responde de forma clara y útil usando la información disponible.

{language_rule}

Reglas:
- Si tienes información de horarios en el conocimiento proporcionado, úsala.
- Si no tienes horarios específicos, indica que se comunicarán con más detalles.
- Sé concisa y amigable — es WhatsApp.
- Si el usuario pregunta por disponibilidad de un curso específico, indica los horarios si los conoces."""


def _format_schedule_knowledge(state: AgentState) -> str:
    """Build knowledge block from FAQs relevant to scheduling."""
    faqs = state.get("faqs") or []
    if not faqs:
        return ""
    faq_lines = [f"- P: {f['question']}\n  R: {f['answer']}" for f in faqs[:10]]
    return "\n\nInformación disponible:\n" + "\n".join(faq_lines)


async def schedule_inquiry_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("schedule_inquiry")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="schedule_inquiry_node",
        metadata={"thread_id": thread_id, "node": "schedule_inquiry"},
    )

    lang_rule = language_instruction(state.get("language", "en"))
    prompt = resolve_prompt(config, "SCHEDULE_INQUIRY", _SCHEDULE_INQUIRY_SYSTEM_PROMPT, state)
    system_content = prompt.format(language_rule=lang_rule)
    system_content += _format_schedule_knowledge(state)
    system_content += format_user_context(state)

    history = [
        {"role": "user" if getattr(m, "type", "") == "human" else "assistant", "content": m.content}
        for m in state["messages"][-6:]
        if hasattr(m, "content") and m.content
    ]
    messages_payload = [{"role": "system", "content": system_content}] + history

    generation = trace.generation(name="schedule_inquiry_llm", model=model, input={"messages": messages_payload})
    write = get_stream_writer()
    full_response = ""
    async for chunk in provider.stream_chat(model=model, messages=messages_payload):
        write({"type": "token", "content": chunk})
        full_response += chunk
    generation.end(output=full_response)

    return {
        "messages": [AIMessage(content=full_response)],
        "turn_usage": [
            make_usage_record(node="schedule_inquiry", provider=provider, model=model)
        ],
    }
