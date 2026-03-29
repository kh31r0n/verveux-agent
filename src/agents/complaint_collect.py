import json

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation
from .utils import format_user_context

logger = structlog.get_logger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """Eres un asistente de extracción de datos de quejas/reclamos.

Extrae del mensaje del usuario la siguiente información:
- order_ref: Número de pedido o referencia relacionada
- issue_description: Descripción del problema o queja
- desired_resolution: Lo que el cliente espera como solución (reembolso, reemplazo, etc.)

Devuelve un objeto JSON con los campos encontrados. Omite campos no mencionados.
Responde SOLO con el objeto JSON — sin markdown, sin explicación.
"""

_CONVERSATIONAL_SYSTEM_PROMPT = """Eres Helena, una asistente de atención al cliente por WhatsApp.
El cliente tiene una queja o reclamo.

Tu tarea: recopilar la información necesaria para procesar la queja de forma empática y profesional.
Necesitas:
- Referencia del pedido (order_ref) — número de pedido u orden
- Descripción del problema (issue_description) — qué pasó
- Resolución deseada (desired_resolution) — qué espera el cliente

Campos ya recopilados: {collected_fields}
Campos faltantes: {missing_fields}

Reglas:
- Sé empática y comprensiva — el cliente tiene un problema.
- Si ya tienes toda la información, confirma y dile que vas a procesar su queja.
- Sé concisa — es WhatsApp.
- Responde en español.
- NO devuelvas JSON.
"""


async def complaint_collect_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("complaint_collect")

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id: str = state.get("thread_id", "unknown")
    complaint_data: dict = dict(state.get("complaint_data") or {})

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="complaint_collect",
        metadata={"thread_id": thread_id, "node": "complaint_collect"},
    )

    write = get_stream_writer()
    write({"type": "step_progress", "step": 1, "total_steps": 1, "topic": "Registro de queja"})

    has_new_message = bool(state["messages"]) and getattr(state["messages"][-1], "type", "") == "human"

    if has_new_message:
        extraction_messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": state["messages"][-1].content},
        ]

        extraction_gen = trace.generation(
            name="complaint_extraction_llm",
            model="gpt-5",
            input={"messages": extraction_messages},
        )

        extraction_stream = await client.chat.completions.create(
            model="gpt-5",
            messages=extraction_messages,
            stream=True,
        )

        extraction_raw = ""
        async for chunk in extraction_stream:
            delta = chunk.choices[0].delta.content if chunk.choices else ""
            if delta:
                extraction_raw += delta

        extraction_gen.end(output=extraction_raw)

        try:
            extracted = json.loads(extraction_raw.strip())
            if isinstance(extracted, dict):
                complaint_data.update({k: v for k, v in extracted.items() if v})
        except (json.JSONDecodeError, ValueError):
            logger.warning("complaint_extraction_failed", raw=extraction_raw[:200])

    # Check completeness — need at least issue_description + one of order_ref or desired_resolution
    required_fields = ["order_ref", "issue_description", "desired_resolution"]
    missing_fields = [f for f in required_fields if not complaint_data.get(f)]
    complaint_complete = len(missing_fields) == 0

    collected_summary = ", ".join(f"{k}={repr(v)}" for k, v in complaint_data.items()) or "ninguno"
    missing_summary = ", ".join(missing_fields) if missing_fields else "todos respondidos"
    user_ctx_str = format_user_context(state)

    if complaint_complete:
        conv_prompt = (
            "Tienes toda la información necesaria para procesar la queja del cliente. "
            "Confirma los datos empáticamente y dile que vas a registrar y procesar su queja. "
            "Sé breve y comprensiva."
        ) + user_ctx_str
        conv_messages = [{"role": "system", "content": conv_prompt}]
    else:
        conv_messages = [
            {
                "role": "system",
                "content": _CONVERSATIONAL_SYSTEM_PROMPT.format(
                    collected_fields=collected_summary,
                    missing_fields=missing_summary,
                ) + user_ctx_str,
            },
            {
                "role": "user",
                "content": (
                    state["messages"][-1].content
                    if has_new_message
                    else "Tengo un problema con mi pedido"
                ),
            },
        ]

    conv_gen = trace.generation(
        name="complaint_conversational_llm",
        model="gpt-5",
        input={"messages": conv_messages},
    )

    conv_stream = await client.chat.completions.create(
        model="gpt-5",
        messages=conv_messages,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_response = ""
    prompt_tokens = 0
    completion_tokens = 0

    async for chunk in conv_stream:
        delta = chunk.choices[0].delta.content if chunk.choices else ""
        if delta:
            write({"type": "token", "content": delta})
            full_response += delta
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    conv_gen.end(output=full_response, usage={"input": prompt_tokens, "output": completion_tokens})

    return {
        "messages": [AIMessage(content=full_response)],
        "complaint_data": complaint_data,
        "complaint_complete": complaint_complete,
    }
