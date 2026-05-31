"""School admissions collection node — two-stage extraction + conversation."""

import json

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from .utils import language_instruction, resolve_prompt, format_user_context

logger = structlog.get_logger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """Eres un asistente de extracción de datos de admisiones escolares.

Extrae del mensaje del usuario la siguiente información:
- student_name: Nombre completo del estudiante
- student_age: Edad o fecha de nacimiento
- grade_level: Grado o nivel al que aplica
- guardian_name: Nombre del padre o tutor
- contact_phone: Teléfono de contacto
- contact_email: Email de contacto

Devuelve un objeto JSON con los campos encontrados. Omite campos no mencionados.
Responde SOLO con el objeto JSON — sin markdown, sin explicación."""

_CONVERSATIONAL_SYSTEM_PROMPT = """Eres Helena, una asistente de admisiones por WhatsApp para una institución educativa.
Eres amable, profesional y orientada a ayudar. {language_rule}

Tu tarea: recopilar la información necesaria para procesar una solicitud de admisión.
Necesitas:
- Nombre completo del estudiante (student_name)
- Edad o fecha de nacimiento (student_age)
- Grado o nivel al que aplica (grade_level)
- Nombre del padre/tutor (guardian_name)
- Teléfono de contacto (contact_phone)
- Email de contacto (contact_email)

Campos ya recopilados: {collected_fields}
Campos faltantes: {missing_fields}

Reglas:
- Sé amable y profesional — es una conversación por WhatsApp.
- Si ya tienes toda la información, confirma los datos y avisa que procesarás la solicitud.
- NO devuelvas JSON — esta es una respuesta en lenguaje natural."""

_REQUIRED_FIELDS = [
    "student_name", "student_age", "grade_level",
    "guardian_name", "contact_phone", "contact_email",
]


async def admissions_collect_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("admissions_collect")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="admissions_collect_node",
        metadata={"thread_id": thread_id, "node": "admissions_collect"},
    )

    admissions_data: dict = state.get("admissions_data") or {}

    # ── Stage 1: Extraction ──────────────────────────────────────────────────
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", "") == "human":
            last_user_msg = msg.content
            break

    if last_user_msg:
        extraction_prompt = resolve_prompt(
            config, "ADMISSIONS_EXTRACTION", _EXTRACTION_SYSTEM_PROMPT
        )
        extraction_messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": last_user_msg},
        ]

        gen = trace.generation(name="admissions_extraction", model=model, input={"messages": extraction_messages})
        extracted_text = ""
        async for chunk in provider.stream_chat(model=model, messages=extraction_messages):
            extracted_text += chunk
        gen.end(output=extracted_text)

        try:
            extracted = json.loads(extracted_text)
            admissions_data.update(extracted)
        except (json.JSONDecodeError, TypeError):
            pass

    # ── Stage 2: Check completion ────────────────────────────────────────────
    collected = [f for f in _REQUIRED_FIELDS if admissions_data.get(f)]
    missing = [f for f in _REQUIRED_FIELDS if not admissions_data.get(f)]
    is_complete = len(missing) == 0

    # ── Stage 3: Conversational response ─────────────────────────────────────
    lang_rule = language_instruction(state.get("language", "en"))
    conv_prompt = resolve_prompt(
        config, "ADMISSIONS_CONVERSATIONAL", _CONVERSATIONAL_SYSTEM_PROMPT
    )
    system_content = conv_prompt.format(
        language_rule=lang_rule,
        collected_fields=", ".join(f"{f}={admissions_data[f]}" for f in collected),
        missing_fields=", ".join(missing) if missing else "Ninguno — todos recopilados",
    )
    system_content += format_user_context(state)

    history = [
        {"role": "user" if getattr(m, "type", "") == "human" else "assistant", "content": m.content}
        for m in state["messages"][-6:]
        if hasattr(m, "content") and m.content
    ]
    messages_payload = [{"role": "system", "content": system_content}] + history

    gen2 = trace.generation(name="admissions_conversational", model=model, input={"messages": messages_payload})
    write = get_stream_writer()
    full_response = ""
    async for chunk in provider.stream_chat(model=model, messages=messages_payload):
        write({"type": "token", "content": chunk})
        full_response += chunk
    gen2.end(output=full_response)

    return {
        "messages": [AIMessage(content=full_response)],
        "admissions_data": admissions_data,
        "admissions_complete": is_complete,
    }
