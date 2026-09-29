"""Lead qualification node (veronica) — two-stage extraction + conversation.

Collects the visitor's contact details and interest. Mandatory fields:
fullName, email, serviceInterest (AI_AGENTS | CRM | ALL). company, phone,
challenge and comments are opportunistic. When the mandatory set completes,
the node mints a ``lead_submission_id`` (once — replays reuse it) and the
graph auto-chains into ``execute_lead`` in the same turn.
"""

import json
import re
from uuid import uuid4

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..json_utils import strip_json_fences
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from ..usage import make_usage_record
from .utils import (
    format_user_context,
    language_instruction,
    latest_user_text,
    resolve_prompt,
)

logger = structlog.get_logger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """Eres un asistente de extracción de datos de contacto de leads.

Extrae del mensaje del visitante la siguiente información:
- fullName: Nombre completo del visitante
- email: Correo electrónico
- phoneCountryCode: Código de país del teléfono (ej. "+57")
- phoneNumber: Número de teléfono
- company: Nombre de la empresa
- serviceInterest: Interés de servicio — EXACTAMENTE uno de: "AI_AGENTS" (agentes de IA, chatbots, automatización de atención), "CRM" (gestión de clientes, ventas, pipeline), "ALL" (ambos o toda la plataforma)
- challenge: Reto o necesidad principal que describe
- comments: Cualquier otra información relevante (horarios de contacto, contexto)

Devuelve un objeto JSON con los campos encontrados. Omite campos no mencionados.
NO inventes datos que el visitante no haya dicho.
Responde SOLO con el objeto JSON — sin markdown, sin explicación."""

_CONVERSATIONAL_SYSTEM_PROMPT = """Eres Verónica, una asistente virtual del sitio web de la empresa. Tu objetivo es calificar al visitante como lead: conocer su interés y obtener sus datos de contacto.
Eres amable, profesional y nunca insistente. {language_rule}

Datos que necesitas (obligatorios):
- Nombre completo (fullName)
- Correo electrónico (email)
- Interés de servicio (serviceInterest): agentes de IA, CRM, o ambos

Campos ya recopilados: {collected_fields}
Campos faltantes: {missing_fields}

Reglas:
- Pide UN dato faltante a la vez, de forma natural dentro de la conversación.
- Si el visitante hace una pregunta, respóndela primero y luego retoma la calificación.
- Si ya tienes todos los datos obligatorios, agradece y confirma que un asesor lo contactará pronto.
- Nunca inventes datos ni presiones al visitante si prefiere no compartir algo.
- Sé concisa — es un chat web. Máximo 3 líneas por respuesta."""

_MANDATORY_FIELDS = ["fullName", "email", "serviceInterest"]
_OPTIONAL_FIELDS = [
    "company",
    "phoneCountryCode",
    "phoneNumber",
    "challenge",
    "comments",
]
_SERVICE_INTERESTS = {"AI_AGENTS", "CRM", "ALL"}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _sanitize(lead_data: dict) -> dict:
    """Drop invalid values so a bad extraction never satisfies completion."""
    clean = {}
    for key in _MANDATORY_FIELDS + _OPTIONAL_FIELDS:
        value = lead_data.get(key)
        if not isinstance(value, str):
            continue
        value = value.strip()
        if not value:
            continue
        if key == "email" and not _EMAIL_RE.match(value):
            continue
        if key == "serviceInterest":
            value = value.upper()
            if value not in _SERVICE_INTERESTS:
                continue
        clean[key] = value
    return clean


async def lead_collect_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("lead_collect")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="lead_collect_node",
        metadata={"thread_id": thread_id, "node": "lead_collect"},
    )

    lead_data: dict = dict(state.get("lead_data") or {})
    turn_usage: list = []

    # The name_capture flow (or the backend Contact snapshot) may already
    # know the visitor's name — never re-ask what the platform knows.
    ctx = state.get("user_context") or {}
    if not lead_data.get("fullName") and isinstance(ctx, dict):
        known_name = (ctx.get("name") or "").strip()
        if known_name:
            lead_data["fullName"] = known_name
    if not lead_data.get("email") and isinstance(ctx, dict):
        known_email = (ctx.get("email") or "").strip()
        if known_email:
            lead_data["email"] = known_email

    # ── Stage 1: extraction over the whole trailing burst ────────────────────
    last_user_msg = latest_user_text(state)
    if last_user_msg:
        extraction_messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": last_user_msg},
        ]
        gen = trace.generation(
            name="lead_extraction",
            model=model,
            input={"messages": extraction_messages},
        )
        extracted_text = ""
        async for chunk in provider.stream_chat(
            model=model, messages=extraction_messages
        ):
            extracted_text += chunk
        turn_usage.append(
            make_usage_record(
                node="lead_collect.extraction", provider=provider, model=model
            )
        )
        gen.end(output=extracted_text)

        try:
            extracted = json.loads(strip_json_fences(extracted_text))
            if isinstance(extracted, dict):
                lead_data.update(_sanitize(extracted))
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "lead_collect_extraction_parse_failed", raw=extracted_text[:200]
            )

    lead_data = _sanitize(lead_data)

    # ── Stage 2: completion check + one-time submission id ──────────────────
    collected = [f for f in _MANDATORY_FIELDS if lead_data.get(f)]
    missing = [f for f in _MANDATORY_FIELDS if not lead_data.get(f)]
    is_complete = len(missing) == 0

    # Mint exactly once: a replayed/coalesced turn keeps the original id so
    # the backend's unique index dedupes any double submission.
    submission_id = state.get("lead_submission_id")
    if is_complete and not submission_id:
        submission_id = str(uuid4())

    # ── Stage 3: conversational reply ────────────────────────────────────────
    lang_rule = language_instruction(state.get("language", "es"))
    conv_prompt = resolve_prompt(
        config, "LEADS_CONVERSATIONAL", _CONVERSATIONAL_SYSTEM_PROMPT, state
    )
    optional_collected = [f for f in _OPTIONAL_FIELDS if lead_data.get(f)]
    system_content = conv_prompt.format(
        language_rule=lang_rule,
        collected_fields=", ".join(
            f"{f}={lead_data[f]}" for f in collected + optional_collected
        )
        or "Ninguno",
        missing_fields=", ".join(missing)
        if missing
        else "Ninguno — todos recopilados",
    )
    system_content += format_user_context(state)

    history = [
        {
            "role": "user" if getattr(m, "type", "") == "human" else "assistant",
            "content": m.content,
        }
        for m in state["messages"][-6:]
        if hasattr(m, "content") and m.content
    ]
    messages_payload = [{"role": "system", "content": system_content}] + history

    gen2 = trace.generation(
        name="lead_conversational",
        model=model,
        input={"messages": messages_payload},
    )
    write = get_stream_writer()
    full_response = ""
    async for chunk in provider.stream_chat(model=model, messages=messages_payload):
        write({"type": "token", "content": chunk})
        full_response += chunk
    turn_usage.append(
        make_usage_record(node="lead_collect.reply", provider=provider, model=model)
    )
    gen2.end(output=full_response)

    logger.info(
        "lead_collect_progress",
        thread_id=thread_id,
        collected=collected,
        missing=missing,
        complete=is_complete,
    )

    return {
        "messages": [AIMessage(content=full_response)],
        "lead_data": lead_data,
        "lead_collection_complete": is_complete,
        "lead_submission_id": submission_id,
        "turn_usage": turn_usage,
    }
