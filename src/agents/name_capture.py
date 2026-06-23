"""Camila name_capture node — asks for, parses, and persists the user's name.

First turn behavior (no name in user_context, school_name_captured=False):
- if the user's latest message clearly contains a name, parse it, save to
  the backend, latch school_name_captured, and emit a personalized greeting
- otherwise, ask the user for their full name and end the turn

On subsequent turns this node is skipped by the camila routing edges.
"""

from __future__ import annotations

import json

import structlog
from httpx import HTTPStatusError, RequestError
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..observability import record_node_invocation
from ..providers.registry import get_provider, resolve_model
from ..usage import make_usage_record
from . import backend_client
from .triage import _strip_json_fences
from .utils import language_instruction

logger = structlog.get_logger(__name__)

_NAME_EXTRACT_PROMPT = """Eres una extractora de nombres para una conversación de WhatsApp con la secretaría de una institución educativa.

Tu tarea: leer el último mensaje del usuario y devolver SOLO un objeto JSON.

Esquema JSON: {{"found": true|false, "name": "nombre y apellidos como los proporcionó el usuario"}}

Reglas:
- Si el usuario presentó claramente un nombre completo (ej. "Soy Patricia Hernández", "Me llamo Camilo Pérez", "Patricia Guevara Ochoa"), devuelve {{"found": true, "name": "..."}}.
- Si el mensaje es un saludo, una pregunta, o no incluye nombre, devuelve {{"found": false, "name": ""}}.
- No inventes nombres. No completes apellidos faltantes.
- Responde SOLO con el JSON en una línea — sin markdown.
"""

_NAME_ASK_TEMPLATE = (
    "¡Hola! Soy Camila, la secretaria virtual. Para atenderte mejor, "
    "¿podrías decirme tu nombre completo, por favor?"
)


def _latest_user_text(state: AgentState) -> str:
    for msg in reversed(state.get("messages") or []):
        if getattr(msg, "type", "") == "human":
            return (msg.content or "") if hasattr(msg, "content") else ""
    return ""


async def name_capture_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("name_capture")

    thread_id = state.get("thread_id", "unknown")
    user_text = _latest_user_text(state)

    provider = get_provider(config)
    model = resolve_model(config)
    lang_rule = language_instruction(state.get("language", "es"))

    messages_payload = [
        {"role": "system", "content": _NAME_EXTRACT_PROMPT + "\n" + lang_rule},
        {"role": "user", "content": user_text},
    ]

    stream = provider.stream_chat(model=model, messages=messages_payload)
    raw = ""
    async for chunk in stream:
        raw += chunk

    usage_record = make_usage_record(
        node="name_capture", provider=provider, model=model
    )

    found = False
    name = ""
    try:
        parsed = json.loads(_strip_json_fences(raw))
        found = bool(parsed.get("found", False))
        name = str(parsed.get("name", "")).strip()
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "name_capture_parse_failed", thread_id=thread_id, error=str(exc), raw=raw
        )

    write = get_stream_writer()

    if not found or not name:
        write({"type": "token", "content": _NAME_ASK_TEMPLATE})
        logger.info("name_capture_asked", thread_id=thread_id)
        return {
            "messages": [AIMessage(content=_NAME_ASK_TEMPLATE)],
            "turn_usage": [usage_record],
        }

    # Persist to backend (non-fatal on failure; we still personalize state)
    tenant_id = state.get("tenant_id", "")
    contact_id = state.get("contact_id", "")
    if tenant_id and contact_id:
        try:
            await backend_client.update_contact_name(
                contact_id=contact_id, name=name, tenant_id=tenant_id
            )
        except (HTTPStatusError, RequestError) as exc:
            logger.error(
                "name_capture_backend_failed",
                thread_id=thread_id,
                contact_id=contact_id,
                error=str(exc),
            )

    greeting = (
        f"¡Mucho gusto, {name}! ¿En qué puedo ayudarte hoy?"
    )
    write({"type": "token", "content": greeting})

    updated_context = dict(state.get("user_context") or {})
    updated_context["name"] = name

    logger.info(
        "name_capture_persisted",
        thread_id=thread_id,
        contact_id=contact_id,
        name_length=len(name),
    )

    return {
        "messages": [AIMessage(content=greeting)],
        "user_context": updated_context,
        "school_name_captured": True,
        "turn_usage": [usage_record],
    }
