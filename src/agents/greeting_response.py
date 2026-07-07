"""Greeting node — LLM-rendered from the tenant's {AGENT_TYPE}_GREETING prompt.

Routed when triage classifies `greeting` AND the contact's name is already
known. The system prompt is resolved from the backend prompt catalog
(tenant override → seeded default → hardcoded fallback below) and rendered
with one LLM call. Any LLM failure falls back to the original deterministic
templates so the customer always gets a greeting.

The reply is emitted as a single `token` stream event (the NestJS backend
builds the outbound WhatsApp message exclusively from token events) plus an
AIMessage so conversation history stays consistent.

Shared by every agent graph: the persona name comes from the conversation
snapshot (`agent_persona_name`) with a per-agent default, and role framing
comes from a per-code-name lookup so each agent greets in its own domain
(no store language for a school agent).
"""

from __future__ import annotations

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..observability import record_node_invocation
from ..providers.registry import get_provider, resolve_model
from ..usage import make_usage_record
from .utils import (
    language_instruction,
    latest_user_text,
    resolve_persona,
    resolve_prompt,
)

logger = structlog.get_logger(__name__)

# Hardcoded fallback for the {AGENT_TYPE}_GREETING prompt — mirrors the seeded
# default in verveux-backend/src/ai-prompts/ai-prompts.constants.ts. Only used
# when the request carries no prompt payload for the key at all.
_GREETING_SYSTEM_PROMPT = """Eres {persona}, {role}. Atiendes por WhatsApp y el usuario —un contacto ya conocido— acaba de saludarte.

Tu única tarea: devolver UN saludo breve y cálido.

Reglas:
- Saluda al usuario por su nombre ({name}) si está disponible.
- Preséntate como {persona} y menciona en una frase en qué puedes ayudar.
- Máximo 2 líneas. Puedes usar un emoji.
- NO pidas datos ni inicies ningún flujo; solo saluda y ofrece ayuda.
- {language_rule}"""


class _SafeDict(dict):
    """format_map helper: unknown placeholders stay literal instead of raising."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _render_prompt(template: str, values: dict) -> str:
    """Substitute single-brace placeholders, tolerating tenant-authored templates.

    Tenants edit these templates freely; a stray `{` or an unknown token must
    degrade to literal text, never a 500.
    """
    try:
        return template.format_map(_SafeDict(values))
    except (ValueError, IndexError):
        # Malformed braces (e.g. a lone "{") — use the template as-is.
        return template

_TEMPLATES = {
    "es": "¡Hola{name_part}! 👋 Soy {persona}, {role}. ¿En qué puedo ayudarte hoy?",
    "en": "Hi{name_part}! 👋 I'm {persona}, {role}. How can I help you today?",
    "pt": "Olá{name_part}! 👋 Sou {persona}, {role}. Como posso te ajudar hoje?",
}

# agent_code_name → (default persona, role phrase per language)
_AGENT_PROFILES: dict[str, tuple[str, dict[str, str]]] = {
    "camila": (
        "Camila",
        {
            "es": "la secretaria académica",
            "en": "the academic secretary",
            "pt": "a secretária acadêmica",
        },
    ),
    "sofia": (
        "Sofía",
        {
            "es": "la asistente académica",
            "en": "the academic assistant",
            "pt": "a assistente acadêmica",
        },
    ),
    "helena": (
        "Helena",
        {
            "es": "tu asesora de ventas",
            "en": "your sales assistant",
            "pt": "sua assessora de vendas",
        },
    ),
    "giulia": (
        "Giulia",
        {
            "es": "tu asistente del restaurante",
            "en": "your restaurant assistant",
            "pt": "sua assistente do restaurante",
        },
    ),
    "marco": (
        "Marco",
        {
            "es": "tu asistente de citas",
            "en": "your appointments assistant",
            "pt": "seu assistente de agendamento",
        },
    ),
}

_GENERIC_ROLES = {
    "es": "tu asistente virtual",
    "en": "your virtual assistant",
    "pt": "seu assistente virtual",
}

# agent_type fallback for env-registered extra code names
# (APPOINTMENTS_EXTRA_CODE_NAMES); persona then falls back to code_name.title().
_TYPE_ROLE_FALLBACK: dict[str, dict[str, str]] = {
    "sales": _AGENT_PROFILES["helena"][1],
    "school": _AGENT_PROFILES["sofia"][1],
    "restaurant": _AGENT_PROFILES["giulia"][1],
    "appointments": _AGENT_PROFILES["marco"][1],
}


def has_contact_name(state: AgentState) -> bool:
    """True when the backend snapshot carries a non-empty contact name.

    Generic counterpart of camila's `_has_name` (camila keeps its own, which
    additionally honours the camila-only `school_name_captured` latch).
    """
    ctx = state.get("user_context") or {}
    if not isinstance(ctx, dict):
        return False
    return bool((ctx.get("name") or "").strip())


async def greeting_response_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("greeting_response")

    code_name = (state.get("agent_code_name") or "").strip().lower()
    agent_type = (state.get("agent_type") or "").strip().lower()

    lang = (state.get("language") or "es").strip().lower()[:2]
    if lang not in _TEMPLATES:
        lang = "es"

    default_persona, roles = _AGENT_PROFILES.get(code_name, ("", {}))
    if not roles:
        roles = _TYPE_ROLE_FALLBACK.get(agent_type, _GENERIC_ROLES)
    if not default_persona:
        default_persona = code_name.title() if code_name else "Asistente"

    persona = resolve_persona(state, default_persona)
    role = roles.get(lang) or _GENERIC_ROLES[lang]

    # Personalize when the name is on file; camila can reach this node with
    # only the `school_name_captured` latch set (persist failure), so an
    # empty name must render cleanly.
    ctx = state.get("user_context") or {}
    name = (ctx.get("name") or "").strip() if isinstance(ctx, dict) else ""
    name_part = f", {name}" if name else ""

    # ── LLM-rendered greeting from the tenant's {AGENT_TYPE}_GREETING prompt ──
    # {persona} is substituted here (not by resolve_prompt) because this node
    # resolves richer per-code-name defaults than the helper's generic one.
    prompt_key = f"{(agent_type or 'sales').upper()}_GREETING"
    template = resolve_prompt(config, prompt_key, _GREETING_SYSTEM_PROMPT)
    system_content = _render_prompt(
        template,
        {
            "persona": persona,
            "role": role,
            "name": name,
            "language_rule": language_instruction(lang),
        },
    )

    text = ""
    turn_usage: list = []
    try:
        provider = get_provider(config)
        model = resolve_model(config)
        messages_payload = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": latest_user_text(state) or "Hola"},
        ]
        # Buffer the full response instead of streaming chunks: if the call
        # died mid-stream we'd have leaked partial tokens ahead of the
        # deterministic fallback, and the backend concatenates every token
        # event into one WhatsApp message.
        async for chunk in provider.stream_chat(model=model, messages=messages_payload):
            text += chunk
        text = text.strip()
        if text:
            turn_usage.append(
                make_usage_record(
                    node="greeting_response", provider=provider, model=model
                )
            )
    except Exception as exc:
        logger.warning(
            "greeting_llm_failed",
            thread_id=state.get("thread_id", "unknown"),
            error=str(exc),
        )
        text = ""

    llm_rendered = bool(text)
    if not text:
        # Deterministic fallback — the customer always gets a greeting even
        # when the LLM call fails or returns nothing.
        text = _TEMPLATES[lang].format(
            name_part=name_part, persona=persona, role=role
        )

    # Emit the reply as a single SSE token so the backend (which accumulates
    # `type: token` events to build the WhatsApp message) actually has text
    # to send. Without this the conversation goes silent.
    write = get_stream_writer()
    write({"type": "token", "content": text})

    logger.info(
        "greeting_response_sent",
        thread_id=state.get("thread_id", "unknown"),
        agent_code_name=code_name or "unknown",
        language=lang,
        persona=persona,
        llm_rendered=llm_rendered,
    )

    return {
        "messages": [AIMessage(content=text)],
        "turn_usage": turn_usage,
    }
