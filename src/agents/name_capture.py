"""Generic name_capture node — asks for, parses, and persists the user's name.

Shared by every agent graph (helena, sofia, camila, giulia, marco). Routed by
`shared_routing.should_capture_name` when the backend snapshot carries no
`user_context.name` (Contact.customName is the single source of truth).

Turn contract — the node ALWAYS returns `name_capture_reply_sent`:
- True  → the node already replied (ask / greeting / acknowledgment); the
          graph ends the turn.
- False → nothing user-facing emitted (or only a short greeting prefix); the
          graph continues to its normal intent routing in the same turn so
          the customer's actual request still gets answered.

Branches:
- **found**: persist via the backend. The API result drives state — on
  `applied: false` a human-edited (MANUAL) name won and is adopted instead;
  on HTTP failure the extracted name personalizes this turn only (no latch),
  keeping the backend snapshot authoritative for the next turn.
- **refused**: record the refusal (`nameCaptureDeferredAt`) — never invent a
  name — and either acknowledge (no other request) or fall through so the
  question in the same message is answered.
- **not found**: ask once; a second miss defers implicitly so the agent
  never nags.
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
from .greeting_response import (
    _AGENT_PROFILES,
    _GENERIC_ROLES,
    _TYPE_ROLE_FALLBACK,
    _render_prompt,
)
from ..json_utils import strip_json_fences
from .utils import (
    language_instruction,
    latest_user_text,
    resolve_persona,
    resolve_prompt,
)

logger = structlog.get_logger(__name__)

# A second fruitless ask is a nag — defer implicitly and answer the message.
MAX_NAME_CAPTURE_ATTEMPTS = 2

_NAME_EXTRACT_PROMPT = """Eres una extractora de nombres para una conversación de WhatsApp entre una empresa y un cliente.

Tu tarea: leer el último mensaje del usuario y devolver SOLO un objeto JSON.

Esquema JSON: {{"found": true|false, "name": "nombre y apellidos como los proporcionó el usuario", "refused": true|false, "has_question": true|false}}

Reglas:
- Si el usuario presentó claramente su nombre (ej. "Soy Patricia Hernández", "Me llamo Camilo Pérez", "Patricia Guevara Ochoa"), devuelve found=true con el nombre.
- Si el usuario se niega explícitamente a dar su nombre (ej. "prefiero no decirlo", "no te lo diré", "¿para qué lo necesitas?"), devuelve refused=true.
- has_question=true cuando el mensaje además contiene una pregunta o solicitud concreta (más allá de saludar o dar/negar el nombre).
- Si el mensaje es solo un saludo o no incluye nombre ni negativa, devuelve found=false y refused=false.
- No inventes nombres. No completes apellidos faltantes.
- Responde SOLO con el JSON en una línea — sin markdown.
"""

_ASK_TEMPLATES = {
    "es": "¡Hola! Soy {persona}, {role}. Para atenderte mejor, ¿podrías decirme tu nombre, por favor?",
    "en": "Hi! I'm {persona}, {role}. To assist you better, could you tell me your name, please?",
    "pt": "Olá! Sou {persona}, {role}. Para atendê-lo melhor, poderia me dizer seu nome, por favor?",
}

_GREETING_TEMPLATES = {
    "es": "¡Mucho gusto, {name}! ¿En qué puedo ayudarte hoy?",
    "en": "Nice to meet you, {name}! How can I help you today?",
    "pt": "Muito prazer, {name}! Como posso te ajudar hoje?",
}

# Found + has_question: short prefix only — the downstream node answers next
# and the backend concatenates all token events into one WhatsApp message.
_GREETING_PREFIX_TEMPLATES = {
    "es": "¡Mucho gusto, {name}! ",
    "en": "Nice to meet you, {name}! ",
    "pt": "Muito prazer, {name}! ",
}

_REFUSAL_ACK_TEMPLATES = {
    "es": "¡Sin problema! ¿En qué puedo ayudarte hoy?",
    "en": "No problem! How can I help you today?",
    "pt": "Sem problema! Como posso te ajudar hoje?",
}

# Hardcoded fallbacks for the editable NAME_CAPTURE_* prompts — mirror the
# seeded defaults in verveux-backend/src/ai-prompts/ai-prompts.constants.ts.
# Used only when the request carries no prompt payload for the key at all;
# the LLM output (or, on failure, the deterministic templates above) is what
# actually reaches the customer.
_ASK_SYSTEM_PROMPT = """Eres {persona}. Atiendes por WhatsApp y es la primera vez que hablas con este usuario; aún no conoces su nombre.

Tu única tarea: saludar brevemente y pedirle su nombre de forma amable.

Reglas:
- Preséntate como {persona} en una frase.
- Pídele su nombre para poder atenderle mejor.
- Máximo 2 líneas. Puedes usar un emoji.
- NO pidas más datos ni inicies ningún flujo; solo saluda y pide el nombre.
- {language_rule}"""

_ACK_SYSTEM_PROMPT = """Eres {persona}. El usuario acaba de decirte su nombre: {name}.

Tu única tarea: agradecer y saludar al usuario por su nombre, y ofrecer ayuda.

Reglas:
- Salúdalo por su nombre ({name}) de forma cálida.
- Ofrece tu ayuda en una frase.
- Máximo 2 líneas. Puedes usar un emoji.
- {language_rule}"""

_REFUSAL_SYSTEM_PROMPT = """Eres {persona}. El usuario prefirió no compartir su nombre.

Tu única tarea: responder con amabilidad, sin insistir, y ofrecer ayuda.

Reglas:
- No vuelvas a pedir el nombre.
- Ofrece tu ayuda en una frase.
- Máximo 2 líneas. Puedes usar un emoji.
- {language_rule}"""


async def _render_llm(
    config: RunnableConfig,
    prompt_key: str,
    fallback_system: str,
    values: dict,
    user_content: str,
    fallback_text: str,
    thread_id: str,
) -> tuple[str, list]:
    """LLM-render one name_capture line from its editable prompt.

    Mirrors greeting_response: resolve the tenant prompt (or the hardcoded
    fallback), substitute {persona}/{name}/{language_rule}, and render via one
    LLM call. Any failure or empty output degrades to `fallback_text` (the
    deterministic template) so the customer always gets a reply. Returns
    (text, turn_usage) — turn_usage is empty when the LLM produced nothing.
    """
    template = resolve_prompt(config, prompt_key, fallback_system)
    system_content = _render_prompt(template, values)
    try:
        provider = get_provider(config)
        model = resolve_model(config)
        text = ""
        async for chunk in provider.stream_chat(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
        ):
            text += chunk
        text = text.strip()
        if text:
            return text, [
                make_usage_record(node="name_capture", provider=provider, model=model)
            ]
    except Exception as exc:
        logger.warning(
            "name_capture_render_failed",
            thread_id=thread_id,
            prompt_key=prompt_key,
            error=str(exc),
        )
    return fallback_text, []


def _resolve_lang(state: AgentState) -> str:
    lang = (state.get("language") or "es").strip().lower()[:2]
    return lang if lang in _ASK_TEMPLATES else "es"


def _resolve_identity(state: AgentState, lang: str) -> tuple[str, str]:
    """(persona, role phrase) — same resolution chain as greeting_response."""
    code_name = (state.get("agent_code_name") or "").strip().lower()
    agent_type = (state.get("agent_type") or "").strip().lower()

    default_persona, roles = _AGENT_PROFILES.get(code_name, ("", {}))
    if not roles:
        roles = _TYPE_ROLE_FALLBACK.get(agent_type, _GENERIC_ROLES)
    if not default_persona:
        default_persona = code_name.title() if code_name else "Asistente"

    persona = resolve_persona(state, default_persona)
    role = roles.get(lang) or _GENERIC_ROLES[lang]
    return persona, role


async def _defer_on_backend(state: AgentState, thread_id: str) -> None:
    """Record the refusal server-side; local state defers regardless so a
    backend hiccup never turns into repeated nagging this thread."""
    tenant_id = state.get("tenant_id", "")
    contact_id = state.get("contact_id", "")
    if not (tenant_id and contact_id):
        return
    try:
        await backend_client.defer_contact_name_capture(
            contact_id=contact_id, tenant_id=tenant_id
        )
        logger.info(
            "name_capture_deferred", thread_id=thread_id, contact_id=contact_id
        )
    except (HTTPStatusError, RequestError) as exc:
        logger.error(
            "name_capture_defer_failed",
            thread_id=thread_id,
            contact_id=contact_id,
            error=str(exc),
        )


async def name_capture_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("name_capture")

    thread_id = state.get("thread_id", "unknown")
    lang = _resolve_lang(state)
    persona, role = _resolve_identity(state, lang)
    # All trailing user messages — the name may arrive in a fragment after a
    # greeting ("Hola" / "Soy Patricia Hernández").
    user_text = latest_user_text(state)

    # Shared substitution values for the editable NAME_CAPTURE_* prompts; `name`
    # is filled per-branch (only the acknowledgment knows it).
    base_values = {
        "persona": persona,
        "role": role,
        "name": "",
        "language_rule": language_instruction(lang),
    }

    provider = get_provider(config)
    model = resolve_model(config)

    messages_payload = [
        {"role": "system", "content": _NAME_EXTRACT_PROMPT},
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
    refused = False
    has_question = False
    try:
        parsed = json.loads(strip_json_fences(raw))
        found = bool(parsed.get("found", False))
        name = str(parsed.get("name", "")).strip()
        refused = bool(parsed.get("refused", False))
        has_question = bool(parsed.get("has_question", False))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "name_capture_parse_failed", thread_id=thread_id, error=str(exc), raw=raw
        )

    write = get_stream_writer()
    attempts = int(state.get("name_capture_attempts") or 0)

    # ── Refusal / attempts exhausted → defer, never invent a name ──────────
    if refused or (not (found and name) and attempts + 1 >= MAX_NAME_CAPTURE_ATTEMPTS):
        await _defer_on_backend(state, thread_id)
        if refused and not has_question:
            # Nothing else to answer — acknowledge and end the turn.
            ack, render_usage = await _render_llm(
                config,
                "NAME_CAPTURE_REFUSAL",
                _REFUSAL_SYSTEM_PROMPT,
                base_values,
                user_text or "Prefiero no dar mi nombre",
                _REFUSAL_ACK_TEMPLATES[lang],
                thread_id,
            )
            write({"type": "token", "content": ack})
            return {
                "messages": [AIMessage(content=ack)],
                "name_capture_deferred": True,
                "name_capture_reply_sent": True,
                "turn_usage": [usage_record, *render_usage],
            }
        # The message carries an actual request (or wasn't a refusal at all,
        # just a second miss) — fall through so the graph answers it now.
        return {
            "name_capture_deferred": True,
            "name_capture_reply_sent": False,
            "turn_usage": [usage_record],
        }

    # ── No name yet, first attempt → ask and end the turn ──────────────────
    if not (found and name):
        ask, render_usage = await _render_llm(
            config,
            "NAME_CAPTURE_ASK",
            _ASK_SYSTEM_PROMPT,
            base_values,
            user_text or "Hola",
            _ASK_TEMPLATES[lang].format(persona=persona, role=role),
            thread_id,
        )
        write({"type": "token", "content": ask})
        logger.info("name_capture_asked", thread_id=thread_id, attempt=attempts + 1)
        return {
            "messages": [AIMessage(content=ask)],
            "name_capture_attempts": attempts + 1,
            "name_capture_reply_sent": True,
            "turn_usage": [usage_record, *render_usage],
        }

    # ── Found → persist; the API result drives state ────────────────────────
    tenant_id = state.get("tenant_id", "")
    contact_id = state.get("contact_id", "")
    persisted = False
    if tenant_id and contact_id:
        try:
            resp = await backend_client.update_contact_name(
                contact_id=contact_id, name=name, tenant_id=tenant_id
            )
            persisted = True
            canonical = str(resp.get("name") or "").strip()
            if resp.get("applied") is False and canonical:
                # A human set this name in the CRM (MANUAL) — theirs wins.
                logger.info(
                    "name_capture_manual_conflict",
                    thread_id=thread_id,
                    contact_id=contact_id,
                )
                name = canonical
            elif canonical:
                # Adopt the backend-sanitized canonical form.
                name = canonical
        except (HTTPStatusError, RequestError) as exc:
            logger.error(
                "name_capture_backend_failed",
                thread_id=thread_id,
                contact_id=contact_id,
                error=str(exc),
            )

    updated_context = dict(state.get("user_context") or {})
    updated_context["name"] = name

    ack_usage: list = []
    if has_question:
        # Short prefix only — the downstream node produces the substantive
        # reply and the backend concatenates token events. Kept deterministic
        # so the LLM can't prepend a full greeting ahead of that answer.
        greeting = _GREETING_PREFIX_TEMPLATES[lang].format(name=name)
    else:
        greeting, ack_usage = await _render_llm(
            config,
            "NAME_CAPTURE_ACK",
            _ACK_SYSTEM_PROMPT,
            {**base_values, "name": name},
            user_text or f"Soy {name}",
            _GREETING_TEMPLATES[lang].format(name=name),
            thread_id,
        )
    write({"type": "token", "content": greeting})

    logger.info(
        "name_capture_persisted",
        thread_id=thread_id,
        contact_id=contact_id,
        name_length=len(name),
        persisted=persisted,
    )

    result: dict = {
        "messages": [AIMessage(content=greeting)],
        "user_context": updated_context,
        # Continue routing when the same message carries a real request so
        # the downstream node answers it in this turn.
        "name_capture_reply_sent": not has_question,
        "turn_usage": [usage_record, *ack_usage],
    }
    if persisted:
        # Latch only on confirmed persistence (or manual adoption): an HTTP
        # failure keeps the backend snapshot authoritative, so the graph asks
        # once more next turn only if the DB really never got the name.
        result["name_captured"] = True
        result["school_name_captured"] = True  # legacy camila compat
    return result
