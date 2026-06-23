"""Camila triage node — school academic-secretary intent classification.

Extends the base triage prompt with handoff-bound intents (payment proofs,
document corrections, academic record lookups) and asks the LLM for a
primary `intent` plus a `secondary_intents` list so multi-intent messages
("aquí va la transferencia y necesito corregir mi apellido") flip the
conversation to a human on the first turn.

A deterministic identity-conflict check runs before the LLM call so the
classic Neiva Cortés / Neiva Torres scenario escalates even if the
classifier is uncertain.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from pydantic import ValidationError

from ..graphs.state import AgentState
from ..observability import get_langfuse, record_node_invocation
from ..providers.registry import get_provider, resolve_model
from ..schemas.intent import IntentType, StructuredIntent
from ..usage import make_usage_record
from .identity_validator import detect_identity_conflict
from .triage import _build_faq_hints, _strip_json_fences
from .utils import format_contact_tags, format_user_context, resolve_prompt

logger = structlog.get_logger(__name__)

_CAMILA_TRIAGE_SYSTEM_PROMPT = """Eres Camila, la secretaria académica virtual de una institución educativa por WhatsApp.

Tu trabajo es clasificar la intención del usuario y devolver SOLO un objeto JSON. Detecta TODAS las intenciones presentes en el mensaje (intención principal + secundarias).

Intenciones disponibles:
- **payment_proof**: El usuario informa que envió un pago, transferencia, consignación, recibo o comprobante.
- **correction_request**: El usuario pide corregir un dato (apellido, nombre, cédula, correo, fecha, etc.).
- **academic_lookup**: El usuario pide información de su historial académico (notas, fechas de estudios previos, estado de matrícula, certificados emitidos).
- **faq**: El usuario pregunta sobre horarios, ubicación, costos, métodos de pago, requisitos generales o políticas de la institución.
- **greeting**: El usuario solo saluda o se presenta sin pedir nada concreto.

Reglas:
- Si el mensaje incluye palabras como "transferencia", "pago", "consignación", "recibo", "envío comprobante" → `payment_proof`.
- Si el mensaje pide cambiar/corregir/actualizar un dato personal → `correction_request`.
- Si el mensaje pregunta por notas, calificaciones, fechas exactas de estudios previos, validación de historial → `academic_lookup`.
- Si la pregunta es general (horarios, costos, requisitos), responde `faq`.
- Si el usuario presenta dos intenciones en el mismo mensaje (ej. "envío pago y necesito corregir apellido"), elige UNA como principal y pon la otra en `secondary_intents`.
- Responde SOLO con un objeto JSON en una línea — sin markdown, sin texto adicional.

Esquema JSON:
{{"intent": "<payment_proof|correction_request|academic_lookup|faq|greeting>", "confidence": 0.0-1.0, "secondary_intents": ["<intent>", ...], "entities": {{"subject": "...", "description": "..."}}, "raw_text": "..."}}
"""


_HANDOFF_INTENTS = {
    IntentType.PAYMENT_PROOF,
    IntentType.CORRECTION_REQUEST,
    IntentType.ACADEMIC_LOOKUP,
    IntentType.IDENTITY_CONFLICT,
}


async def camila_triage_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("camila_triage")

    thread_id = state.get("thread_id", "unknown")
    messages = state.get("messages") or []

    # ── Deterministic identity-conflict short-circuit ────────────────────────
    # Pure-Python, no LLM. If the user mentioned two distinct full names or
    # two distinct documento numbers across the conversation, classify as
    # IDENTITY_CONFLICT and skip the classifier call.
    if detect_identity_conflict(messages):
        logger.info("camila_triage_identity_conflict", thread_id=thread_id)
        structured = StructuredIntent(
            intent=IntentType.IDENTITY_CONFLICT,
            confidence=1.0,
            raw_text="",
        )
        write = get_stream_writer()
        write(
            {
                "type": "intent_classified",
                "intent": IntentType.IDENTITY_CONFLICT.value,
                "confidence": 1.0,
            }
        )
        return {
            "intent": IntentType.IDENTITY_CONFLICT.value,
            "structured_intent": structured,
            "handoff_reason": "identity_conflict",
        }

    # ── LLM classification ───────────────────────────────────────────────────
    provider = get_provider(config)
    model = resolve_model(config)

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="camila_triage",
        metadata={"thread_id": thread_id, "node": "camila_triage"},
    )

    triage_prompt = resolve_prompt(
        config, "SCHOOL_CAMILA_TRIAGE", _CAMILA_TRIAGE_SYSTEM_PROMPT
    )

    faqs: list = state.get("faqs") or []
    faq_hints = _build_faq_hints(faqs)

    system_content = (
        triage_prompt
        + faq_hints
        + format_user_context(state)
        + format_contact_tags(state)
    )

    messages_payload = [{"role": "system", "content": system_content}]
    for msg in messages:
        role = "assistant" if getattr(msg, "type", "") == "ai" else "user"
        content = msg.content if hasattr(msg, "content") else str(msg)
        messages_payload.append({"role": role, "content": content})

    generation = trace.generation(
        name="camila_triage_llm",
        model=model,
        input={"messages": messages_payload},
    )

    stream = provider.stream_chat(model=model, messages=messages_payload)
    full_response = ""
    async for chunk in stream:
        full_response += chunk

    usage_record = make_usage_record(
        node="camila_triage", provider=provider, model=model
    )
    generation.end(output=full_response)

    structured: StructuredIntent
    intent: IntentType
    try:
        parsed_json = json.loads(_strip_json_fences(full_response))
        structured = StructuredIntent.model_validate(parsed_json)
        intent = structured.intent
    except (ValidationError, json.JSONDecodeError) as exc:
        logger.warning(
            "camila_triage_parse_failed",
            thread_id=thread_id,
            error=str(exc),
            raw=full_response,
        )
        structured = StructuredIntent(
            intent=IntentType.UNKNOWN, confidence=0.0, raw_text=full_response
        )
        intent = IntentType.UNKNOWN

    # If the LLM returned a secondary intent that requires handoff, promote it.
    handoff_intent: IntentType | None = None
    if intent in _HANDOFF_INTENTS:
        handoff_intent = intent
    else:
        for sec in structured.secondary_intents:
            if sec in _HANDOFF_INTENTS:
                handoff_intent = sec
                break

    write = get_stream_writer()
    write(
        {
            "type": "intent_classified",
            "intent": intent.value,
            "confidence": structured.confidence,
        }
    )

    logger.info(
        "camila_triage_classified",
        thread_id=thread_id,
        intent=intent.value,
        secondary_intents=[s.value for s in structured.secondary_intents],
        promoted_handoff=handoff_intent.value if handoff_intent else None,
    )

    update: dict = {
        "intent": (handoff_intent or intent).value,
        "structured_intent": structured,
        "turn_usage": [usage_record],
    }
    if handoff_intent is not None:
        update["handoff_reason"] = handoff_intent.value
    return update
