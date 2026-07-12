import json
from typing import Literal

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..json_utils import strip_json_fences
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from ..schemas.intent import StructuredIntent, IntentType
from ..usage import make_usage_record
from .utils import format_contact_tags, format_user_context, resolve_prompt

logger = structlog.get_logger(__name__)

_APPOINTMENTS_TRIAGE_SYSTEM_PROMPT = """Eres el agente de clasificación de Marco, un asistente de agendamiento de citas por WhatsApp.

Tu trabajo es clasificar la intención del usuario y devolver SOLO un objeto JSON.

Intenciones disponibles:
- **booking**: El usuario quiere reservar/agendar una cita nueva, ver opciones, o iniciar el flujo de booking.
- **appointment_cancel**: El usuario quiere cancelar una cita existente.
- **appointment_reschedule**: El usuario quiere mover/cambiar una cita existente a otro horario.
- **faq**: El usuario hace una pregunta general (horarios de atención, precios, ubicación, qué servicios ofrecen, etc.).
- **greeting**: El usuario saluda sin intención clara.
- **escalation**: El usuario pide hablar con una persona humana o expresa una queja seria.

Reglas:
- Si el usuario menciona "agendar", "reservar", "tomar cita", "quiero una cita" → **booking**.
- Si menciona "cancelar", "anular" su cita → **appointment_cancel**.
- Si menciona "reagendar", "cambiar fecha", "mover" su cita → **appointment_reschedule**.
- Preguntas sobre disponibilidad general SIN compromiso de reservar → **faq** (el flujo de booking ya muestra disponibilidad).
- Responde SOLO con un objeto JSON de una línea, sin markdown.
- Esquema JSON: {{"intent": "<booking|appointment_cancel|appointment_reschedule|faq|greeting|escalation>", "confidence": 0.0-1.0, "raw_text": "..."}}
"""


_TRIAGE_SYSTEM_PROMPT = """Eres el agente de clasificación de Helena, un asistente de atención al cliente por WhatsApp para una tienda de productos físicos.

Tu trabajo es clasificar la intención del usuario y devolver SOLO un objeto JSON.

Intenciones disponibles:
- **sales**: El usuario quiere comprar productos, consultar precios, ver catálogo, o hacer un pedido.
- **faq**: El usuario pregunta sobre horarios, ubicación, métodos de pago, envíos, políticas, o cualquier pregunta general.
- **tracking**: El usuario quiere rastrear un pedido existente, consultar el estado de un envío, o verificar una entrega.
- **complaint**: El usuario tiene una queja, reclamo, problema con un producto recibido, o quiere una devolución.
- **greeting**: El usuario envía un saludo.

Reglas de clasificación estrictas:
- Preguntas sobre PROMOCIONES, DESCUENTOS, PRECIOS, OFERTAS → clasifica como **faq** (NO como sales).
- Preguntas sobre horarios, ubicación, métodos de pago, envíos, políticas → **faq**.
- Intención clara de COMPRA ("quiero comprar", "pedir", "agregar al carrito") → **sales**.
- Un saludo o mensaje sin intención clara → **greeting**.
- Responde SOLO con un objeto JSON en una línea — sin markdown, sin texto adicional.
- Esquema JSON: {{"intent": "<sales|faq|tracking|complaint|greeting>", "confidence": 0.0-1.0, "entities": {{"items": [{{"product_identifier": "...", "quantity": 1, "notes": "..."}}], "order_id": "...", "subject": "...", "description": "..."}}, "raw_text": "..."}}
"""

_FAQ_HINTS_BLOCK = """
Preguntas frecuentes disponibles (si el usuario pregunta algo similar, clasifica como **faq**):
{faq_lines}

IMPORTANTE: Si la pregunta del usuario coincide o es similar a alguna de las preguntas frecuentes anteriores, SIEMPRE clasifica como **faq**, incluso si menciona precios o productos.
"""


def _build_faq_hints(faqs: list) -> str:
    """Format FAQs as hint block for the triage classifier."""
    if not faqs:
        return ""
    lines = []
    for faq in faqs:
        q = faq.get("question", "").strip()
        cat = faq.get("category", "")
        if q:
            suffix = f" [{cat}]" if cat else ""
            lines.append(f"- {q}{suffix}")
    if not lines:
        return ""
    return _FAQ_HINTS_BLOCK.format(faq_lines="\n".join(lines))


# Moved to src/json_utils.py; alias kept for existing importers.
_strip_json_fences = strip_json_fences


async def triage_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("triage")

    # ── Skip re-classification for in-progress flows ──────────────────────────
    #
    # Guard conditions must be CONSERVATIVE: only skip when there is clear
    # evidence the user is mid-flow, not just because state has stale values.
    #
    # IMPORTANT: `product_selection_turns > 0` alone is NOT a safe signal.
    # A previous abandoned sales conversation leaves turns > 0 in the
    # checkpoint even after the user moves on to an unrelated question.
    # Using it here caused stale `intent=sales` to suppress re-classification
    # and send FAQ queries (e.g. "¿Qué promocion tienen?") into sales_collect.
    #
    # Rule: skip only when the cart actually contains items (real in-progress order).
    if (
        state.get("intent") == "sales"
        and not state.get("execute_confirmed", False)
        and bool(state.get("cart"))  # must have actual cart items, not just stale turns
    ):
        return {}

    # Restaurant: skip re-classification while an order is actively in
    # progress. Conservative like the sales guard: require real cart items OR
    # the awaiting-confirmation latch — restaurant_order_data alone can be
    # stale from an abandoned order.
    if (
        state.get("intent") == "order"
        and not state.get("execute_confirmed", False)
        and (bool(state.get("cart")) or state.get("restaurant_phase") == "confirmation")
    ):
        return {}

    if (
        state.get("intent") == "tracking"
        and not state.get("execute_confirmed", False)
        and state.get("tracking_data")
    ):
        return {}

    if (
        state.get("intent") == "complaint"
        and not state.get("execute_confirmed", False)
        and state.get("complaint_data")
    ):
        return {}

    # ── Appointments: skip when a booking flow is mid-run ───────────────────
    # The lifecycle nodes set `booking_intent` ("book" | "cancel" | "reschedule")
    # so follow-up messages ("la 2 por favor", "sí confirmo") don't get
    # re-triaged into faq.
    if (
        state.get("booking_intent")
        and not state.get("booking_confirmed", False)
    ):
        return {}

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="triage",
        metadata={"thread_id": thread_id, "node": "triage"},
    )

    # ── Build system prompt with optional FAQ hints ───────────────────────────
    agent_type = (state.get("agent_type") or "sales").upper()
    code_name = (state.get("agent_code_name") or "").lower()
    triage_key = f"{agent_type}_TRIAGE"
    # Pick the appointments-specific default when running under the appointments
    # graph so the LLM doesn't try to classify booking talk as `sales`.
    default_prompt = (
        _APPOINTMENTS_TRIAGE_SYSTEM_PROMPT
        if code_name == "marco" or agent_type == "APPOINTMENTS"
        else _TRIAGE_SYSTEM_PROMPT
    )
    triage_prompt = resolve_prompt(config, triage_key, default_prompt, state)

    faqs: list = state.get("faqs") or []
    faq_hints = _build_faq_hints(faqs)

    system_content = (
        triage_prompt
        + faq_hints
        + format_user_context(state)
        + format_contact_tags(state)
    )

    messages_payload = [{"role": "system", "content": system_content}]
    for msg in state["messages"]:
        role = "assistant" if getattr(msg, "type", "") == "ai" else "user"
        content = msg.content if hasattr(msg, "content") else str(msg)
        messages_payload.append({"role": role, "content": content})

    generation = trace.generation(
        name="triage_llm",
        model=model,
        input={"messages": messages_payload},
    )

    stream = provider.stream_chat(
        model=model,
        messages=messages_payload,
    )

    full_response = ""

    async for chunk in stream:
        full_response += chunk

    usage_record = make_usage_record(node="triage", provider=provider, model=model)

    generation.end(
        output=full_response,
    )

    try:
        # Strip markdown code fences that Gemini adds despite being told not to.
        parsed_json = json.loads(strip_json_fences(full_response))
        structured_intent = StructuredIntent.model_validate(parsed_json)
        intent = structured_intent.intent
    except (ValidationError, json.JSONDecodeError) as e:
        logger.warning("triage_parse_failed", error=str(e), raw=full_response)
        structured_intent = StructuredIntent(intent=IntentType.UNKNOWN, confidence=0.0, raw_text=full_response)
        intent = IntentType.UNKNOWN

    write = get_stream_writer()
    write({
        "type": "intent_classified",
        "intent": intent.value,
        "confidence": structured_intent.confidence,
    })

    logger.info(
        "triage_classified",
        thread_id=thread_id,
        intent=intent.value,
        faq_hints_injected=len(faqs),
        normalization_applied=bool((state.get("normalization") or {}).get("applied")),
    )

    # Emit create_deal SSE event when classifying as "sales" for the first time
    deal_created = state.get("deal_created", False)
    contact_id: str = state.get("contact_id", "")
    if intent == IntentType.SALES and not deal_created and contact_id:
        write({
            "type": "create_deal",
            "contact_id": contact_id,
            "conversation_id": state.get("conversation_id", ""),
            "title": "Pedido de cliente",
            "source": "WHATSAPP",
        })
        logger.info("triage_deal_created", thread_id=thread_id, contact_id=contact_id)
        return {
            "intent": intent.value,
            # Stored as a plain JSON dict — checkpoints must never carry
            # custom Python types (LangGraph blocks their deserialization
            # in future versions). route_from_triage handles both shapes.
            "structured_intent": structured_intent.model_dump(mode="json"),
            "deal_created": True,
            "turn_usage": [usage_record],
        }

    return {
        "intent": intent.value,
        # Plain JSON dict — keeps custom Python types out of checkpoints.
        "structured_intent": structured_intent.model_dump(mode="json"),
        "turn_usage": [usage_record],
    }


def route_from_triage(
    state: AgentState,
) -> Literal[
    "sales_collect",
    "sales_confirm",
    "customer_data_collect",
    "order_summary",
    "tracking_collect",
    "complaint_collect",
    "faq_response",
    "execute",
]:
    structured_intent = state.get("structured_intent")
    if structured_intent is not None:
        # structured_intent may come back from checkpoint deserialization as a
        # plain dict (if the Pydantic model could not be fully reconstructed).
        # Fall back to the string intent field in that case.
        if hasattr(structured_intent, "intent"):
            intent = structured_intent.intent
        elif isinstance(structured_intent, dict):
            intent = structured_intent.get("intent", state.get("intent", "faq"))
        else:
            intent = state.get("intent", "faq")
    else:
        intent = state.get("intent", "faq")

    # ── Sales — route to the correct phase ────────────────────────────────────
    if intent == IntentType.SALES:
        if state.get("execute_confirmed", False):
            # Order already executed; treat new messages as FAQ
            return "faq_response"

        # Final confirmation happened → trigger execution
        if state.get("order_confirmed", False):
            return "execute"

        # Explicit phase field drives routing when mid-flow
        phase = state.get("sales_phase", "product_selection")

        if phase == "payment":
            # Customer data collected, show final summary
            return "order_summary"

        if phase == "customer_data":
            return "customer_data_collect"

        if phase == "product_confirmation":
            return "sales_confirm"

        # Default / product_selection phase
        return "sales_collect"

    # ── Tracking ──────────────────────────────────────────────────────────────
    if intent == IntentType.TRACKING:
        if state.get("execute_confirmed", False):
            return "faq_response"
        if state.get("tracking_complete", False):
            return "execute"
        return "tracking_collect"

    # ── Complaint ─────────────────────────────────────────────────────────────
    if intent == IntentType.COMPLAINT:
        if state.get("execute_confirmed", False):
            return "faq_response"
        if state.get("complaint_complete", False):
            return "execute"
        return "complaint_collect"

    return "faq_response"