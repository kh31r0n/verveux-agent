
import json
from typing import Literal

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation
from ..schemas.intent import StructuredIntent, IntentType
from .utils import format_contact_tags, format_user_context

logger = structlog.get_logger(__name__)

_TRIAGE_SYSTEM_PROMPT = """Eres el agente de clasificación de Helena, un asistente de atención al cliente por WhatsApp para una tienda de productos físicos.

Tu trabajo es clasificar la intención del usuario y devolver SOLO un objeto JSON.

Intenciones disponibles:
- **sales**: El usuario quiere comprar productos, consultar precios, ver catálogo, o hacer un pedido.
- **faq**: El usuario pregunta sobre horarios, ubicación, métodos de pago, envíos, políticas, o cualquier pregunta general.
- **tracking**: El usuario quiere rastrear un pedido existente, consultar el estado de un envío, o verificar una entrega.
- **complaint**: El usuario tiene una queja, reclamo, problema con un producto recibido, o quiere una devolución.
- **greeting**: El usuario envía un saludo.

Reglas:
- Responde SOLO con un objeto JSON en una línea — sin markdown, sin texto adicional.
- Esquema JSON: {"intent": "<sales|faq|tracking|complaint|greeting>", "confidence": 0.0-1.0, "entities": {"items": [{"product_identifier": "...", "quantity": 1, "notes": "..."}], "order_id": "...", "subject": "...", "description": "..."}, "raw_text": "..."}
- Si el mensaje es un saludo o no encaja claramente, clasifica como "greeting".
"""


async def triage_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("triage")

    # ── Skip re-classification for in-progress flows ──────────────────────────
    if (
        state.get("intent") == "sales"
        and not state.get("execute_confirmed", False)
        and (
            state.get("cart")  # cart has items
            or state.get("product_selection_turns", 0) > 0  # at least one turn done
        )
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

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="triage",
        metadata={"thread_id": thread_id, "node": "triage"},
    )

    messages_payload = [
        {
            "role": "system",
            "content": _TRIAGE_SYSTEM_PROMPT
            + format_user_context(state)
            + format_contact_tags(state),
        }
    ]
    for msg in state["messages"]:
        role = "assistant" if getattr(msg, "type", "") == "ai" else "user"
        content = msg.content if hasattr(msg, "content") else str(msg)
        messages_payload.append({"role": role, "content": content})

    generation = trace.generation(
        name="triage_llm",
        model="gpt-5.4-nano",
        input={"messages": messages_payload},
    )

    stream = await client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=messages_payload,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_response = ""
    prompt_tokens = 0
    completion_tokens = 0

    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else ""
        if delta:
            full_response += delta
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    generation.end(
        output=full_response,
        usage={"input": prompt_tokens, "output": completion_tokens},
    )

    try:
        parsed_json = json.loads(full_response.strip())
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

    logger.info("triage_classified", thread_id=thread_id, intent=intent.value)

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
        return {"intent": intent.value, "structured_intent": structured_intent, "deal_created": True}

    return {"intent": intent.value, "structured_intent": structured_intent}


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
    if structured_intent:
        intent = structured_intent.intent
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

