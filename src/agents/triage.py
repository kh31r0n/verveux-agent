import json
from typing import Literal

import structlog
from langchain_core.runnables import RunnableConfig

from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation
from .utils import format_contact_tags, format_user_context, resolve_prompt

logger = structlog.get_logger(__name__)

_TRIAGE_SYSTEM_PROMPT = """Eres el agente de clasificación de Helena, un asistente de atención al cliente por WhatsApp para una tienda de productos físicos.

Tu trabajo es clasificar la intención del usuario y devolver SOLO un objeto JSON.

Intenciones disponibles:
- **sales**: El usuario quiere comprar productos, consultar precios, ver catálogo, o hacer un pedido.
- **tracking**: El usuario quiere rastrear un pedido existente, consultar el estado de un envío, o verificar una entrega.
- **complaint**: El usuario tiene una queja, reclamo, problema con un producto recibido, o quiere una devolución.
- **order_history**: El usuario pregunta por sus pedidos anteriores, historial de compras, o quiere ver qué ha comprado antes.
- **faq**: El usuario pregunta sobre horarios, ubicación, métodos de pago, envíos, políticas, o cualquier pregunta general.

Reglas:
- Responde SOLO con un objeto JSON en una línea — sin markdown, sin texto adicional.
- Esquema JSON: {"intent": "<sales|tracking|complaint|order_history|faq>", "suggest_tags": ["NombreTag"]}
- "suggest_tags" es opcional. Incluye "Urgente" si el usuario suena muy molesto, frustrado, o tiene un problema serio.
- Si el mensaje es un saludo o no encaja claramente, clasifica como "faq".
"""


async def triage_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("triage")

    # If mid-sales data collection (steps not yet complete), skip re-classification
    # so the agent stays focused on collecting order info.
    # Once sales_complete=True (at summary/confirmation phase) we DO re-classify
    # so the user can ask FAQ questions without being treated as making corrections.
    if (
        state.get("intent") == "sales"
        and state.get("sales_step", 0) > 0
        and not state.get("sales_complete", False)
        and not state.get("execute_confirmed", False)
    ):
        return {}

    # If already mid-tracking flow and not yet complete, skip re-classification
    if state.get("intent") == "tracking" and not state.get("execute_confirmed", False) and state.get("tracking_data"):
        return {}

    # If already mid-complaint flow and not yet complete, skip re-classification
    if state.get("intent") == "complaint" and not state.get("execute_confirmed", False) and state.get("complaint_data"):
        return {}

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="triage",
        metadata={"thread_id": thread_id, "node": "triage"},
    )

    triage_prompt = resolve_prompt(config, "TRIAGE", _TRIAGE_SYSTEM_PROMPT)
    messages_payload = [{"role": "system", "content": triage_prompt + format_user_context(state) + format_contact_tags(state)}]
    for msg in state["messages"]:
        if hasattr(msg, "type"):
            role = "assistant" if msg.type == "ai" else "user"
        else:
            role = "user"
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
        parsed = json.loads(full_response.strip())
        intent: str = parsed.get("intent", "faq")
    except (json.JSONDecodeError, AttributeError):
        logger.warning("triage_json_parse_failed", raw=full_response[:200])
        parsed = {}
        intent = "faq"

    valid_intents = {"sales", "tracking", "complaint", "order_history", "faq"}
    if intent not in valid_intents:
        intent = "faq"

    # Emit tag_contact SSE events for suggested tags
    suggested_tags: list = parsed.get("suggest_tags", []) if isinstance(parsed, dict) else []
    contact_id: str = state.get("contact_id", "")
    if suggested_tags and contact_id:
        write = get_stream_writer()
        for tag_name in suggested_tags:
            if isinstance(tag_name, str) and tag_name.strip():
                write({
                    "type": "tag_contact",
                    "contact_id": contact_id,
                    "tag_name": tag_name.strip(),
                })
                logger.info("triage_suggest_tag", thread_id=thread_id, tag=tag_name)

    logger.info("triage_classified", thread_id=thread_id, intent=intent)

    # Emit create_deal SSE event when classifying as "sales" for the first time
    deal_created = state.get("deal_created", False)
    if intent == "sales" and not deal_created and contact_id:
        write = get_stream_writer()
        write({
            "type": "create_deal",
            "contact_id": contact_id,
            "conversation_id": state.get("conversation_id", ""),
            "title": "Pedido de cliente",
            "source": "WHATSAPP",
        })
        logger.info("triage_deal_created", thread_id=thread_id, contact_id=contact_id)
        return {"intent": intent, "deal_created": True}

    # Triage is silent — downstream nodes handle all user-facing messages
    return {"intent": intent}


def route_from_triage(
    state: AgentState,
) -> Literal["sales_collect", "order_summary", "tracking_collect", "complaint_collect", "order_history", "faq_response", "execute"]:
    intent = state.get("intent", "faq")

    if intent == "sales":
        # Once execution is confirmed, treat as faq so the user can start fresh
        if state.get("execute_confirmed", False):
            return "faq_response"

        if state.get("order_confirmed", False):
            return "execute"
        if state.get("sales_complete", False):
            return "order_summary"
        return "sales_collect"

    if intent == "tracking":
        if state.get("execute_confirmed", False):
            return "faq_response"
        if state.get("tracking_complete", False):
            return "execute"
        return "tracking_collect"

    if intent == "complaint":
        if state.get("execute_confirmed", False):
            return "faq_response"
        if state.get("complaint_complete", False):
            return "execute"
        return "complaint_collect"

    if intent == "order_history":
        return "order_history"

    return "faq_response"
