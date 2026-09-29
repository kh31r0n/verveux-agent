"""Restaurant order confirmation node — LLM-classified confirm / modify / unclear.

Entered only on a resumed turn (restaurant_phase == "confirmation"), i.e. the
customer is replying to the order summary. An LLM call classifies the reply;
on LLM failure a deterministic keyword fallback (shared with sales_confirm)
keeps the flow moving.

Outcomes:
  confirm → short streamed reply + restaurant_order_confirmed=True; the router
            auto-chains to execute (checkout) in the same turn.
  modify  → no reply here; phase returns to "collect" and the router re-enters
            order_collect in the same turn so the edit request is answered.
            The cart is preserved (edit, not restart).
  unclear → streamed re-ask (answering a simple question if there was one);
            phase stays "confirmation".
"""

import json

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..json_utils import strip_json_fences
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from ..services.cart import CartService, normalize_cart
from ..usage import make_usage_record
from .sales_confirm import _NO, _YES
from .utils import (
    format_user_context,
    language_instruction,
    latest_user_text,
    resolve_persona,
    resolve_prompt,
)

logger = structlog.get_logger(__name__)

_VALID_DECISIONS = {"confirm", "modify", "unclear"}

_CLASSIFY_SYSTEM_PROMPT = """Eres un clasificador de respuestas de clientes de un restaurante por WhatsApp.

Al cliente se le acaba de mostrar el resumen de su pedido y se le pidió responder
"confirmar" para enviarlo, o decir qué cambiar.

Clasifica su respuesta en UNA de estas decisiones:
- "confirm": aprueba claramente el pedido (ej. "confirmar", "sí, envíalo", "dale", "todo bien").
- "modify": quiere cambiar, agregar o quitar algo, o cancelar (ej. "quita las papas", "mejor 2 tacos", "no, espera").
- "unclear": la respuesta es ambigua o es una pregunta no relacionada con confirmar/modificar.

Resumen del pedido mostrado:
{order_summary}

Responde SOLO con un objeto JSON de una línea, sin markdown:
{{"decision": "confirm" | "modify" | "unclear", "confidence": 0.0-1.0}}"""

_CONFIRM_REPLY_PROMPT = """Eres {persona}, una asistente de pedidos por WhatsApp para un restaurante.
{language_rule}

El cliente acaba de confirmar su pedido. Responde con UN mensaje muy breve y cálido:
agradece, dile que su pedido fue enviado a la cocina y que le avisaremos cuando esté listo.
No repitas el detalle del pedido. No devuelvas JSON."""

_UNCLEAR_REPLY_PROMPT = """Eres {persona}, una asistente de pedidos por WhatsApp para un restaurante.
{language_rule}

El cliente respondió al resumen de su pedido con algo ambiguo o con una pregunta.
Su mensaje: "{user_text}"

Resumen del pedido pendiente:
{order_summary}

Responde brevemente (contesta su pregunta si la hay) y vuelve a pedirle que responda
**confirmar** para enviar el pedido, o que diga qué desea cambiar. No devuelvas JSON."""


def _keyword_fallback(user_text: str) -> str:
    """Deterministic classification when the LLM call fails."""
    text = user_text.strip().lower()
    tokens = set(text.split())
    if any(kw in text or kw in tokens for kw in _YES):
        return "confirm"
    if any(kw in text or kw in tokens for kw in _NO):
        return "modify"
    return "unclear"


async def restaurant_confirm_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("restaurant_confirm")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="restaurant_confirm_node",
        metadata={"thread_id": thread_id, "node": "restaurant_confirm"},
    )

    user_text = latest_user_text(state)
    cart = normalize_cart(state.get("cart"))
    order_data = state.get("restaurant_order_data") or {}

    summary_lines = [CartService.format_cart(cart)]
    if order_data.get("service_type"):
        summary_lines.append(f"Servicio: {order_data['service_type']}")
    if order_data.get("delivery_address"):
        summary_lines.append(f"Dirección: {order_data['delivery_address']}")
    order_summary_str = "\n".join(summary_lines)

    turn_usage: list = []

    # ── Classify the reply ────────────────────────────────────────────────────
    classify_prompt = resolve_prompt(
        config, "RESTAURANT_ORDER_CONFIRM", _CLASSIFY_SYSTEM_PROMPT, state
    )
    decision = ""
    try:
        classify_messages = [
            {"role": "system", "content": classify_prompt.format(order_summary=order_summary_str)},
            {"role": "user", "content": user_text or "(sin mensaje)"},
        ]
        gen = trace.generation(name="restaurant_confirm_classify", model=model, input={"messages": classify_messages})
        raw = ""
        async for chunk in provider.stream_chat(model=model, messages=classify_messages):
            raw += chunk
        turn_usage.append(
            make_usage_record(
                node="restaurant_confirm.classify", provider=provider, model=model,
            )
        )
        gen.end(output=raw)

        parsed = json.loads(strip_json_fences(raw))
        candidate = str(parsed.get("decision", "")).strip().lower()
        if candidate in _VALID_DECISIONS:
            decision = candidate
    except Exception as exc:
        logger.warning(
            "restaurant_confirm_classify_failed",
            thread_id=thread_id,
            error=str(exc),
        )

    if not decision:
        decision = _keyword_fallback(user_text)
        logger.info(
            "restaurant_confirm_llm_fallback",
            thread_id=thread_id,
            decision=decision,
        )

    logger.info(
        "restaurant_confirm_decision",
        thread_id=thread_id,
        decision=decision,
    )

    # ── modify: no reply here — order_collect answers this same message ───────
    if decision == "modify":
        return {
            "restaurant_order_confirmed": False,
            "restaurant_phase": "collect",
            "restaurant_order_complete": False,
            "turn_usage": turn_usage,
        }

    # ── confirm / unclear: streamed reply ─────────────────────────────────────
    persona = resolve_persona(state, "Giulia")
    lang_rule = language_instruction(state.get("language", "en"))

    if decision == "confirm":
        system_content = _CONFIRM_REPLY_PROMPT.format(
            persona=persona, language_rule=lang_rule
        )
    else:
        system_content = _UNCLEAR_REPLY_PROMPT.format(
            persona=persona,
            language_rule=lang_rule,
            user_text=user_text,
            order_summary=order_summary_str,
        )
    system_content += format_user_context(state)

    reply_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_text or "(sin mensaje)"},
    ]
    gen2 = trace.generation(name="restaurant_confirm_reply", model=model, input={"messages": reply_messages})
    write = get_stream_writer()
    full_response = ""
    async for chunk in provider.stream_chat(model=model, messages=reply_messages):
        write({"type": "token", "content": chunk})
        full_response += chunk
    turn_usage.append(
        make_usage_record(
            node="restaurant_confirm.reply", provider=provider, model=model,
        )
    )
    gen2.end(output=full_response)

    result: dict = {
        "messages": [AIMessage(content=full_response)],
        "turn_usage": turn_usage,
    }
    if decision == "confirm":
        result["restaurant_order_confirmed"] = True
    return result
