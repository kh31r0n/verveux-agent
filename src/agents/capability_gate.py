"""Single policy gate for the CATALOG capability toggle.

Mirrors the `shared_routing` single-gate pattern: every node that can put
product/catalog data in front of the model consults `catalog_allowed(state)`.
The decision is resolved by the backend fresh every turn and forwarded as
`state["catalog_access_enabled"]`, so an admin flipping the toggle takes effect
on the conversation's next turn. When off, the backend already sends an empty
`product_catalog` and the internal cart/order endpoints 403 — this module drives
the Python-side prompt gate and the degraded, no-hallucination reply.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState

# Instruction appended (only when off) to nodes that otherwise interpolate the
# catalog — e.g. faq_response. Keeps FAQ knowledge intact (FAQs are out of scope
# for this toggle) while forbidding product/price/order invention. Never leaks
# internal configuration to the end user.
CATALOG_DEGRADED_INSTRUCTION = (
    "\n\nIMPORTANTE — SIN ACCESO AL CATÁLOGO: No tienes acceso al catálogo de "
    "productos ni a la información de pedidos en este momento. No inventes, "
    "listes ni describas productos. No infieras ni adivines precios. Si el "
    "usuario pregunta por productos, el menú, precios, disponibilidad o el "
    "estado de un pedido, explícale amablemente que no puedes compartir esa "
    "información y pídele que contacte directamente con el negocio. Nunca "
    "menciones configuraciones internas, ajustes ni errores."
)

# Ready-made reply for nodes whose entire purpose is catalog/order work
# (sales_collect, menu_inquiry, ...). They early-return this instead of trying
# to build an order against an empty catalog.
_DEGRADED_REPLY: dict[str, str] = {
    "es": (
        "Lo siento, en este momento no puedo consultar el catálogo de productos "
        "ni la información de pedidos. Por favor, contacta directamente con el "
        "negocio para ayudarte con eso. ¿Hay algo más en lo que pueda ayudarte?"
    ),
    "en": (
        "I'm sorry, I can't access the product catalog or order information "
        "right now. Please contact the business directly for help with that. "
        "Is there anything else I can help you with?"
    ),
    "pt": (
        "Desculpe, não consigo acessar o catálogo de produtos nem informações "
        "de pedidos no momento. Por favor, entre em contato diretamente com o "
        "negócio. Posso ajudar em mais alguma coisa?"
    ),
}


def catalog_allowed(state: AgentState) -> bool:
    """True when the tenant permits catalog access for this agent.

    Absent (old checkpoints, or an older backend that doesn't send the flag) is
    treated as allowed so behavior is unchanged unless the backend explicitly
    disables it.
    """
    value = state.get("catalog_access_enabled")
    if value is None:
        return True
    return bool(value)


def _degraded_reply_text(state: AgentState) -> str:
    lang = (state.get("language") or "es")[:2].lower()
    return _DEGRADED_REPLY.get(lang, _DEGRADED_REPLY["es"])


def emit_degraded_catalog_reply(state: AgentState) -> dict:
    """Stream + return the degraded reply for an early node exit.

    Streams the text as `token` events (how NestJS reconstructs the outbound
    message) and returns it as the turn's AIMessage. No LLM call, so
    `turn_usage` is empty (nothing billed), mirroring the greeting fallback.
    """
    text = _degraded_reply_text(state)
    try:
        write = get_stream_writer()
        write({"type": "token", "content": text})
    except Exception:  # pragma: no cover - stream writer only absent off-graph
        pass
    return {
        "messages": [AIMessage(content=text)],
        "turn_usage": [],
    }
