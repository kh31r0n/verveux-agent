"""Restaurant menu inquiry node — answers questions about menu items."""

import unicodedata

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from ..usage import make_usage_record
from .utils import (
    language_instruction,
    latest_user_text,
    resolve_prompt,
    format_user_context,
)
from .capability_gate import catalog_allowed, emit_degraded_catalog_reply

logger = structlog.get_logger(__name__)

# Spanish filler words that carry no signal when matching dish names.
_NAME_STOPWORDS = {"de", "del", "la", "el", "con", "los", "las", "y", "a", "al", "en"}

_MENU_INQUIRY_SYSTEM_PROMPT = """Eres Helena, una asistente de restaurante por WhatsApp.

El usuario tiene una pregunta sobre el menú, platillos, ingredientes, precios o disponibilidad.
Responde de forma clara y apetitosa usando el catálogo de productos proporcionado.

{language_rule}

Reglas:
- Usa la información del catálogo/menú proporcionado para responder.
- Si preguntan por ingredientes o alérgenos que no conoces, indica que pueden consultar con el restaurante.
- Puedes sugerir platillos populares o complementos si es apropiado.
- Sé concisa y amigable — es WhatsApp.
- Si el usuario quiere ordenar, indícale que puedes tomar su pedido.
- Nunca digas que no puedes enviar imágenes. Los platillos marcados con [foto disponible] tienen fotografía y el sistema la adjunta automáticamente a tu respuesta cuando el platillo está claro — confírmalo con naturalidad (p. ej. «¡Claro! Aquí te comparto la foto 📸»). Si el platillo no está marcado con [foto disponible], discúlpate brevemente y descríbelo de forma apetitosa."""


def _format_menu(state: AgentState) -> str:
    """Build a menu block from product catalog."""
    catalog = state.get("product_catalog") or []
    if not catalog:
        return ""

    items = []
    for item in catalog[:30]:
        line = f"- {item.get('name', 'Item')}"
        if item.get("price"):
            line += f" — ${item['price']}"
        if item.get("description"):
            line += f" ({item['description']})"
        if item.get("has_image"):
            line += " [foto disponible]"
        items.append(line)

    return "\n\nMenú disponible:\n" + "\n".join(items)


def _normalize_tokens(text: str) -> set[str]:
    """Lowercase, accent-strip, and tokenize free text for name matching."""
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c)).lower()
    return {t for t in "".join(c if c.isalnum() else " " for c in folded).split() if t}


def _name_matches(name: str, text_tokens: set[str]) -> bool:
    """A catalog name matches when its significant tokens appear in the text.

    Single-token names need that token; longer names need any 2 significant
    tokens, so "bandeja paisa" still hits "Bandeja Paisa Tradicional".
    """
    significant = [
        t for t in _normalize_tokens(name) if len(t) >= 3 and t not in _NAME_STOPWORDS
    ]
    if not significant:
        return False
    needed = min(len(significant), 2)
    return sum(1 for t in significant if t in text_tokens) >= needed


def _mentioned_in(catalog: list[dict], text: str) -> set[str]:
    tokens = _normalize_tokens(text)
    if not tokens:
        return set()
    return {
        p["product_id"]
        for p in catalog
        if p.get("product_id") and _name_matches(p.get("name", ""), tokens)
    }


def _compute_mentioned_product_ids(
    catalog: list[dict],
    burst_text: str,
    history_texts: list[str],
) -> list[str]:
    """
    Returns ``[product_id]`` when exactly one distinct dish is referenced,
    otherwise ``[]`` (mirrors the sales-graph single-product image rule).

    Heuristic-only, like sales' ``_compute_mentioned_product_ids`` — this is a
    best-effort signal for image attachment, not an authoritative operation,
    so no LLM call. The trailing user burst is authoritative; when it names
    nothing (e.g. "¿tienes fotos del plato?"), the recent conversation window
    is scanned so the dish the bot just described still resolves. More than
    one match anywhere is ambiguous → no image.
    """
    if not catalog:
        return []

    burst_hits = _mentioned_in(catalog, burst_text)
    if len(burst_hits) == 1:
        return list(burst_hits)
    if len(burst_hits) > 1:
        return []

    history_hits = _mentioned_in(catalog, "\n".join(history_texts))
    return list(history_hits) if len(history_hits) == 1 else []


def _photo_attachment_note(catalog: list[dict], mentioned_ids: list[str]) -> str:
    """System note telling the LLM whether a photo rides this reply.

    The backend attaches the image only for a single mentioned product that
    has an image and stock > 0 — this mirrors that gate so the reply text and
    the actual attachment can never contradict each other. The explicit
    "aunque lo hayas dicho antes" wording exists because the conversation
    history may contain the bot's own pre-feature denials, which otherwise
    outweigh a generic rule (observed with gemini-2.5-flash).
    """
    if len(mentioned_ids) != 1:
        return ""
    product = next(
        (p for p in catalog if p.get("product_id") == mentioned_ids[0]), None
    )
    if not product:
        return ""
    if product.get("has_image") and (product.get("stock") or 0) > 0:
        return (
            f"\n\nNOTA DEL SISTEMA: la foto de «{product.get('name', '').strip()}» "
            "se está adjuntando automáticamente a esta respuesta. Confírmalo con "
            "naturalidad (p. ej. «¡Aquí tienes la foto!») y NUNCA digas que no "
            "puedes enviar imágenes, aunque lo hayas dicho antes en esta "
            "conversación."
        )
    return (
        f"\n\nNOTA DEL SISTEMA: «{product.get('name', '').strip()}» no tiene foto "
        "disponible en este momento. Si el cliente pide una imagen, discúlpate "
        "brevemente y descríbelo de forma apetitosa — no prometas enviarla."
    )


async def menu_inquiry_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("menu_inquiry")

    # CATALOG gate: the menu IS the catalog. With access off, deflect politely.
    if not catalog_allowed(state):
        logger.info(
            "capability_block",
            capability="CATALOG",
            source="node",
            node="menu_inquiry",
            thread_id=state.get("thread_id", "unknown"),
        )
        return emit_degraded_catalog_reply(state)

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="menu_inquiry_node",
        metadata={"thread_id": thread_id, "node": "menu_inquiry"},
    )

    catalog = state.get("product_catalog") or []
    mentioned_product_ids = _compute_mentioned_product_ids(
        catalog,
        latest_user_text(state),
        [
            m.content
            for m in state["messages"][-6:]
            if hasattr(m, "content") and isinstance(m.content, str)
        ],
    )

    lang_rule = language_instruction(state.get("language", "en"))
    prompt = resolve_prompt(config, "MENU_INQUIRY", _MENU_INQUIRY_SYSTEM_PROMPT, state)
    system_content = prompt.format(language_rule=lang_rule)
    system_content += _format_menu(state)
    system_content += format_user_context(state)
    system_content += _photo_attachment_note(catalog, mentioned_product_ids)

    faqs = state.get("faqs") or []
    if faqs:
        faq_lines = [f"- P: {f['question']}\n  R: {f['answer']}" for f in faqs[:5]]
        system_content += "\n\nPreguntas frecuentes:\n" + "\n".join(faq_lines)

    history = [
        {"role": "user" if getattr(m, "type", "") == "human" else "assistant", "content": m.content}
        for m in state["messages"][-6:]
        if hasattr(m, "content") and m.content
    ]
    messages_payload = [{"role": "system", "content": system_content}] + history

    generation = trace.generation(name="menu_inquiry_llm", model=model, input={"messages": messages_payload})
    write = get_stream_writer()
    full_response = ""
    async for chunk in provider.stream_chat(model=model, messages=messages_payload):
        write({"type": "token", "content": chunk})
        full_response += chunk
    generation.end(output=full_response)

    return {
        "messages": [AIMessage(content=full_response)],
        "mentioned_product_ids": mentioned_product_ids,
        "turn_usage": [
            make_usage_record(node="menu_inquiry", provider=provider, model=model)
        ],
    }
