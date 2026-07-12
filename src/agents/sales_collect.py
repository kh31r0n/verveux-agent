"""
sales_collect — PRODUCT_SELECTION phase.

Responsibility:
  Collect product items from the user and build a consistent cart.

Design rules enforced here:
  - The LLM extracts *intent* only (items + signals). It never writes to cart.
  - CartService applies all mutations and returns a new cart list.
  - ProductResolver maps free text → product_id (fuzzy → LLM fallback).
  - The backend (this node code) decides when the phase ends, NOT the LLM.
  - MAX_PRODUCT_TURNS prevents infinite loops even for indecisive users.

Backend sync strategy:
  - Individual item operations are applied to the in-memory cart via CartService.
  - After ALL operations succeed locally, the FULL cart is synced to the backend
    in a single pass (each item's final quantity is upserted).
  - This replaces the fragile item-by-item sync that left the backend in a
    partial state when any single upsert failed.
  - If the full sync fails, a warning is logged and `backend_cart_sync_failed`
    is set in state so order_summary can fall back to the state cart.

Phase transition:
  Sets sales_phase = "product_confirmation" and product_selection_complete = True
  when EITHER:
    (a) user_done_signal is True AND cart is not empty, OR
    (b) product_selection_turns >= MAX_PRODUCT_TURNS AND cart is not empty.
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
from ..services.product_resolver import ProductResolver
from .utils import format_user_context, language_instruction, latest_user_text, resolve_persona
from .cart_sync import sync_full_cart_to_backend

logger = structlog.get_logger(__name__)

MAX_PRODUCT_TURNS = 3


def _format_money(value) -> str:
    """Render a catalog price as ``X.YY``. Defensive — older payloads sometimes
    arrived with the price coerced into a Decimal-shaped dict ({s,e,d}) by the
    backend's nulls-to-empty-strings pass; falling back to ``0.00`` here keeps
    the catalog string usable instead of crashing the whole turn."""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "0.00"

# ── Extraction prompt ────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM_PROMPT = """Eres un extractor de intención de compra.

Analiza el mensaje del usuario y devuelve un JSON con:
- "items": lista de productos mencionados, cada uno con:
    - "name":           nombre del producto tal como lo dijo el usuario (string)
    - "quantity":       cantidad numérica (int, default 1)
    - "operation":      "add" | "remove" | "update_quantity" | "replace"
    - "old_product_id": solo si operation="replace", el nombre del producto a reemplazar (string o null)
    - "notes":          nota adicional del usuario sobre este ítem (string o "")
- "referenced_product_names": lista de nombres de productos sobre los que el usuario
    está preguntando o hablando, AUNQUE NO los esté agregando al carrito
    (ej. "¿cuánto cuesta el X?", "muéstrame el Y", "háblame del Z",
    "qué tiene el W"). Incluye también los nombres que aparecen en "items".
    Si no menciona ningún producto específico, devuelve [].
- "user_done_signal":  true si el usuario indica que terminó de agregar productos
    (frases como "eso es todo", "nada más", "listo", "ya está", "es todo")

Responde SOLO con el objeto JSON. Sin markdown, sin explicación.
Ejemplo:
{
  "items": [
    {"name": "Arroz integral", "quantity": 2, "operation": "add", "old_product_id": null, "notes": ""}
  ],
  "referenced_product_names": ["Arroz integral"],
  "user_done_signal": false
}
"""

# ── Conversational prompt ─────────────────────────────────────────────────────

_CONV_SYSTEM_PROMPT = """Eres {persona}, asistente de ventas por WhatsApp. Eres amable y concisa.
{language_rule}

Estás ayudando al usuario a armar su carrito de compras.

Estado actual del carrito:
{cart_summary}

{unresolved_block}

{catalog_block}

Instrucción:
{instruction}

Reglas:
- Sé muy breve — es WhatsApp.
- Si hay productos no encontrados, muestra las alternativas disponibles numeradas para que el usuario elija.
- NO devuelvas JSON.
- NO confirmes el pedido en este paso; eso se hace por separado.
"""


def _compute_mentioned_product_ids(
    resolved_items: list[dict],
    referenced_names: list[str],
    catalog: list[dict],
) -> list[str]:
    """
    Returns ``[product_id]`` when exactly one distinct catalog product is
    referenced this turn, otherwise ``[]``.

    Combines two signals:
      * Stage-1 cart-op items already resolved to product_ids.
      * "referenced_product_names" — products the user asked or talked
        about, even without a cart op (e.g. "¿cuánto cuesta el X?").

    Names are resolved via heuristic-only ``ProductResolver.resolve`` (no
    LLM fallback) — this is a best-effort signal for image attachment, not
    an authoritative cart operation, so the extra latency/cost of an LLM
    call would be disproportionate.
    """
    if not resolved_items and not referenced_names:
        return []

    pids: set[str] = set()

    for item in resolved_items:
        pid = item.get("product_id")
        if pid:
            pids.add(pid)

    if referenced_names:
        resolver = ProductResolver(catalog)
        for name in referenced_names:
            if not isinstance(name, str) or not name.strip():
                continue
            result = resolver.resolve(name)
            if result.resolved:
                pids.add(result.resolved.product_id)

    return [next(iter(pids))] if len(pids) == 1 else []


async def sales_collect_node(state: AgentState, config: RunnableConfig) -> dict:
    record_node_invocation("sales_collect")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="sales_collect",
        metadata={"thread_id": thread_id, "node": "sales_collect"},
    )

    write = get_stream_writer()
    write({"type": "step_progress", "step": 1, "total_steps": 1, "topic": "Selección de productos"})

    # ── Load current state ────────────────────────────────────────────────────
    cart: list = normalize_cart(state.get("cart"))
    turns: int = int(state.get("product_selection_turns") or 0)
    catalog: list = state.get("product_catalog") or []
    contact_id: str = state.get("contact_id", "")
    conversation_id: str = state.get("conversation_id", "")
    has_new_message = bool(state["messages"]) and getattr(state["messages"][-1], "type", "") == "human"
    # The whole trailing burst of user messages — users often split one order
    # across several rapid WhatsApp messages, and each must be considered.
    user_turn_text = latest_user_text(state)

    resolved_items: list[dict] = []
    unresolved_items: list[dict] = []
    user_done_signal = False
    backend_sync_ok = True  # optimistic until proven otherwise
    turn_usage: list = []   # populated after each provider call below
    referenced_names: list[str] = []  # raw names the user asked about

    # ── Step 1: Extract intent from user message ──────────────────────────────
    if has_new_message:
        turns += 1
        catalog_str = "\n".join(
            f"- {p.get('name', 'N/A')}: ${_format_money(p.get('price', 0))} (stock: {p.get('stock', 'N/A')})"
            for p in catalog
        ) or "Sin catálogo disponible."

        extraction_messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Catálogo disponible:\n{catalog_str}\n\n"
                    f"Mensaje del usuario:\n{user_turn_text}"
                ),
            },
        ]

        extraction_gen = trace.generation(
            name="product_extraction_llm",
            model=model,
            input={"messages": extraction_messages},
        )

        extraction_stream = provider.stream_chat(
            model=model,
            messages=extraction_messages,
        )
        extraction_raw = ""
        async for chunk in extraction_stream:
            extraction_raw += chunk
        turn_usage.append(
            make_usage_record(
                node="sales_collect.extraction", provider=provider, model=model,
            )
        )
        extraction_gen.end(output=extraction_raw)

        try:
            parsed = json.loads(strip_json_fences(extraction_raw))
            extracted_items: list[dict] = parsed.get("items") or []
            raw_refs = parsed.get("referenced_product_names") or []
            referenced_names = [r for r in raw_refs if isinstance(r, str)]
            user_done_signal = bool(parsed.get("user_done_signal", False))
        except (json.JSONDecodeError, ValueError):
            logger.warning("sales_collect_extraction_parse_failed", raw=extraction_raw[:200])
            extracted_items = []

        # ── Step 2: Resolve product names → catalog ids ───────────────────────
        if extracted_items:
            resolver = ProductResolver(catalog)
            resolved_items, unresolved_items = await resolver.resolve_many(
                extracted_items, provider=provider, model=model, usage_sink=turn_usage,
            )
            logger.info(
                "product_resolution_done",
                thread_id=thread_id,
                resolved=len(resolved_items),
                unresolved=len(unresolved_items),
            )

        # ── Step 3: Apply CartService operations (pure in-memory) ─────────────
        #
        # CartService is the single source of truth for cart mutations.
        # No backend calls happen here — we do ONE full sync after all
        # operations are applied locally (Step 4 below).
        for item in resolved_items:
            cart = CartService.apply_operation(
                cart=cart,
                operation=item["operation"],
                product_id=item["product_id"],
                name=item["name"],
                qty=item["qty"],
                price=item["price"],
                old_product_id=item.get("old_product_id"),
                notes=item.get("notes", ""),
            )
            logger.info(
                "cart_operation_applied",
                thread_id=thread_id,
                operation=item["operation"],
                product_id=item["product_id"],
                qty=item["qty"],
            )

        # ── Step 4: Full cart sync to backend ─────────────────────────────────
        #
        # Sync the ENTIRE cart after all operations succeed locally.
        # This avoids partial backend state from item-by-item upserts that
        # could fail independently and leave the backend inconsistent.
        if resolved_items and contact_id:
            backend_sync_ok = await sync_full_cart_to_backend(
                cart=cart,
                contact_id=contact_id,
                conversation_id=conversation_id or None,
                thread_id=thread_id,
            )

    # ── Step 5: Decide phase transition ───────────────────────────────────────
    cart_has_items = not CartService.is_empty(cart)
    turn_limit_reached = turns >= MAX_PRODUCT_TURNS
    advance = cart_has_items and (user_done_signal or turn_limit_reached)

    if turn_limit_reached and not cart_has_items:
        advance = False
        logger.warning("sales_collect_turn_limit_no_items", thread_id=thread_id)

    product_selection_complete = advance
    sales_phase = "product_confirmation" if advance else "product_selection"

    # ── Step 6: Build conversational response ──────────────────────────────────
    cart_summary = CartService.format_cart(cart)

    # Unresolved block
    unresolved_block = ""
    if unresolved_items:
        parts = []
        for u in unresolved_items:
            if u["alternatives"]:
                alt_str = "\n".join(
                    f"  {i + 1}. {a['name']} — ${a['price']:.2f}"
                    for i, a in enumerate(u["alternatives"])
                )
                parts.append(f"❓ No encontré *{u['name']}*. ¿Quisiste decir alguno de estos?\n{alt_str}")
            else:
                parts.append(f"❓ No encontré *{u['name']}* en el catálogo.")
        unresolved_block = "\n\n".join(parts)

    # Catalog block (only show when cart is empty to help the user)
    catalog_block = ""
    if not cart_has_items and catalog:
        catalog_lines = "\n".join(
            f"- {p.get('name', 'N/A')} — ${_format_money(p.get('price', 0))}"
            for p in catalog[:10]
        )
        catalog_block = f"📦 *Catálogo disponible:*\n{catalog_lines}"

    # Instruction for the conversational LLM
    if advance and not unresolved_items:
        instruction = (
            "El usuario terminó de seleccionar productos. "
            "Resume brevemente el carrito y dile que a continuación revisaremos el resumen para confirmar. "
            "No pidas confirmación todavía."
        )
    elif advance and unresolved_items:
        instruction = (
            "Se acabaron los intentos para agregar productos. "
            "Informa al usuario que pasaremos a confirmar con los productos que ya están en el carrito, "
            "y que los productos no encontrados no serán incluidos. Sé empático."
        )
    elif not cart_has_items:
        instruction = (
            "El carrito está vacío. Saluda al usuario, indica que estás listo para tomar su pedido "
            "y pregunta qué productos desea. "
            + (f"Hay {turns} de {MAX_PRODUCT_TURNS} turnos utilizados." if turns > 0 else "")
        )
    elif unresolved_items:
        instruction = (
            "Algunos productos no fueron encontrados (mostrados en 'unresolved_block'). "
            "Muestra las alternativas disponibles y pregunta si el usuario quiere alguna de ellas "
            "o si desea continuar sin esos productos."
        )
    else:
        turns_left = MAX_PRODUCT_TURNS - turns
        instruction = (
            f"Se agregaron productos al carrito exitosamente. "
            f"Confirma lo que se agregó y pregunta si quiere algo más. "
            f"Menciona que puede decir 'listo' o 'eso es todo' cuando termine. "
            f"({'último turno disponible' if turns_left <= 1 else f'{turns_left} turnos restantes'})"
        )

    conv_messages = [
        {
            "role": "system",
            "content": _CONV_SYSTEM_PROMPT.format(
                persona=resolve_persona(state, "Helena"),
                language_rule=language_instruction(state.get("language", "es")),
                cart_summary=cart_summary,
                unresolved_block=unresolved_block,
                catalog_block=catalog_block,
                instruction=instruction,
            ) + format_user_context(state),
        },
        {
            "role": "user",
            "content": (
                user_turn_text
                if has_new_message
                else "Hola, quiero hacer un pedido."
            ),
        },
    ]

    conv_gen = trace.generation(
        name="sales_collect_conv_llm",
        model=model,
        input={"messages": conv_messages},
    )

    conv_stream = provider.stream_chat(
        model=model,
        messages=conv_messages,
    )

    full_response = ""

    async for chunk in conv_stream:
        write({"type": "token", "content": chunk})
        full_response += chunk

    turn_usage.append(
        make_usage_record(
            node="sales_collect.reply", provider=provider, model=model,
        )
    )

    conv_gen.end(output=full_response)

    logger.info(
        "sales_collect_turn_done",
        thread_id=thread_id,
        turns=turns,
        cart_size=len(cart),
        advance=advance,
        sales_phase=sales_phase,
        backend_sync_ok=backend_sync_ok,
    )

    mentioned_product_ids = _compute_mentioned_product_ids(
        resolved_items=resolved_items,
        referenced_names=referenced_names,
        catalog=catalog,
    )

    return {
        "messages": [AIMessage(content=full_response)],
        "cart": cart,
        "product_selection_turns": turns,
        "pending_unknown_items": unresolved_items,
        "product_selection_complete": product_selection_complete,
        "sales_phase": sales_phase,
        # Signals to order_summary that it must use state cart as fallback
        "backend_cart_sync_failed": not backend_sync_ok,
        "turn_usage": turn_usage,
        # Surfaced to NestJS via the SSE `done` event so it can decide
        # whether to attach a product image to the outbound WhatsApp reply.
        "mentioned_product_ids": mentioned_product_ids,
    }