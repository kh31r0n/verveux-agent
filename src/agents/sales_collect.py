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
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation
from ..services.cart import CartService
from ..services.product_resolver import ProductResolver
from .utils import format_user_context, language_instruction
from .backend_client import upsert_cart_item

logger = structlog.get_logger(__name__)

MAX_PRODUCT_TURNS = 3

# ── Extraction prompt ────────────────────────────────────────────────────────
# The LLM is responsible for understanding the user's intent and converting it
# into a structured list.  ALL business logic lives in the node code below.

_EXTRACTION_SYSTEM_PROMPT = """Eres un extractor de intención de compra.

Analiza el mensaje del usuario y devuelve un JSON con:
- "items": lista de productos mencionados, cada uno con:
    - "name":           nombre del producto tal como lo dijo el usuario (string)
    - "quantity":       cantidad numérica (int, default 1)
    - "operation":      "add" | "remove" | "update_quantity" | "replace"
    - "old_product_id": solo si operation="replace", el nombre del producto a reemplazar (string o null)
    - "notes":          nota adicional del usuario sobre este ítem (string o "")
- "user_done_signal":  true si el usuario indica que terminó de agregar productos
    (frases como "eso es todo", "nada más", "listo", "ya está", "es todo")

Responde SOLO con el objeto JSON. Sin markdown, sin explicación.
Ejemplo:
{
  "items": [
    {"name": "Arroz integral", "quantity": 2, "operation": "add", "old_product_id": null, "notes": ""}
  ],
  "user_done_signal": false
}
"""

# ── Conversational prompt ─────────────────────────────────────────────────────

_CONV_SYSTEM_PROMPT = """Eres Helena, asistente de ventas por WhatsApp. Eres amable y concisa.
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


async def sales_collect_node(state: AgentState, config: RunnableConfig) -> dict:
    record_node_invocation("sales_collect")

    api_key = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="sales_collect",
        metadata={"thread_id": thread_id, "node": "sales_collect"},
    )

    write = get_stream_writer()
    write({"type": "step_progress", "step": 1, "total_steps": 1, "topic": "Selección de productos"})

    # ── Load current state ────────────────────────────────────────────────────
    cart: list = list(state.get("cart") or [])
    turns: int = int(state.get("product_selection_turns") or 0)
    catalog: list = state.get("product_catalog") or []
    has_new_message = bool(state["messages"]) and getattr(state["messages"][-1], "type", "") == "human"

    resolved_items: list[dict] = []
    unresolved_items: list[dict] = []
    user_done_signal = False

    # ── Step 1: Extract intent from user message ──────────────────────────────
    if has_new_message:
        turns += 1
        catalog_str = "\n".join(
            f"- {p.get('name', 'N/A')}: ${p.get('price', 0):.2f} (stock: {p.get('stock', 'N/A')})"
            for p in catalog
        ) or "Sin catálogo disponible."

        extraction_messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Catálogo disponible:\n{catalog_str}\n\n"
                    f"Mensaje del usuario:\n{state['messages'][-1].content}"
                ),
            },
        ]

        extraction_gen = trace.generation(
            name="product_extraction_llm",
            model="gpt-5.4-nano",
            input={"messages": extraction_messages},
        )

        extraction_stream = await client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=extraction_messages,
            stream=True,
        )
        extraction_raw = ""
        async for chunk in extraction_stream:
            delta = chunk.choices[0].delta.content if chunk.choices else ""
            if delta:
                extraction_raw += delta
        extraction_gen.end(output=extraction_raw)

        try:
            parsed = json.loads(extraction_raw.strip())
            extracted_items: list[dict] = parsed.get("items") or []
            user_done_signal = bool(parsed.get("user_done_signal", False))
        except (json.JSONDecodeError, ValueError):
            logger.warning("sales_collect_extraction_parse_failed", raw=extraction_raw[:200])
            extracted_items = []

        # ── Step 2: Resolve product names → catalog ids ───────────────────────
        if extracted_items:
            resolver = ProductResolver(catalog)
            resolved_items, unresolved_items = await resolver.resolve_many(
                extracted_items, client=client
            )
            logger.info(
                "product_resolution_done",
                thread_id=thread_id,
                resolved=len(resolved_items),
                unresolved=len(unresolved_items),
            )

        # ── Step 3: Apply CartService operations ──────────────────────────────
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
            
            # Sync with the backend
            contact_id = state.get("contact_id")
            conversation_id = state.get("conversation_id")
            if contact_id:
                try:
                    if item["operation"] == "remove":
                        await upsert_cart_item(contact_id, item["product_id"], 0, conversation_id)
                    elif item["operation"] == "replace" and item.get("old_product_id"):
                        await upsert_cart_item(contact_id, item.get("old_product_id"), 0, conversation_id)
                        new_item = CartService.find_item(cart, item["product_id"])
                        if new_item:
                            await upsert_cart_item(contact_id, item["product_id"], new_item["qty"], conversation_id)
                    else:
                        new_item = CartService.find_item(cart, item["product_id"])
                        qty_to_sync = new_item["qty"] if new_item else 0
                        await upsert_cart_item(contact_id, item["product_id"], qty_to_sync, conversation_id)
                except Exception as e:
                    logger.error("backend_cart_sync_error", error=str(e), product_id=item["product_id"])

            logger.info(
                "cart_operation_applied",
                thread_id=thread_id,
                operation=item["operation"],
                product_id=item["product_id"],
                qty=item["qty"],
            )

    # ── Step 4: Decide phase transition ───────────────────────────────────────
    cart_has_items = not CartService.is_empty(cart)
    turn_limit_reached = turns >= MAX_PRODUCT_TURNS
    advance = cart_has_items and (user_done_signal or turn_limit_reached)

    if turn_limit_reached and not cart_has_items:
        # Edge case: user burned all turns without adding anything valid
        advance = False
        logger.warning("sales_collect_turn_limit_no_items", thread_id=thread_id)

    product_selection_complete = advance
    sales_phase = "product_confirmation" if advance else "product_selection"

    # ── Step 5: Build conversational response ──────────────────────────────────
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

    # Catalog block (only show when cart is empty or has few items to help the user)
    catalog_block = ""
    if not cart_has_items and catalog:
        catalog_lines = "\n".join(
            f"- {p.get('name', 'N/A')} — ${p.get('price', 0):.2f}"
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
                state["messages"][-1].content
                if has_new_message
                else "Hola, quiero hacer un pedido."
            ),
        },
    ]

    conv_gen = trace.generation(
        name="sales_collect_conv_llm",
        model="gpt-5.4-nano",
        input={"messages": conv_messages},
    )

    conv_stream = await client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=conv_messages,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_response = ""
    prompt_tokens = 0
    completion_tokens = 0

    async for chunk in conv_stream:
        delta = chunk.choices[0].delta.content if chunk.choices else ""
        if delta:
            write({"type": "token", "content": delta})
            full_response += delta
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    conv_gen.end(output=full_response, usage={"input": prompt_tokens, "output": completion_tokens})

    logger.info(
        "sales_collect_turn_done",
        thread_id=thread_id,
        turns=turns,
        cart_size=len(cart),
        advance=advance,
        sales_phase=sales_phase,
    )

    return {
        "messages": [AIMessage(content=full_response)],
        "cart": cart,
        "product_selection_turns": turns,
        "pending_unknown_items": unresolved_items,
        "product_selection_complete": product_selection_complete,
        "sales_phase": sales_phase,
    }
