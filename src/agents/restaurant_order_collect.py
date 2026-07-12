"""Restaurant order collection node — extraction → cart operations → reply.

Items are resolved against the tenant catalog (ProductResolver) and stored in
the shared `cart` state channel via CartService, mirroring sales_collect, so
the backend cart/checkout machinery works unchanged. `restaurant_order_data`
holds only the order-level fields (service_type / delivery_address /
special_notes).
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
from ..services.product_resolver import ProductResolver
from ..usage import make_usage_record
from .cart_sync import sync_full_cart_to_backend
from .utils import (
    format_user_context,
    language_instruction,
    latest_user_text,
    resolve_persona,
    resolve_prompt,
)

logger = structlog.get_logger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """Eres un asistente de extracción de datos de pedidos de restaurante.

Analiza el mensaje del usuario y devuelve un JSON con:
- "items": lista de platillos mencionados, cada uno con:
    - "name":      nombre del platillo tal como lo dijo el usuario (string)
    - "quantity":  cantidad numérica (int, default 1)
    - "operation": "add" | "remove" | "update_quantity"
    - "notes":     preparación o nota especial del ítem (string o "")
- "service_type": "delivery" o "pickup" (solo si el usuario lo indica; "a domicilio" = delivery, "para llevar"/"recoger" = pickup)
- "delivery_address": dirección de entrega (solo si la menciona)
- "special_notes": notas generales del pedido (solo si las menciona)

Para los items, usa los nombres exactos del menú/catálogo si coinciden.
Omite campos no mencionados.
Responde SOLO con el objeto JSON — sin markdown, sin explicación."""

_CONVERSATIONAL_SYSTEM_PROMPT = """Eres {persona}, una asistente de pedidos por WhatsApp para un restaurante.
Eres amable y eficiente. {language_rule}

Tu tarea: ayudar al cliente a armar su pedido de comida.
Necesitas recopilar:
- Platillos seleccionados con cantidad (items)
- Tipo de servicio: para llevar o a domicilio (service_type)
- Si es a domicilio: dirección de entrega (delivery_address)

Pedido actual:
{cart_summary}

Campos ya recopilados: {collected_fields}
Campos faltantes: {missing_fields}

{unresolved_block}

{catalog_info}

Instrucción:
{instruction}

Reglas:
- Sé muy breve — es WhatsApp.
- Confirma cada item agregado y sugiere complementos (bebidas, postres) naturalmente.
- Si hay platillos no encontrados, muestra las alternativas numeradas para que el cliente elija.
- NO devuelvas JSON — esta es una respuesta en lenguaje natural."""

# Spanish/English phrasings the extraction LLM may echo back verbatim.
_SERVICE_TYPE_MAP = {
    "delivery": "delivery",
    "domicilio": "delivery",
    "a domicilio": "delivery",
    "entrega": "delivery",
    "pickup": "pickup",
    "para llevar": "pickup",
    "llevar": "pickup",
    "recoger": "pickup",
    "recojo": "pickup",
}


def _normalize_service_type(value) -> str | None:
    if not isinstance(value, str):
        return None
    return _SERVICE_TYPE_MAP.get(value.strip().lower())


def _format_catalog(catalog: list) -> str:
    if not catalog:
        return ""
    items = []
    for item in catalog[:30]:
        line = f"- {item.get('name', 'Item')}"
        if item.get("price"):
            line += f" — ${item['price']}"
        items.append(line)
    return "Menú disponible:\n" + "\n".join(items)


def _build_unresolved_block(unresolved_items: list) -> str:
    parts = []
    for u in unresolved_items:
        if u.get("alternatives"):
            alt_str = "\n".join(
                f"  {i + 1}. {a['name']} — ${a['price']:.2f}"
                for i, a in enumerate(u["alternatives"])
            )
            parts.append(
                f"❓ No encontré *{u['name']}* en el menú. ¿Quisiste decir alguno de estos?\n{alt_str}"
            )
        else:
            parts.append(f"❓ No encontré *{u['name']}* en el menú.")
    return "\n\n".join(parts)


async def restaurant_order_collect_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("restaurant_order_collect")

    provider = get_provider(config)
    model = resolve_model(config)
    thread_id: str = state.get("thread_id", "unknown")
    contact_id: str = state.get("contact_id", "")
    conversation_id: str = state.get("conversation_id", "")
    catalog: list = state.get("product_catalog") or []

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="restaurant_order_collect_node",
        metadata={"thread_id": thread_id, "node": "restaurant_order_collect"},
    )

    # Fresh start after a completed checkout: the previous order is immutable,
    # so a new "order" intent begins with an empty cart. (Backend-safe: the
    # checked-out cart is CHECKED_OUT; the next upsert opens a fresh one.)
    starting_fresh = bool(state.get("execute_confirmed"))
    if starting_fresh:
        cart: list = []
        order_data: dict = {}
        logger.info("restaurant_order_fresh_start", thread_id=thread_id)
    else:
        cart = normalize_cart(state.get("cart"))
        order_data = dict(state.get("restaurant_order_data") or {})

    turn_usage: list = []
    unresolved_items: list = []
    backend_sync_ok = True

    # ── Stage 1: Extraction ──────────────────────────────────────────────────
    # Order items and delivery details often arrive split across several
    # rapid WhatsApp messages — extract from the whole trailing burst.
    last_user_msg = latest_user_text(state)

    if last_user_msg:
        extraction_prompt = resolve_prompt(
            config, "RESTAURANT_ORDER_EXTRACTION", _EXTRACTION_SYSTEM_PROMPT, state
        )
        catalog_str = "\n".join(
            f"- {p.get('name', 'N/A')}" for p in catalog
        ) or "Sin menú disponible."
        extraction_messages = [
            {"role": "system", "content": extraction_prompt},
            {
                "role": "user",
                "content": (
                    f"Menú disponible:\n{catalog_str}\n\n"
                    f"Mensaje del usuario:\n{last_user_msg}"
                ),
            },
        ]

        gen = trace.generation(name="restaurant_order_extraction", model=model, input={"messages": extraction_messages})
        extracted_text = ""
        async for chunk in provider.stream_chat(model=model, messages=extraction_messages):
            extracted_text += chunk
        turn_usage.append(
            make_usage_record(
                node="restaurant_order_collect.extraction",
                provider=provider,
                model=model,
            )
        )
        gen.end(output=extracted_text)

        extracted_items: list = []
        try:
            extracted = json.loads(strip_json_fences(extracted_text))
            extracted_items = extracted.get("items") or []
            service_type = _normalize_service_type(extracted.get("service_type"))
            if service_type:
                order_data["service_type"] = service_type
            if extracted.get("delivery_address"):
                order_data["delivery_address"] = str(extracted["delivery_address"]).strip()
            if extracted.get("special_notes"):
                order_data["special_notes"] = str(extracted["special_notes"]).strip()
        except (json.JSONDecodeError, TypeError, AttributeError):
            logger.warning(
                "restaurant_order_collect_extraction_parse_failed",
                thread_id=thread_id,
                raw=extracted_text[:200],
            )

        # ── Stage 2: Resolve dish names → catalog ids, apply cart ops ─────────
        if extracted_items:
            resolver = ProductResolver(catalog)
            resolved_items, unresolved_items = await resolver.resolve_many(
                extracted_items, provider=provider, model=model, usage_sink=turn_usage,
            )
            logger.info(
                "restaurant_product_resolution_done",
                thread_id=thread_id,
                resolved=len(resolved_items),
                unresolved=len(unresolved_items),
            )

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

            if resolved_items and contact_id:
                backend_sync_ok = await sync_full_cart_to_backend(
                    cart=cart,
                    contact_id=contact_id,
                    conversation_id=conversation_id or None,
                    thread_id=thread_id,
                )

    # ── Stage 3: Check completion ────────────────────────────────────────────
    cart_has_items = not CartService.is_empty(cart)
    service_type_set = bool(order_data.get("service_type"))
    needs_address = (
        order_data.get("service_type") == "delivery"
        and not order_data.get("delivery_address")
    )

    collected = []
    if cart_has_items:
        collected.append("items")
    if service_type_set:
        collected.append(f"service_type={order_data['service_type']}")
    if order_data.get("delivery_address"):
        collected.append(f"delivery_address={order_data['delivery_address']}")

    missing = []
    if not cart_has_items:
        missing.append("items")
    if not service_type_set:
        missing.append("service_type")
    if needs_address:
        missing.append("delivery_address")

    # Unresolved dishes hold the flow open so the user can pick an alternative
    # before the summary auto-chains.
    is_complete = not missing and not unresolved_items

    # ── Stage 4: Conversational response ─────────────────────────────────────
    if is_complete:
        instruction = (
            "Ya tienes todos los datos del pedido. Dile brevemente al cliente que "
            "a continuación verá el resumen para confirmar. No repitas el pedido completo."
        )
    elif unresolved_items:
        instruction = (
            "Algunos platillos no fueron encontrados (ver bloque de no encontrados). "
            "Muestra las alternativas numeradas y pregunta si desea alguna o si continúa sin ellos."
        )
    elif not cart_has_items:
        instruction = (
            "El pedido está vacío. Pregunta al cliente qué platillos desea; "
            "muestra el menú si ayuda."
        )
    else:
        faltan = ", ".join(missing)
        instruction = (
            f"Confirma lo agregado al pedido y pregunta por los datos faltantes: {faltan}."
        )

    lang_rule = language_instruction(state.get("language", "en"))
    conv_prompt = resolve_prompt(
        config, "RESTAURANT_ORDER_CONVERSATIONAL", _CONVERSATIONAL_SYSTEM_PROMPT, state
    )

    system_content = conv_prompt.format(
        persona=resolve_persona(state, "Giulia"),
        language_rule=lang_rule,
        cart_summary=CartService.format_cart(cart),
        collected_fields=", ".join(collected) or "Ninguno aún",
        missing_fields=", ".join(missing) if missing else "Ninguno — pedido completo",
        unresolved_block=_build_unresolved_block(unresolved_items),
        catalog_info=_format_catalog(catalog) if not cart_has_items else "",
        instruction=instruction,
    )
    system_content += format_user_context(state)

    history = [
        {"role": "user" if getattr(m, "type", "") == "human" else "assistant", "content": m.content}
        for m in state["messages"][-6:]
        if hasattr(m, "content") and m.content
    ]
    messages_payload = [{"role": "system", "content": system_content}] + history

    gen2 = trace.generation(name="restaurant_order_conversational", model=model, input={"messages": messages_payload})
    write = get_stream_writer()
    full_response = ""
    async for chunk in provider.stream_chat(model=model, messages=messages_payload):
        write({"type": "token", "content": chunk})
        full_response += chunk
    turn_usage.append(
        make_usage_record(
            node="restaurant_order_collect.reply", provider=provider, model=model,
        )
    )
    gen2.end(output=full_response)

    result = {
        "messages": [AIMessage(content=full_response)],
        "cart": cart,
        "restaurant_order_data": order_data,
        "restaurant_order_complete": is_complete,
        "restaurant_phase": "collect",
        "pending_unknown_items": unresolved_items,
        "backend_cart_sync_failed": not backend_sync_ok,
        "turn_usage": turn_usage,
    }
    if starting_fresh:
        result["execute_confirmed"] = False
        result["restaurant_order_confirmed"] = False
    return result
