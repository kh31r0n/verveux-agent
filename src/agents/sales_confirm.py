"""
sales_confirm — PRODUCT_CONFIRMATION phase.

Responsibility:
  Present the confirmed cart to the user and collect an explicit yes/no.

Design rules:
  - The routing decision (yes / no) is DETERMINISTIC keyword matching.
    The LLM is only used to compose the response text.
  - On YES:  cart_confirmed = True,  sales_phase = "customer_data"
  - On NO:   cart_confirmed = False, sales_phase = "product_selection",
             product_selection_complete = False, product_selection_turns = 0
             (cart is preserved so the user can modify it, not start from scratch)
  - On first visit (no user message yet): just show the cart summary.
"""

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation
from ..services.cart import CartService
from .utils import language_instruction

logger = structlog.get_logger(__name__)

# ── Deterministic keyword sets ────────────────────────────────────────────────
# We intentionally keep these broad — a user who says "todo bien" or "va" should
# be understood as a confirmation without requiring an LLM interpretation.

_YES = {
    "sí", "si", "yes", "confirmar", "confirmo", "ok", "okay", "dale",
    "listo", "perfecto", "adelante", "procede", "correcto", "exacto",
    "todo bien", "está bien", "va", "continuar", "continua", "aceptar",
}

_NO = {
    "no", "nope", "cancelar", "cancel", "cambiar", "modificar",
    "quiero cambiar", "espera", "para", "detente", "volver", "regresar",
    "editar", "corregir",
}

# ── Prompts ──────────────────────────────────────────────────────────────────

_PRESENT_PROMPT = """Eres Helena, asistente de ventas por WhatsApp. {language_rule}

El usuario acaba de terminar de seleccionar sus productos.
Presenta el siguiente resumen de carrito de forma clara y amigable,
y pregunta explícitamente si desea confirmar o modificar algo.

Usa exactamente este resumen (no lo cambies):
{cart_summary}

Al final, agrega:
"¿Confirmamos el pedido con estos productos? Responde *Sí* para continuar o *No* para hacer cambios."

Sé breve. No agregues más información.
"""

_CONFIRMED_PROMPT = """Eres Helena, asistente de ventas por WhatsApp. {language_rule}

El usuario acaba de confirmar su carrito.
Responde de forma breve y positiva: confirma que los productos están listos
y dile que ahora necesitas sus datos de entrega.
No repitas el resumen del carrito.
"""

_REJECTED_PROMPT = """Eres Helena, asistente de ventas por WhatsApp. {language_rule}

El usuario quiere hacer cambios en su carrito.
Responde amablemente: dile que con gusto puedes modificar los productos
y pregunta qué desea cambiar.
No repitas el resumen del carrito.
"""

_UNCLEAR_PROMPT = """Eres Helena, asistente de ventas por WhatsApp. {language_rule}

El usuario respondió algo que no es claramente sí o no.

Carrito actual:
{cart_summary}

Pide amablemente que confirme con "Sí" o "No" si desea continuar con estos productos.
Sé muy breve.
"""


async def sales_confirm_node(state: AgentState, config: RunnableConfig) -> dict:
    record_node_invocation("sales_confirm")

    api_key = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id = state.get("thread_id", "unknown")
    lang = state.get("language", "es")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="sales_confirm",
        metadata={"thread_id": thread_id, "node": "sales_confirm"},
    )

    write = get_stream_writer()
    cart: list = list(state.get("cart") or [])
    cart_summary = CartService.format_cart(cart)

    has_new_message = (
        bool(state["messages"])
        and getattr(state["messages"][-1], "type", "") == "human"
    )

    # ── Deterministic decision ────────────────────────────────────────────────
    cart_confirmed: bool | None = None   # True=yes, False=no, None=unclear/first visit

    if has_new_message:
        user_text = (state["messages"][-1].content or "").lower().strip()
        # Tokenise loosely: split on spaces and check individual words too
        tokens = set(user_text.split())

        if any(kw in user_text or kw in tokens for kw in _YES):
            cart_confirmed = True
        elif any(kw in user_text or kw in tokens for kw in _NO):
            cart_confirmed = False
        # else: ambiguous — cart_confirmed stays None

    # ── Compose response text via LLM ─────────────────────────────────────────
    lang_rule = language_instruction(lang)

    if cart_confirmed is True:
        system_content = _CONFIRMED_PROMPT.format(language_rule=lang_rule)
    elif cart_confirmed is False:
        system_content = _REJECTED_PROMPT.format(language_rule=lang_rule)
    elif has_new_message:
        # Ambiguous user reply
        system_content = _UNCLEAR_PROMPT.format(
            language_rule=lang_rule, cart_summary=cart_summary
        )
    else:
        # First visit to this node — present the cart
        system_content = _PRESENT_PROMPT.format(
            language_rule=lang_rule, cart_summary=cart_summary
        )

    messages_payload = [{"role": "system", "content": system_content}]
    if has_new_message:
        messages_payload.append(
            {"role": "user", "content": state["messages"][-1].content}
        )

    gen = trace.generation(
        name="sales_confirm_llm",
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
            write({"type": "token", "content": delta})
            full_response += delta
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    gen.end(output=full_response, usage={"input": prompt_tokens, "output": completion_tokens})

    # ── Build state update ────────────────────────────────────────────────────
    update: dict = {"messages": [AIMessage(content=full_response)]}

    if cart_confirmed is True:
        update["cart_confirmed"] = True
        update["sales_phase"] = "customer_data"
        logger.info("sales_confirm_accepted", thread_id=thread_id, cart_size=len(cart))

    elif cart_confirmed is False:
        # Reset product selection so the user can modify the cart
        update["cart_confirmed"] = False
        update["sales_phase"] = "product_selection"
        update["product_selection_complete"] = False
        update["product_selection_turns"] = 0
        # Cart is intentionally preserved — user modifies, not restarts
        logger.info("sales_confirm_rejected", thread_id=thread_id)

    # cart_confirmed is None → stay in product_confirmation, no state changes

    return update
