import json

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation
from .utils import format_user_context

logger = structlog.get_logger(__name__)

# Each step covers a group of order fields.
_STEP_CONFIGS = [
    {
        "step": 1,
        "topic": "Datos del cliente",
        "fields": ["customer_name", "customer_phone", "customer_email"],
        "questions": (
            "Para procesar tu pedido necesito algunos datos:\n"
            "1. ¿Cuál es tu nombre completo?\n"
            "2. ¿Cuál es tu número de teléfono de contacto?\n"
            "3. ¿Tienes un correo electrónico? (opcional)"
        ),
    },
    {
        "step": 2,
        "topic": "Productos del pedido",
        "fields": ["items"],
        "questions": (
            "¿Qué productos te gustaría ordenar?\n"
            "Puedes indicarme el nombre del producto y la cantidad.\n"
            "Si necesitas ver nuestro catálogo, con gusto te lo muestro."
        ),
    },
    {
        "step": 3,
        "topic": "Entrega y pago",
        "fields": ["delivery_address", "delivery_date_preference", "payment_method"],
        "questions": (
            "Para finalizar el pedido:\n"
            "1. ¿Cuál es la dirección de entrega?\n"
            "2. ¿Tienes preferencia de fecha de entrega?\n"
            "3. ¿Cuál será el método de pago? (efectivo, transferencia, tarjeta)"
        ),
    },
]

_EXTRACTION_SYSTEM_PROMPT = """Eres un asistente de extracción de datos de pedidos.

Se te dará:
1. Las preguntas que se hicieron al usuario en este paso
2. Los campos ya recopilados
3. El mensaje más reciente del usuario
4. El catálogo de productos disponibles (si aplica)

Tu tarea: extraer las respuestas del usuario y mapearlas a los nombres de campo correctos.
Devuelve un objeto JSON con los nombres de campo como claves y las respuestas del usuario como valores.
Si un campo no fue respondido, omítelo del JSON.

Para el campo "items", devuelve una lista de objetos: [{"product": "nombre", "quantity": número, "notes": "notas opcionales"}]
Si el usuario menciona un producto del catálogo, usa el nombre exacto del catálogo.

Responde SOLO con el objeto JSON — sin markdown, sin explicación.
"""

_CONVERSATIONAL_SYSTEM_PROMPT = """Eres Helena, una asistente de ventas por WhatsApp para una tienda de productos físicos.
Eres amable, eficiente y siempre respondes en español.

Estás en el paso {step} de 3 del proceso de pedido, recopilando: {topic}.

Tu tarea:
1. Hacer las preguntas de este paso de forma amigable y conversacional.
2. Si el usuario ya proporcionó algunas respuestas, reconócelas y pregunta solo por las faltantes.
3. Si todas las respuestas del paso están completas, confirma y avisa que pasarás al siguiente paso.

Campos faltantes: {missing_fields}
Campos recopilados: {collected_fields}

{catalog_info}

Reglas:
- Sé concisa y amigable — es una conversación por WhatsApp.
- NO preguntes por campos de otros pasos.
- NO devuelvas JSON — esta es una respuesta en lenguaje natural.
"""


async def sales_collect_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("sales_collect")

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id: str = state.get("thread_id", "unknown")
    order_data: dict = dict(state.get("order_data") or {})
    sales_step: int = state.get("sales_step", 0)

    # Determine current step config (1-3)
    current_step_idx = max(0, min(sales_step, 2))  # clamp to 0-2 for indexing
    step_config = _STEP_CONFIGS[current_step_idx]
    step_num = step_config["step"]
    fields = step_config["fields"]
    topic = step_config["topic"]

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="sales_collect",
        metadata={"thread_id": thread_id, "node": "sales_collect", "step": step_num},
    )

    write = get_stream_writer()

    # --- Step 1: Emit progress event ---
    write({"type": "step_progress", "step": step_num, "total_steps": 3, "topic": topic})

    # --- Step 2: Extract answers from latest user message ---
    has_new_message = bool(state["messages"]) and getattr(state["messages"][-1], "type", "") == "human"
    extracted_fields: dict = {}

    if has_new_message:
        already_collected = {k: order_data[k] for k in fields if order_data.get(k)}
        still_missing = [f for f in fields if not order_data.get(f)]

        # Build catalog context for extraction
        catalog = state.get("product_catalog") or []
        catalog_str = ""
        if catalog:
            catalog_str = "\n\nCatálogo de productos:\n" + "\n".join(
                f"- {p.get('name', 'N/A')}: {p.get('description', '')} — ${p.get('price', 'N/A')} (stock: {p.get('stock', 'N/A')})"
                for p in catalog
            )

        extraction_messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Paso {step_num} — {topic}\n\n"
                    f"Preguntas hechas al usuario:\n{step_config['questions']}\n\n"
                    f"Campos ya recopilados (no repetir): {list(already_collected.keys())}\n"
                    f"Campos faltantes: {still_missing}\n\n"
                    f"Mensaje del usuario:\n{state['messages'][-1].content}"
                    f"{catalog_str}"
                ),
            },
        ]

        extraction_gen = trace.generation(
            name="sales_extraction_llm",
            model="gpt-5",
            input={"messages": extraction_messages},
        )

        extraction_stream = await client.chat.completions.create(
            model="gpt-5",
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
            extracted_fields = json.loads(extraction_raw.strip())
            if not isinstance(extracted_fields, dict):
                extracted_fields = {}
        except (json.JSONDecodeError, ValueError):
            logger.warning("sales_extraction_parse_failed", step=step_num, raw=extraction_raw[:200])
            extracted_fields = {}

        # Merge into order_data
        order_data.update({k: v for k, v in extracted_fields.items() if v})
        logger.info("sales_fields_extracted", thread_id=thread_id, step=step_num, fields=list(extracted_fields.keys()))

    # --- Step 3: Determine missing fields for this step ---
    missing_fields = [f for f in fields if not order_data.get(f)]
    collected_fields = {f: order_data[f] for f in fields if order_data.get(f)}

    # --- Step 4: If all fields for this step are collected, advance ---
    step_complete = len(missing_fields) == 0
    new_sales_step = sales_step
    sales_complete = state.get("sales_complete", False)

    if step_complete and has_new_message:
        new_sales_step = min(sales_step + 1, 3)
        if new_sales_step >= 3:
            sales_complete = True
            logger.info("sales_collect_complete", thread_id=thread_id)

        # Emit update_deal_stage when advancing from step 0 → 1 (Lead → Prospecto)
        if sales_step == 0 and new_sales_step == 1:
            contact_id: str = state.get("contact_id", "")
            if contact_id:
                write({
                    "type": "update_deal_stage",
                    "contact_id": contact_id,
                    "conversation_id": state.get("conversation_id", ""),
                    "stage_position": 1,
                })
                logger.info("sales_deal_stage_advanced", thread_id=thread_id, stage_position=1)

    # --- Step 5: Generate conversational response ---
    collected_summary = ", ".join(f"{k}={repr(v)}" for k, v in collected_fields.items()) or "ninguno aún"
    missing_summary = ", ".join(missing_fields) if missing_fields else "todos respondidos"

    user_ctx_str = format_user_context(state)

    # Build catalog info for conversational prompt
    catalog = state.get("product_catalog") or []
    catalog_info = ""
    if catalog and step_num == 2:
        catalog_info = "Catálogo disponible:\n" + "\n".join(
            f"- {p.get('name', 'N/A')}: ${p.get('price', 'N/A')}"
            for p in catalog
        )

    if step_complete and sales_complete:
        # Transition message to order summary
        conv_prompt = (
            "Has terminado de recopilar toda la información del pedido en 3 pasos. "
            "Dile al usuario que has completado la recopilación y que ahora le presentarás "
            "un resumen del pedido para su confirmación. "
            "Sé breve y positiva."
        ) + user_ctx_str
        conv_messages = [
            {"role": "system", "content": conv_prompt},
        ]
    elif step_complete and has_new_message:
        # Step complete, moving to next step
        next_step_config = _STEP_CONFIGS[new_sales_step] if new_sales_step < 3 else None
        next_topic = next_step_config["topic"] if next_step_config else ""
        conv_prompt = (
            f"Has recopilado todos los datos del paso {step_num} ({topic}). "
            f"Confirma brevemente lo recopilado y pasa al paso {new_sales_step + 1}: {next_topic}. "
            f"Luego haz las preguntas del siguiente paso."
        ) + user_ctx_str
        next_questions = _STEP_CONFIGS[new_sales_step]["questions"] if new_sales_step < 3 else ""
        conv_messages = [
            {"role": "system", "content": conv_prompt},
            {"role": "assistant", "content": f"Preguntas del siguiente paso:\n{next_questions}"},
        ]
    else:
        # Ask/re-ask questions for current step
        conv_messages = [
            {
                "role": "system",
                "content": _CONVERSATIONAL_SYSTEM_PROMPT.format(
                    step=step_num,
                    topic=topic,
                    missing_fields=missing_summary,
                    collected_fields=collected_summary,
                    catalog_info=catalog_info,
                ) + user_ctx_str,
            },
            {
                "role": "user",
                "content": (
                    state["messages"][-1].content
                    if has_new_message
                    else f"Iniciar paso {step_num}: {_STEP_CONFIGS[current_step_idx]['questions']}"
                ),
            },
        ]

    conv_gen = trace.generation(
        name="sales_conversational_llm",
        model="gpt-5",
        input={"messages": conv_messages},
    )

    conv_stream = await client.chat.completions.create(
        model="gpt-5",
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

    conv_gen.end(
        output=full_response,
        usage={"input": prompt_tokens, "output": completion_tokens},
    )

    return {
        "messages": [AIMessage(content=full_response)],
        "order_data": order_data,
        "sales_step": new_sales_step,
        "sales_complete": sales_complete,
    }
