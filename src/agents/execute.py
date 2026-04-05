import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..observability import record_node_invocation
from .backend_client import checkout_cart

logger = structlog.get_logger(__name__)


async def execute_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("execute")

    thread_id: str = state.get("thread_id", "unknown")
    conversation_id: str = state.get("conversation_id", "")
    intent: str = state.get("intent", "")
    contact_id: str = state.get("contact_id", "")

    write = get_stream_writer()

    if intent == "sales" and contact_id:
        # Checkout the backend cart into an immutable order
        try:
            order = await checkout_cart(
                contact_id=contact_id,
                conversation_id=conversation_id or None,
            )
            logger.info(
                "cart_checked_out",
                thread_id=thread_id,
                order_id=order.get("id"),
                total=order.get("grandTotal"),
            )
            write({
                "type": "execute_workflow",
                "conversation_id": conversation_id,
                "intent": intent,
                "order_id": order.get("id"),
                "order_data": {
                    "id": order.get("id"),
                    "status": order.get("status"),
                    "grandTotal": order.get("grandTotal"),
                    "currency": order.get("currency"),
                    "items": order.get("items", []),
                },
            })
        except Exception as exc:
            logger.error("checkout_failed", thread_id=thread_id, error=str(exc))
            write({
                "type": "execute_workflow",
                "conversation_id": conversation_id,
                "intent": intent,
                "error": str(exc),
            })
    else:
        # Tracking and complaint flows — emit as before
        payload: dict = {
            "type": "execute_workflow",
            "conversation_id": conversation_id,
            "intent": intent,
        }
        if intent == "tracking":
            payload["tracking_data"] = dict(state.get("tracking_data") or {})
        elif intent == "complaint":
            payload["complaint_data"] = dict(state.get("complaint_data") or {})

        write(payload)

    logger.info(
        "execute_triggered",
        thread_id=thread_id,
        conversation_id=conversation_id,
        intent=intent,
    )

    return {
        "messages": [],
        "execute_confirmed": True,
    }
