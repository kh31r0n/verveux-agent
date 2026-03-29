import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..observability import record_node_invocation

logger = structlog.get_logger(__name__)


async def execute_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("execute")

    thread_id: str = state.get("thread_id", "unknown")
    conversation_id: str = state.get("conversation_id", "")
    intent: str = state.get("intent", "")

    write = get_stream_writer()

    # Build payload based on intent
    payload: dict = {
        "type": "execute_workflow",
        "conversation_id": conversation_id,
        "intent": intent,
    }

    if intent == "sales":
        payload["order_data"] = dict(state.get("order_data") or {})
    elif intent == "tracking":
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
