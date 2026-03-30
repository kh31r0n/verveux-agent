from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    thread_id: str
    # Conversation context (stable for the lifetime of the conversation)
    project_id: str          # project linked to this conversation
    conversation_id: str     # NestJS conversation UUID (used by execute_workflow SSE event)
    product_catalog: list    # [{product_id, name, description, price, stock}] from NestJS
    user_context: dict       # {name, email, phone, address}
    contact_id: str          # NestJS contact UUID
    contact_tags: list       # [{"id", "name", "color"}] from NestJS — current tags on the contact
    # Triage
    intent: str  # "sales" | "tracking" | "complaint" | "faq"
    # Sales flow
    sales_step: int          # 0 = not started, 1-3 = collection steps
    order_data: dict         # accumulated order fields across all steps
    sales_complete: bool
    order_confirmed: bool
    # Tracking flow
    tracking_data: dict
    tracking_complete: bool
    # Complaint flow
    complaint_data: dict
    complaint_complete: bool
    # Deals
    deal_created: bool
    # Execution
    execute_confirmed: bool
