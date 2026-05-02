
from typing import Annotated, List, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from ..schemas.intent import StructuredIntent


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    thread_id: str

    # ── Conversation context (stable per conversation) ─────────────────
    tenant_id: str
    conversation_id: str
    product_catalog: list    # [{product_id, name, description, price, stock}]
    knowledge: Optional[List[dict]] = None # Unified knowledge payload
    user_context: dict       # {name, email, phone, address}
    contact_id: str
    contact_tags: list       # [{"id", "name", "color"}]
    language: str            # "en" | "es" | "pt"

    # ── Triage ─────────────────────────────────────────────────────────
    intent: str              # "sales" | "tracking" | "complaint" | "faq"
    structured_intent: Optional[StructuredIntent] = None

    # ── Sales — explicit phase machine ─────────────────────────────────
    #
    # Lifecycle:
    #   product_selection → product_confirmation → customer_data → payment
    #
    # Transitions are set exclusively by node code, never by the LLM.
    sales_phase: str         # see lifecycle above; default "product_selection"

    # Cart — authoritative state owned by CartService
    cart: Optional[list] = None # Mirror of the backend cart
    cart_confirmed: bool = False

    # Anti-loop counter: incremented each turn in PRODUCT_SELECTION.
    # When it reaches MAX_PRODUCT_TURNS the phase advances regardless
    # of whether the user has explicitly said "done".
    product_selection_turns: int

    # Items the ProductResolver could not map to a catalog product_id.
    # Cleared on each turn and rebuilt from the latest extraction.
    pending_unknown_items: list  # [{name, qty, alternatives}]

    # Phase completion flags (set by node code, read by routing edges)
    product_selection_complete: bool  # True → advance to product_confirmation
    customer_data_complete: bool      # True → advance to order_summary (payment)

    # Customer / delivery data collected in CUSTOMER_DATA phase
    # Keys: customer_name, customer_phone, customer_email,
    #       delivery_address, delivery_date_preference, payment_method
    order_data: dict

    # Legacy sales fields (kept for backward compat with order_summary + execute)
    order_confirmed: bool   # final confirmation in order_summary

    # ── Tracking flow ──────────────────────────────────────────────────
    tracking_data: dict
    tracking_complete: bool

    # ── Complaint flow ─────────────────────────────────────────────────
    complaint_data: dict
    complaint_complete: bool

    # ── Deals ──────────────────────────────────────────────────────────
    deal_created: bool

    # ── Observability ──────────────────────────────────────────────────
    faq_used: Optional[List[dict]] = None
    last_command_key: Optional[str] = None

    # ── Execution ──────────────────────────────────────────────────────
    execute_confirmed: bool
