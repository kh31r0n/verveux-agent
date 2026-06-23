import operator
from typing import Annotated, List, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from ..schemas.intent import StructuredIntent
from ..usage import InvocationUsage


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    thread_id: str

    # ── Multi-agent identity ──────────────────────────────────────────
    agent_type: str              # "sales" | "school" | "restaurant" | "appointments"
    # Permanent code name of the graph topology (e.g. "helena", "sofia").
    # Set from the request body; optional so existing checkpoints written
    # before the migration deserialize cleanly.
    agent_code_name: str
    # Position of the AgentAssignment in the connection's history (1-indexed).
    # Snapshotted on the conversation at creation; immutable.
    agent_version: int
    capabilities: dict           # capability contract from NestJS (read-only)
    domain_state: dict           # generic bag for domain-specific data (prevents state pollution)

    # ── Conversation context (stable per conversation) ─────────────────
    tenant_id: str
    conversation_id: str
    product_catalog: list    # [{product_id, name, description, price, stock}]
    knowledge: Optional[List[dict]]  # Unified knowledge payload
    user_context: dict       # {name, email, phone, address}
    contact_id: str
    contact_tags: list       # [{"id", "name", "color"}]
    language: str            # "en" | "es" | "pt"

    # ── FAQs — injected per-request, NOT persisted across turns ────────
    # Schema: [{question: str, answer: str, category: str, priority: int}]
    # Populated from rawFaqs in the NestJS request; used by triage and
    # faq_response in the current turn only. Intentionally overwritten
    # on every request so stale FAQ data never leaks between turns.
    faqs: list

    # ── Triage ─────────────────────────────────────────────────────────
    intent: str              # "sales" | "tracking" | "complaint" | "faq"
    structured_intent: Optional[StructuredIntent]

    # ── Sales — explicit phase machine ─────────────────────────────────
    #
    # Lifecycle:
    #   product_selection → product_confirmation → customer_data → payment
    #
    # Transitions are set exclusively by node code, never by the LLM.
    sales_phase: str         # see lifecycle above; default "product_selection"

    # Cart — authoritative state owned by CartService
    cart: Optional[list]     # Mirror of the backend cart
    cart_confirmed: bool

    # backend_cart_sync_failed: set by sales_collect when the full-cart
    # sync to the backend fails after retries. order_summary uses this
    # flag to fall back to the state cart instead of fetching from the backend.
    backend_cart_sync_failed: bool

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

    # ── School flow ───────────────────────────────────────────────────
    admissions_data: dict
    admissions_complete: bool

    # ── Restaurant flow ───────────────────────────────────────────────
    restaurant_order_data: dict
    restaurant_order_complete: bool

    # ── Appointments flow ─────────────────────────────────────────────
    booking_data: dict
    booking_complete: bool
    booking_confirmed: bool

    # ── Deals ──────────────────────────────────────────────────────────
    deal_created: bool

    # ── Observability ──────────────────────────────────────────────────
    faq_used: Optional[List[dict]]
    last_command_key: Optional[str]

    # ── Execution ──────────────────────────────────────────────────────
    execute_confirmed: bool

    # ── Token usage ────────────────────────────────────────────────────
    # Append-only list: each node adds one entry per provider call. The
    # operator.add reducer concatenates partial state updates, so multiple
    # nodes (e.g. sales_collect → order_summary in a single turn) all
    # contribute without overwriting. Forwarded to NestJS on the SSE
    # `done` event, then persisted as AiInvocationUsage rows.
    turn_usage: Annotated[List[InvocationUsage], operator.add]

    # ── Camila (academic-secretary) fields ─────────────────────────────
    # Non-text content carried by the inbound message. Populated per-turn
    # from the NestJS request; intentionally overwritten each turn so
    # stale attachments don't bleed across turns. Camila escalates to
    # human as soon as this list is non-empty.
    attachments: list
    # Latched True once the contact's name has been parsed and persisted
    # to the backend by name_capture. Prevents the graph from re-asking.
    school_name_captured: bool
    # Free-text reason recorded when handoff runs; mirrors the value sent
    # to /internal/conversations/:id/handoff so the audit trail stays
    # consistent between graph state and the backend Incident record.
    handoff_reason: Optional[str]