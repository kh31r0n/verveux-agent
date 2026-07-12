import operator
from typing import Annotated, List, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

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
    # Custom persona name the agent uses to introduce itself on this channel
    # (e.g. "Helena", "Admisiones"). Empty = use the graph's default identity.
    agent_persona_name: str
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
    # StructuredIntent serialized via model_dump(mode="json") — checkpoints
    # must only carry JSON-native values so LangGraph can deserialize them
    # without allow-listing custom Python classes. Legacy checkpoints written
    # before 2026-07 may still yield StructuredIntent instances (allow-listed
    # in main.py); readers handle both shapes.
    structured_intent: Optional[dict]

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

    # Product ids the SALES reply is "about" this turn. Holds exactly one id
    # when a single catalog product was referenced (info query or cart op);
    # empty when zero or >1 distinct products were referenced. NestJS reads
    # this off the SSE `done` event and attaches the product image to the
    # outbound WhatsApp message only on the single-id case.
    mentioned_product_ids: list  # list[str]

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
    # restaurant_order_data holds ONLY order-level fields collected in
    # conversation (service_type, delivery_address, special_notes) — the
    # items themselves live in `cart` (shared with the sales flow) so the
    # backend cart/checkout machinery works unchanged.
    restaurant_order_data: dict
    # True when cart is non-empty + service_type set (+ address if delivery).
    restaurant_order_complete: bool
    # "collect" | "confirmation". The summary node latches "confirmation"
    # so the next turn resumes at restaurant_confirm instead of re-collecting.
    # Readers must use .get() — old checkpoints lack the key.
    restaurant_phase: str
    # Set by restaurant_confirm on a clear yes; routes the same turn to
    # execute (checkout). Reset when a new order starts post-checkout.
    restaurant_order_confirmed: bool

    # ── Appointments flow ─────────────────────────────────────────────
    # Legacy fields (kept for backward compat with existing checkpoints
    # that may have been written by the partial graph) and the new
    # lifecycle fields used by the restructured appointments graph.
    booking_data: dict
    booking_complete: bool
    booking_confirmed: bool

    # New lifecycle. `booking_intent` records what the user is here to do
    # so subsequent turns can rejoin the flow mid-conversation without
    # re-triage. Values: "book" | "reschedule" | "cancel" | "inquire".
    booking_intent: Optional[str]
    # Selected AppointmentType. Resolved by appointment_collect from the
    # tenant's catalogue (loaded once per turn).
    appointment_type_id: Optional[str]
    appointment_type_name: Optional[str]
    appointment_type_duration_min: Optional[int]
    # Schema snapshot of the type's required customer fields. Drives the
    # validation loop in appointment_collect.
    required_customer_fields: List[dict]
    # Values collected to satisfy required_customer_fields, keyed by `key`.
    collected_customer_data: dict
    # Candidate slots returned by availability_lookup (capped server-side).
    # Schema: [{startsAt, endsAt, resources: [{kind, resourceId, name}], score}]
    candidate_slots: List[dict]
    # User-selected slot from candidate_slots (set by reservation_propose).
    chosen_slot: Optional[dict]
    # IDs returned by /internal/appointments/holds; populated after confirm.
    reservation_appointment_id: Optional[str]
    hold_expires_at: Optional[str]
    # For cancel / reschedule disambiguation (set by appointment_cancel /
    # appointment_reschedule when the user has multiple active bookings).
    cancellation_target_id: Optional[str]
    reschedule_source_id: Optional[str]
    # Sentinel set when the backend rejects a hold with code SLOT_TAKEN so
    # the graph can route back to availability_lookup and re-search.
    slot_conflict: bool

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
    # LEGACY latch (camila-only, superseded by `name_captured` below).
    # Honoured read-only by shared_routing.has_name so existing camila
    # checkpoints never re-ask; new captures set both flags.
    school_name_captured: bool

    # ── Name capture (generic, all graphs) ─────────────────────────────
    # Latched once a name is confirmed persisted by the backend (or adopted
    # from a MANUAL human-edited name on applied=false). NOT set on backend
    # HTTP failure — next turn the backend snapshot stays authoritative.
    name_captured: bool
    # Extraction attempts without success this thread; reaching
    # MAX_NAME_CAPTURE_ATTEMPTS triggers an implicit deferral so the agent
    # never nags indefinitely.
    name_capture_attempts: int
    # Local mirror of Contact.nameCaptureDeferredAt, set when the user
    # declines. `user_context.name_capture_deferred` (backend snapshot,
    # re-sent every turn with window expiry applied) is authoritative on
    # subsequent turns.
    name_capture_deferred: bool
    # Per-turn flag ALWAYS written by name_capture_node: True → the node
    # already replied (ask or greeting), end the turn; False → continue to
    # the graph's normal intent routing in the same turn.
    name_capture_reply_sent: bool
    # Free-text reason recorded when handoff runs; mirrors the value sent
    # to /internal/conversations/:id/handoff so the audit trail stays
    # consistent between graph state and the backend Incident record.
    handoff_reason: Optional[str]

    # ── Query normalization (per-turn; sole writer: query_normalizer) ──
    # Snapshot of the user's turn text before any typo correction. Stamped
    # every turn by query_normalizer_node. Readers must use .get() — old
    # checkpoints (and turns where the node was bypassed) lack the key.
    original_text: Optional[str]
    # Write-once provenance for the turn's normalization decision. Shape:
    #   {enabled: bool, model: str|None, confidence: float|None,
    #    changed_meaning_risk: "LOW"|"MEDIUM"|"HIGH"|None, reason: str,
    #    applied: bool, corrected_text: str|None}
    # JSON-native values only (checkpoint rule). None = node did not run
    # (legacy checkpoint). Downstream nodes treat this as read-only.
    normalization: Optional[dict]