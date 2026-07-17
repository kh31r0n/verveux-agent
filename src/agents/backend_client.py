"""
Lightweight async client for internal NestJS cart/orders endpoints.
Uses X-Agent-Key header (same WEBHOOK_API_KEY shared between services).
"""

import structlog
from httpx import AsyncClient, HTTPStatusError, RequestError

from ..config import settings

logger = structlog.get_logger(__name__)


class CapabilityDisabledError(Exception):
    """An internal endpoint returned 403 with code CAPABILITY_DISABLED — the
    tenant disabled a capability (e.g. CATALOG) for this agent. Callers turn
    this into the degraded reply; it is never retried and never surfaced raw to
    the customer. Distinct from other 403s (e.g. the inquiries codeName-mismatch
    check) which stay HTTPStatusError."""

    def __init__(self, capability: str = "", path: str = ""):
        self.capability = capability
        self.path = path
        super().__init__(f"capability disabled: {capability or '?'} ({path})")


_BASE = settings.nestjs_base_url.rstrip("/")
_HEADERS = {
    "Content-Type": "application/json",
    "X-Agent-Key": settings.webhook_api_key,
}


async def get_or_create_cart(contact_id: str, conversation_id: str | None = None) -> dict:
    """GET /internal/carts/:contactId — returns or creates the active cart."""
    params = {}
    if conversation_id:
        params["conversationId"] = conversation_id
    return await _get(f"/api/v1/internal/carts/{contact_id}", params=params)


async def upsert_cart_item(
    contact_id: str,
    product_id: str,
    quantity: int,
    conversation_id: str | None = None,
    notes: str | None = None,
) -> dict:
    """POST /internal/carts/:contactId/items — add/update item (quantity=0 removes it)."""
    params = {}
    if conversation_id:
        params["conversationId"] = conversation_id
    body: dict = {"productId": product_id, "quantity": quantity}
    if notes:
        body["notes"] = notes
    return await _post(
        f"/api/v1/internal/carts/{contact_id}/items",
        json=body,
        params=params,
    )


async def get_order_history(
    contact_id: str,
    limit: int = 5,
    conversation_id: str | None = None,
) -> list:
    """GET /internal/orders?contactId=... — returns recent orders.

    ``conversation_id`` lets the backend resolve the per-agent CATALOG policy
    from the conversation snapshot; without it the tenant-default applies. May
    raise CapabilityDisabledError when catalog access is off (tracking flow)."""
    params: dict = {"contactId": contact_id, "limit": limit}
    if conversation_id:
        params["conversationId"] = conversation_id
    data = await _get("/api/v1/internal/orders", params=params)
    return data if isinstance(data, list) else []


async def checkout_cart(
    contact_id: str,
    conversation_id: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """POST /internal/orders/checkout — converts active cart into an order.

    `metadata` is persisted verbatim on Order.metadata (restaurant orders send
    serviceType / deliveryAddress / specialNotes).
    """
    body: dict = {"contactId": contact_id}
    if conversation_id:
        body["conversationId"] = conversation_id
    if metadata:
        body["metadata"] = metadata
    return await _post("/api/v1/internal/orders/checkout", json=body)


async def search_faqs(conversation_id: str, query: str, limit: int = 5) -> list:
    """GET /internal/faqs/search — conversation-scoped FAQ retrieval.

    Used by query_normalizer to re-run retrieval with a typo-corrected query.
    The backend resolves the agentCodeName from the conversation snapshot.
    Returns [{id, question, answer, category, score}]; [] on non-list.
    """
    data = await _get(
        "/api/v1/internal/faqs/search",
        params={"conversationId": conversation_id, "query": query, "limit": limit},
    )
    return data if isinstance(data, list) else []


async def fetch_agent_credentials(tenant_id: str) -> dict:
    """
    GET /api/v1/internal/agent/credentials?tenantId=...
    Returns the decrypted runtime credentials for the tenant's effective LLM provider.
    Response shape:
      { provider, model, apiKey? } for OpenAI/Anthropic
      { provider, model, vertexCredentials, vertexProjectId, vertexLocation } for Vertex
    """
    return await _get(
        "/api/v1/internal/agent/credentials",
        params={"tenantId": tenant_id},
    )


async def fetch_active_code_names() -> set[str]:
    """GET /api/v1/internal/agent-versioning/active-code-names — all platform-active code names.

    Used at startup to validate that this deployment's CODE_NAME_REGISTRY can serve
    every code name the backend considers active (invariant #5).
    """
    data = await _get("/api/v1/internal/agent-versioning/active-code-names")
    code_names = data.get("codeNames", []) if isinstance(data, dict) else []
    return {c for c in code_names if isinstance(c, str)}


async def fetch_in_use_code_names() -> list[str]:
    """GET /api/v1/internal/agent-versioning/in-use-code-names — code names currently
    assigned to at least one active, non-deleted channel connection. Used for startup
    warm-up so we only compile graphs that real traffic will hit.
    """
    data = await _get("/api/v1/internal/agent-versioning/in-use-code-names")
    code_names = data.get("codeNames", []) if isinstance(data, dict) else []
    return [c for c in code_names if isinstance(c, str)]


async def update_contact_name(contact_id: str, name: str, tenant_id: str) -> dict:
    """POST /internal/contacts/:contactId/name — persist the captured contact
    name (name_capture node, all graphs). Backend sanitizes, writes
    Contact.customName (unless a human set it: nameSource=MANUAL) and emits a
    realtime contact:updated event. Response: {ok, applied, name, contactId} —
    applied=false means a MANUAL name won; adopt `name` from the response."""
    return await _post(
        f"/api/v1/internal/contacts/{contact_id}/name",
        json={"name": name, "tenantId": tenant_id},
    )


async def defer_contact_name_capture(contact_id: str, tenant_id: str) -> dict:
    """POST /internal/contacts/:contactId/name-capture/defer — the customer
    declined to give their name. Backend stamps Contact.nameCaptureDeferredAt
    so no graph re-asks until the deferral window expires."""
    return await _post(
        f"/api/v1/internal/contacts/{contact_id}/name-capture/defer",
        json={"tenantId": tenant_id},
    )


async def submit_inquiry(
    tenant_id: str,
    conversation_id: str,
    idempotency_key: str,
    lead_data: dict,
) -> dict:
    """POST /internal/inquiries — persist a qualified lead (veronica).

    ``idempotency_key`` is the state-backed lead_submission_id: the backend
    holds a unique index on it, so retries after network failures (and
    LangGraph node replays) return the original Inquiry instead of inserting
    a duplicate.
    """
    body = {
        "idempotencyKey": idempotency_key,
        "conversationId": conversation_id,
        "tenantId": tenant_id,
        **{
            k: v
            for k, v in lead_data.items()
            if k
            in {
                "fullName",
                "email",
                "serviceInterest",
                "company",
                "phoneCountryCode",
                "phoneNumber",
                "challenge",
                "comments",
                "locale",
            }
            and v
        },
    }
    return await _post("/api/v1/internal/inquiries", json=body)


async def request_handoff(
    conversation_id: str,
    tenant_id: str,
    reason: str,
    intents: list[str] | None = None,
    has_attachments: bool = False,
) -> dict:
    """POST /internal/conversations/:conversationId/handoff — flip the
    conversation's aiEnabled to false and record an Incident. Idempotent."""
    return await _post(
        f"/api/v1/internal/conversations/{conversation_id}/handoff",
        json={
            "reason": reason,
            "tenantId": tenant_id,
            "intents": intents or [],
            "hasAttachments": has_attachments,
        },
    )


# ─── Appointments (/internal/appointments/*) ────────────────────────────────
#
# These call the NestJS internal controllers added in Phase 1. The agent never
# touches the Prisma database directly — all booking validation, the EXCLUDE
# constraint, and the audit trail stay on the backend.


async def list_appointment_types(tenant_id: str) -> list[dict]:
    """GET /internal/appointments/types?tenantId=... — active types catalog."""
    data = await _get(
        "/api/v1/internal/appointments/types",
        params={"tenantId": tenant_id},
    )
    return data if isinstance(data, list) else []


async def search_appointment_availability(
    tenant_id: str,
    appointment_type_id: str,
    from_iso: str,
    to_iso: str,
    location_id: str | None = None,
    resource_preferences: dict | None = None,
    timezone: str | None = None,
    limit: int = 10,
) -> dict:
    """POST /internal/appointments/availability — on-demand slot search."""
    body: dict = {
        "tenantId": tenant_id,
        "appointmentTypeId": appointment_type_id,
        "from": from_iso,
        "to": to_iso,
        "limit": limit,
    }
    if location_id:
        body["locationId"] = location_id
    if resource_preferences:
        body["resourcePreferences"] = resource_preferences
    if timezone:
        body["timezone"] = timezone
    return await _post("/api/v1/internal/appointments/availability", json=body)


async def create_appointment_hold(
    tenant_id: str,
    contact_id: str,
    appointment_type_id: str,
    starts_at_iso: str,
    ends_at_iso: str,
    resources: list[dict],
    customer_data: dict | None = None,
    location_id: str | None = None,
    hold_minutes: int = 10,
    thread_id: str | None = None,
    conversation_id: str | None = None,
) -> dict:
    """POST /internal/appointments/holds — create a PENDING_HOLD appointment.

    The backend enforces double-booking via the EXCLUDE constraint, so this
    call may raise HTTP 409 with ``code: SLOT_TAKEN``. The caller is
    responsible for catching that and offering an alternative slot.

    ``conversation_id`` links the resulting audit event to the originating
    WhatsApp conversation (BD-3) so the CRM can deep-link the booking.
    """
    body: dict = {
        "tenantId": tenant_id,
        "contactId": contact_id,
        "appointmentTypeId": appointment_type_id,
        "startsAt": starts_at_iso,
        "endsAt": ends_at_iso,
        "resources": resources,
        "holdMinutes": hold_minutes,
    }
    if location_id:
        body["locationId"] = location_id
    if customer_data:
        body["customerData"] = customer_data
    if thread_id:
        body["threadId"] = thread_id
    if conversation_id:
        body["conversationId"] = conversation_id
    return await _post("/api/v1/internal/appointments/holds", json=body)


async def confirm_appointment(
    appointment_id: str,
    tenant_id: str,
    conversation_id: str | None = None,
) -> dict:
    """POST /internal/appointments/:id/confirm — promote hold to CONFIRMED."""
    body: dict = {"tenantId": tenant_id}
    if conversation_id:
        body["conversationId"] = conversation_id
    return await _post(
        f"/api/v1/internal/appointments/{appointment_id}/confirm",
        json=body,
    )


async def cancel_appointment(
    appointment_id: str,
    tenant_id: str,
    reason: str | None = None,
    conversation_id: str | None = None,
) -> dict:
    """POST /internal/appointments/:id/cancel — cancel by id."""
    body: dict = {"tenantId": tenant_id}
    if reason:
        body["reason"] = reason
    if conversation_id:
        body["conversationId"] = conversation_id
    return await _post(
        f"/api/v1/internal/appointments/{appointment_id}/cancel", json=body
    )


async def reschedule_appointment(
    appointment_id: str,
    tenant_id: str,
    starts_at_iso: str,
    ends_at_iso: str,
    resources: list[dict],
    conversation_id: str | None = None,
) -> dict:
    """POST /internal/appointments/:id/reschedule — move to a new slot."""
    body: dict = {
        "tenantId": tenant_id,
        "startsAt": starts_at_iso,
        "endsAt": ends_at_iso,
        "resources": resources,
    }
    if conversation_id:
        body["conversationId"] = conversation_id
    return await _post(
        f"/api/v1/internal/appointments/{appointment_id}/reschedule",
        json=body,
    )


async def list_active_appointments_for_contact(
    contact_id: str,
    tenant_id: str,
) -> list[dict]:
    """GET /internal/appointments/contacts/:contactId/active — used by the
    cancel / reschedule flows to disambiguate which booking the user means.
    """
    data = await _get(
        f"/api/v1/internal/appointments/contacts/{contact_id}/active",
        params={"tenantId": tenant_id},
    )
    return data if isinstance(data, list) else []


# ─── HTTP helpers ─────────────────────────────────────────────────────────────


def _error_body(exc: HTTPStatusError) -> str:
    """Truncated response body for logging — surfaces backend error messages
    (e.g. 'Insufficient stock …') that the bare status code hides."""
    try:
        return exc.response.text[:500]
    except Exception:
        return ""


def _capability_disabled(exc: HTTPStatusError) -> str | None:
    """Return the disabled capability name if this is a CAPABILITY_DISABLED 403,
    else None. Keyed on the body `code` — NEVER on status alone, so unrelated
    403s (inquiries codeName mismatch) are not misclassified."""
    if exc.response.status_code != 403:
        return None
    try:
        body = exc.response.json()
    except Exception:
        return None
    if isinstance(body, dict) and body.get("code") == "CAPABILITY_DISABLED":
        return str(body.get("capability") or "")
    return None


async def _get(path: str, params: dict | None = None) -> dict | list:
    url = f"{_BASE}{path}"
    try:
        async with AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=_HEADERS, params=params)
            resp.raise_for_status()
            return resp.json()
    except HTTPStatusError as exc:
        capability = _capability_disabled(exc)
        if capability is not None:
            logger.info(
                "capability_block", capability=capability, source="backstop_403", path=path
            )
            raise CapabilityDisabledError(capability, path) from exc
        logger.error(
            "backend_get_error",
            path=path,
            status=exc.response.status_code,
            body=_error_body(exc),
        )
        raise
    except RequestError as exc:
        logger.error("backend_get_network_error", path=path, error=str(exc))
        raise


async def _post(path: str, json: dict | None = None, params: dict | None = None) -> dict:
    url = f"{_BASE}{path}"
    try:
        async with AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=_HEADERS, json=json, params=params)
            resp.raise_for_status()
            return resp.json()
    except HTTPStatusError as exc:
        capability = _capability_disabled(exc)
        if capability is not None:
            logger.info(
                "capability_block", capability=capability, source="backstop_403", path=path
            )
            raise CapabilityDisabledError(capability, path) from exc
        logger.error(
            "backend_post_error",
            path=path,
            status=exc.response.status_code,
            body=_error_body(exc),
        )
        raise
    except RequestError as exc:
        logger.error("backend_post_network_error", path=path, error=str(exc))
        raise
