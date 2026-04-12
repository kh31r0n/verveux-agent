"""
Lightweight async client for internal NestJS cart/orders endpoints.
Uses X-Agent-Key header (same WEBHOOK_API_KEY shared between services).
"""

import structlog
from httpx import AsyncClient, HTTPStatusError, RequestError

from ..config import settings

logger = structlog.get_logger(__name__)

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
) -> dict:
    """POST /internal/carts/:contactId/items — add/update item (quantity=0 removes it)."""
    params = {}
    if conversation_id:
        params["conversationId"] = conversation_id
    return await _post(
        f"/api/v1/internal/carts/{contact_id}/items",
        json={"productId": product_id, "quantity": quantity},
        params=params,
    )


async def get_order_history(contact_id: str, limit: int = 5) -> list:
    """GET /internal/orders?contactId=... — returns recent orders."""
    data = await _get("/api/v1/internal/orders", params={"contactId": contact_id, "limit": limit})
    return data if isinstance(data, list) else []


async def checkout_cart(contact_id: str, conversation_id: str | None = None) -> dict:
    """POST /internal/orders/checkout — converts active cart into an order."""
    body: dict = {"contactId": contact_id}
    if conversation_id:
        body["conversationId"] = conversation_id
    return await _post("/api/v1/internal/orders/checkout", json=body)


# ─── HTTP helpers ─────────────────────────────────────────────────────────────


async def _get(path: str, params: dict | None = None) -> dict | list:
    url = f"{_BASE}{path}"
    try:
        async with AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=_HEADERS, params=params)
            resp.raise_for_status()
            return resp.json()
    except HTTPStatusError as exc:
        logger.error("backend_get_error", path=path, status=exc.response.status_code)
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
        logger.error("backend_post_error", path=path, status=exc.response.status_code)
        raise
    except RequestError as exc:
        logger.error("backend_post_network_error", path=path, error=str(exc))
        raise
