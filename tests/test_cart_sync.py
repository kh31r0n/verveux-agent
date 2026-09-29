"""Tests for backend cart synchronisation retry semantics."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.agents.cart_sync import sync_full_cart_to_backend

CONTACT = "contact-1"
CONVERSATION = "conv-1"
THREAD = "thread-1"
CART = [{"product_id": "prod-1", "qty": 1}]


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://backend/api/v1/internal/carts/c/items")
    response = httpx.Response(status, request=request, text="Insufficient stock")
    return httpx.HTTPStatusError("boom", request=request, response=response)


async def test_4xx_does_not_retry():
    upsert = AsyncMock(side_effect=_http_error(400))
    with patch("src.agents.cart_sync.upsert_cart_item", new=upsert):
        ok = await sync_full_cart_to_backend(CART, CONTACT, CONVERSATION, THREAD)

    assert ok is False
    # Deterministic 4xx — attempted exactly once, no retry.
    assert upsert.call_count == 1


async def test_5xx_retries_to_limit():
    upsert = AsyncMock(side_effect=_http_error(503))
    with patch("src.agents.cart_sync.upsert_cart_item", new=upsert):
        ok = await sync_full_cart_to_backend(CART, CONTACT, CONVERSATION, THREAD)

    assert ok is False
    # Transient 5xx — retried up to the retry limit (2 attempts).
    assert upsert.call_count == 2


async def test_generic_exception_retries():
    upsert = AsyncMock(side_effect=RuntimeError("network glitch"))
    with patch("src.agents.cart_sync.upsert_cart_item", new=upsert):
        ok = await sync_full_cart_to_backend(CART, CONTACT, CONVERSATION, THREAD)

    assert ok is False
    assert upsert.call_count == 2


async def test_success_syncs_once():
    upsert = AsyncMock(return_value={})
    with patch("src.agents.cart_sync.upsert_cart_item", new=upsert):
        ok = await sync_full_cart_to_backend(CART, CONTACT, CONVERSATION, THREAD)

    assert ok is True
    assert upsert.call_count == 1
