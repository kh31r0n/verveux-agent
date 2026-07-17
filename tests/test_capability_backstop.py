"""CATALOG backstop on the Python side (T-P3 / T-P4).

Defense in depth: even if a turn's flag said "allowed" but the backend 403s
(stale race), the internal-endpoint layer raises CapabilityDisabledError and the
node degrades gracefully — a polite deflection, no retry, no raw error, and
crucially order_history never claims the customer has no orders.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from langchain_core.messages import HumanMessage

from src.agents.backend_client import CapabilityDisabledError, _capability_disabled


def _http_error(status: int, body: dict | str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://x/api/v1/internal/orders")
    if isinstance(body, dict):
        response = httpx.Response(status, json=body, request=request)
    else:
        response = httpx.Response(status, text=body, request=request)
    return httpx.HTTPStatusError("err", request=request, response=response)


class TestCapabilityDisabledParsing:
    def test_403_with_code_maps(self):
        exc = _http_error(403, {"code": "CAPABILITY_DISABLED", "capability": "CATALOG"})
        assert _capability_disabled(exc) == "CATALOG"

    def test_other_403_does_not_map(self):
        # e.g. the inquiries codeName-mismatch 403 — must stay HTTPStatusError.
        exc = _http_error(403, {"message": "Forbidden agentType"})
        assert _capability_disabled(exc) is None

    def test_non_403_does_not_map(self):
        exc = _http_error(409, {"code": "CAPABILITY_DISABLED"})
        assert _capability_disabled(exc) is None

    def test_non_json_body_does_not_crash(self):
        exc = _http_error(403, "plain text")
        assert _capability_disabled(exc) is None


class TestGetRaisesTypedError:
    async def test_get_translates_403_to_capability_error(self):
        from src.agents import backend_client

        async def _raise(*a, **k):
            raise _http_error(403, {"code": "CAPABILITY_DISABLED", "capability": "CATALOG"})

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=_raise)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(backend_client, "AsyncClient", return_value=mock_client):
            with pytest.raises(CapabilityDisabledError):
                await backend_client.get_order_history("c1", conversation_id="conv1")


class TestOrderHistoryBackstop:
    async def test_403_deflects_and_does_not_claim_no_orders(self):
        # T-P3: flag allowed, backend denies -> degraded reply, no "no orders".
        from src.agents.order_history import order_history_node

        state = {
            "messages": [HumanMessage(content="¿dónde está mi pedido?")],
            "thread_id": "t:helena:1",
            "language": "es",
            "contact_id": "c1",
            "conversation_id": "conv1",
            "catalog_access_enabled": True,  # flag says OK; backend will 403
        }

        fetch = AsyncMock(
            side_effect=CapabilityDisabledError("CATALOG", "/api/v1/internal/orders")
        )
        events: list[dict] = []
        writer = lambda e: events.append(e)  # noqa: E731
        with (
            patch("src.agents.order_history.get_order_history", fetch),
            patch("src.agents.order_history.get_provider", return_value=object()),
            patch("src.agents.order_history.resolve_model", return_value="m"),
            patch("src.agents.order_history.get_stream_writer", return_value=writer),
            patch("src.agents.capability_gate.get_stream_writer", return_value=writer),
        ):
            result = await order_history_node(state, {"configurable": {}})

        reply = result["messages"][0].content
        assert "negocio" in reply.lower()
        assert "no se encontraron pedidos" not in reply.lower()
        assert "no tienes pedidos" not in reply.lower()
        # No retry — the backend was called exactly once.
        assert fetch.await_count == 1


class TestExecuteBackstop:
    async def test_checkout_403_deflects_no_raw_error(self):
        from src.agents.execute import execute_node

        state = {
            "messages": [HumanMessage(content="confirmar")],
            "thread_id": "t:helena:1",
            "language": "es",
            "conversation_id": "conv1",
            "contact_id": "c1",
            "intent": "sales",
            "catalog_access_enabled": True,
        }

        checkout = AsyncMock(
            side_effect=CapabilityDisabledError("CATALOG", "/api/v1/internal/orders/checkout")
        )
        events: list[dict] = []
        writer = lambda e: events.append(e)  # noqa: E731
        with (
            patch("src.agents.execute.checkout_cart", checkout),
            patch("src.agents.execute.get_stream_writer", return_value=writer),
            patch("src.agents.capability_gate.get_stream_writer", return_value=writer),
        ):
            result = await execute_node(state, {"configurable": {}})

        assert checkout.await_count == 1
        assert result["execute_confirmed"] is True
        assert "negocio" in result["messages"][0].content.lower()
        # No execute_workflow error event leaked the raw exception.
        assert not any(e.get("error") for e in events)
