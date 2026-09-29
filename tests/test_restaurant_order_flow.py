"""Routing + wiring tests for the restaurant order lifecycle:
collect → summary (auto-chain) → [next turn] confirm → execute (auto-chain),
with the modify path re-entering collection in the same turn, plus the
triage skip-guard that keeps mid-order turns from being re-classified."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END

from src.agents.triage import triage_node
from src.graphs.restaurant_graph import (
    _route_from_confirm,
    _route_from_triage,
    build_restaurant_graph,
)

_CONFIG = {"configurable": {"thread_id": "test-user:rest-1", "openai_api_key": "sk-test"}}

_CART = [
    {"product_id": "p-tacos", "name": "Tacos al pastor", "qty": 2, "price": 12.5, "notes": ""},
]


def _base_state(**overrides) -> dict:
    state = {
        "messages": [HumanMessage(content="hola")],
        "intent": "order",
        "user_context": {"name": "Ana"},
        "agent_type": "restaurant",
    }
    state.update(overrides)
    return state


class TestTriageRouter:
    def test_fresh_order_goes_to_collect(self):
        assert _route_from_triage(_base_state()) == "order_collect"

    def test_confirmation_phase_resumes_at_confirm(self):
        state = _base_state(cart=_CART, restaurant_phase="confirmation")
        assert _route_from_triage(state) == "restaurant_confirm"

    def test_confirmed_but_unexecuted_resumes_at_execute(self):
        state = _base_state(
            cart=_CART,
            restaurant_phase="confirmation",
            restaurant_order_confirmed=True,
        )
        assert _route_from_triage(state) == "execute"

    def test_post_checkout_order_starts_fresh_collection(self):
        state = _base_state(
            cart=_CART,
            restaurant_phase="confirmation",
            restaurant_order_confirmed=True,
            execute_confirmed=True,
        )
        assert _route_from_triage(state) == "order_collect"

    def test_menu_inquiry_without_order_routes_to_menu(self):
        state = _base_state(intent="menu_inquiry")
        assert _route_from_triage(state) == "menu_inquiry"

    def test_complaint_beats_order_state(self):
        state = _base_state(intent="complaint", cart=_CART)
        assert _route_from_triage(state) == "complaint_collect"


class TestConfirmRouter:
    def test_confirmed_chains_to_execute(self):
        state = _base_state(restaurant_order_confirmed=True)
        assert _route_from_confirm(state) == "execute"

    def test_modify_chains_to_collect(self):
        state = _base_state(restaurant_order_confirmed=False, restaurant_phase="collect")
        assert _route_from_confirm(state) == "order_collect"

    def test_unclear_ends_turn(self):
        state = _base_state(restaurant_phase="confirmation")
        assert _route_from_confirm(state) == END


class TestTriageGuard:
    async def test_mid_order_turn_skips_reclassification(self):
        """intent=order + non-empty cart must short-circuit before any LLM use."""
        state = _base_state(cart=_CART)
        assert await triage_node(state, _CONFIG) == {}

    async def test_awaiting_confirmation_skips_reclassification(self):
        state = _base_state(restaurant_phase="confirmation")
        assert await triage_node(state, _CONFIG) == {}

    async def test_post_checkout_reenables_triage(self):
        """execute_confirmed disables the guard — the node proceeds past it
        (and would hit the provider, which we stub to prove it got there)."""
        state = _base_state(cart=_CART, execute_confirmed=True)
        provider = Mock()
        provider.stream_chat.side_effect = RuntimeError("classification attempted")
        with patch("src.agents.triage.get_provider", return_value=provider):
            with pytest.raises(RuntimeError, match="classification attempted"):
                await triage_node(state, _CONFIG)


class TestGraphWiring:
    async def _stream(self, graph, inputs):
        results = []
        async for chunk in graph.astream(inputs, config=_CONFIG, stream_mode="updates"):
            results.append(chunk)
        return [list(r.keys())[0] for r in results if isinstance(r, dict)]

    async def test_collect_complete_autochains_to_summary(self):
        mock_triage = AsyncMock(return_value={"intent": "order"})
        mock_collect = AsyncMock(
            return_value={
                "messages": [AIMessage(content="Pedido armado")],
                "cart": _CART,
                "restaurant_order_complete": True,
                "restaurant_phase": "collect",
            }
        )
        mock_summary = AsyncMock(
            return_value={
                "messages": [AIMessage(content="Resumen… responde confirmar")],
                "restaurant_phase": "confirmation",
            }
        )
        with (
            patch("src.graphs.restaurant_graph.triage_node", new=mock_triage),
            patch("src.graphs.restaurant_graph.restaurant_order_collect_node", new=mock_collect),
            patch("src.graphs.restaurant_graph.restaurant_order_summary_node", new=mock_summary),
        ):
            graph = build_restaurant_graph(MemorySaver())
            nodes = await self._stream(
                graph,
                {
                    "messages": [HumanMessage(content="2 tacos para llevar")],
                    "thread_id": "test-user:rest-1",
                    "user_context": {"name": "Ana"},
                    "agent_type": "restaurant",
                },
            )
        assert "order_collect" in nodes
        assert "order_summary" in nodes

    async def test_confirm_autochains_to_execute(self):
        mock_triage = AsyncMock(return_value={})  # guard skip: state drives routing
        mock_confirm = AsyncMock(
            return_value={
                "messages": [AIMessage(content="¡Gracias! Enviando tu pedido…")],
                "restaurant_order_confirmed": True,
            }
        )
        mock_execute = AsyncMock(return_value={"execute_confirmed": True})
        with (
            patch("src.graphs.restaurant_graph.triage_node", new=mock_triage),
            patch("src.graphs.restaurant_graph.restaurant_confirm_node", new=mock_confirm),
            patch("src.graphs.restaurant_graph.execute_node", new=mock_execute),
        ):
            graph = build_restaurant_graph(MemorySaver())
            nodes = await self._stream(
                graph,
                {
                    "messages": [HumanMessage(content="confirmar")],
                    "thread_id": "test-user:rest-1",
                    "user_context": {"name": "Ana"},
                    "agent_type": "restaurant",
                    "intent": "order",
                    "cart": _CART,
                    "restaurant_phase": "confirmation",
                },
            )
        assert "restaurant_confirm" in nodes
        assert "execute" in nodes

    async def test_confirm_modify_reenters_collect_same_turn(self):
        mock_triage = AsyncMock(return_value={})
        mock_confirm = AsyncMock(
            return_value={
                "restaurant_order_confirmed": False,
                "restaurant_phase": "collect",
                "restaurant_order_complete": False,
            }
        )
        mock_collect = AsyncMock(
            return_value={
                "messages": [AIMessage(content="Claro, ¿qué cambiamos?")],
                "cart": _CART,
                "restaurant_order_complete": False,
                "restaurant_phase": "collect",
            }
        )
        with (
            patch("src.graphs.restaurant_graph.triage_node", new=mock_triage),
            patch("src.graphs.restaurant_graph.restaurant_confirm_node", new=mock_confirm),
            patch("src.graphs.restaurant_graph.restaurant_order_collect_node", new=mock_collect),
        ):
            graph = build_restaurant_graph(MemorySaver())
            nodes = await self._stream(
                graph,
                {
                    "messages": [HumanMessage(content="quita el flan")],
                    "thread_id": "test-user:rest-1",
                    "user_context": {"name": "Ana"},
                    "agent_type": "restaurant",
                    "intent": "order",
                    "cart": _CART,
                    "restaurant_phase": "confirmation",
                },
            )
        assert "restaurant_confirm" in nodes
        assert "order_collect" in nodes
        assert "execute" not in nodes
