"""
Tests for sales flow phase progression and API key security.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graphs.main_graph import build_graph


class TestSalesFlow:
    @pytest.mark.asyncio
    async def test_sales_collect_invoked_for_sales_intent(self):
        """When triage returns sales intent, sales_collect is called."""
        config = {
            "configurable": {
                "thread_id": "test-user:sales-thread-1",
                "openai_api_key": "sk-test",
            }
        }

        mock_triage = AsyncMock(return_value={"intent": "sales"})
        mock_sales = AsyncMock(
            return_value={
                "messages": [AIMessage(content="¿Cuál es tu nombre?")],
                "sales_step": 1,
            }
        )

        with (
            patch("src.graphs.main_graph.triage_node", new=mock_triage),
            patch("src.graphs.main_graph.sales_collect_node", new=mock_sales),
        ):
            graph = build_graph(MemorySaver())
            chunks = []
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="Quiero hacer un pedido")],
                    "thread_id": "test-user:sales-thread-1",
                },
                config=config,
                stream_mode="updates",
            ):
                chunks.append(chunk)

        node_names = [list(c.keys())[0] for c in chunks if isinstance(c, dict)]
        assert "sales_collect" in node_names

    @pytest.mark.asyncio
    async def test_triage_skipped_when_sales_already_in_progress(self):
        """triage is a no-op (returns {}) when intent=sales and sales_step > 0."""
        config = {
            "configurable": {
                "thread_id": "test-user:sales-thread-2",
                "openai_api_key": "sk-test",
            }
        }

        # When already in sales flow, triage returns {} (its early-exit path)
        mock_triage = AsyncMock(return_value={})
        mock_sales = AsyncMock(
            return_value={
                "messages": [AIMessage(content="Paso 2: productos.")],
                "sales_step": 2,
            }
        )

        with (
            patch("src.graphs.main_graph.triage_node", new=mock_triage),
            patch("src.graphs.main_graph.sales_collect_node", new=mock_sales),
        ):
            graph = build_graph(MemorySaver())
            chunks = []
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="2 camisetas azules")],
                    "thread_id": "test-user:sales-thread-2",
                    "intent": "sales",
                    "sales_step": 1,
                },
                config=config,
                stream_mode="updates",
            ):
                chunks.append(chunk)

        node_names = [list(c.keys())[0] for c in chunks if isinstance(c, dict)]
        assert "sales_collect" in node_names

    @pytest.mark.asyncio
    async def test_sales_auto_advances_through_all_phases(self):
        """When each phase completes, the graph auto-advances through all sales phases."""
        config = {
            "configurable": {
                "thread_id": "test-user:sales-thread-3",
                "openai_api_key": "sk-test",
            }
        }

        mock_triage = AsyncMock(return_value={"intent": "sales"})
        mock_sales = AsyncMock(
            return_value={
                "messages": [AIMessage(content="Datos completos.")],
                "sales_step": 3,
                "sales_complete": True,
                "order_data": {"customer_name": "Juan", "items": [{"product": "Camiseta", "quantity": 2}]},
            }
        )
        mock_summary = AsyncMock(
            return_value={
                "messages": [AIMessage(content="Pedido confirmado.")],
                "order_confirmed": True,
            }
        )
        mock_execute = AsyncMock(
            return_value={"messages": [], "execute_confirmed": True}
        )

        with (
            patch("src.graphs.main_graph.triage_node", new=mock_triage),
            patch("src.graphs.main_graph.sales_collect_node", new=mock_sales),
            patch("src.graphs.main_graph.order_summary_node", new=mock_summary),
            patch("src.graphs.main_graph.execute_node", new=mock_execute),
        ):
            graph = build_graph(MemorySaver())
            chunks = []
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="Quiero comprar camisetas")],
                    "thread_id": "test-user:sales-thread-3",
                },
                config=config,
                stream_mode="updates",
            ):
                chunks.append(chunk)

        node_names = [list(c.keys())[0] for c in chunks if isinstance(c, dict)]
        assert "sales_collect" in node_names
        assert "order_summary" in node_names
        assert "execute" in node_names

    @pytest.mark.asyncio
    async def test_tracking_routes_to_execute(self):
        """Tracking intent auto-advances to execute when complete."""
        config = {
            "configurable": {
                "thread_id": "test-user:tracking-thread-1",
                "openai_api_key": "sk-test",
            }
        }

        mock_triage = AsyncMock(return_value={"intent": "tracking"})
        mock_tracking = AsyncMock(
            return_value={
                "messages": [AIMessage(content="Buscando tu pedido.")],
                "tracking_data": {"order_id": "ORD-123"},
                "tracking_complete": True,
            }
        )
        mock_execute = AsyncMock(
            return_value={"messages": [], "execute_confirmed": True}
        )

        with (
            patch("src.graphs.main_graph.triage_node", new=mock_triage),
            patch("src.graphs.main_graph.tracking_collect_node", new=mock_tracking),
            patch("src.graphs.main_graph.execute_node", new=mock_execute),
        ):
            graph = build_graph(MemorySaver())
            chunks = []
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="¿Dónde está mi pedido ORD-123?")],
                    "thread_id": "test-user:tracking-thread-1",
                },
                config=config,
                stream_mode="updates",
            ):
                chunks.append(chunk)

        node_names = [list(c.keys())[0] for c in chunks if isinstance(c, dict)]
        assert "tracking_collect" in node_names
        assert "execute" in node_names

    @pytest.mark.asyncio
    async def test_complaint_routes_to_execute(self):
        """Complaint intent auto-advances to execute when complete."""
        config = {
            "configurable": {
                "thread_id": "test-user:complaint-thread-1",
                "openai_api_key": "sk-test",
            }
        }

        mock_triage = AsyncMock(return_value={"intent": "complaint"})
        mock_complaint = AsyncMock(
            return_value={
                "messages": [AIMessage(content="Registrando tu queja.")],
                "complaint_data": {"order_ref": "ORD-456", "issue_description": "Producto dañado", "desired_resolution": "Reemplazo"},
                "complaint_complete": True,
            }
        )
        mock_execute = AsyncMock(
            return_value={"messages": [], "execute_confirmed": True}
        )

        with (
            patch("src.graphs.main_graph.triage_node", new=mock_triage),
            patch("src.graphs.main_graph.complaint_collect_node", new=mock_complaint),
            patch("src.graphs.main_graph.execute_node", new=mock_execute),
        ):
            graph = build_graph(MemorySaver())
            chunks = []
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="Mi pedido llegó dañado")],
                    "thread_id": "test-user:complaint-thread-1",
                },
                config=config,
                stream_mode="updates",
            ):
                chunks.append(chunk)

        node_names = [list(c.keys())[0] for c in chunks if isinstance(c, dict)]
        assert "complaint_collect" in node_names
        assert "execute" in node_names

    @pytest.mark.asyncio
    async def test_api_key_not_in_state_updates(self):
        """API key must never appear in any state update chunk."""
        config = {
            "configurable": {
                "thread_id": "test-user:sales-thread-4",
                "openai_api_key": "sk-must-not-leak",
            }
        }

        mock_triage = AsyncMock(return_value={"intent": "faq"})
        mock_faq = AsyncMock(return_value={"messages": [AIMessage(content="Hola")]})

        with (
            patch("src.graphs.main_graph.triage_node", new=mock_triage),
            patch("src.graphs.main_graph.faq_response_node", new=mock_faq),
        ):
            graph = build_graph(MemorySaver())
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="Hola")],
                    "thread_id": "test-user:sales-thread-4",
                },
                config=config,
                stream_mode="updates",
            ):
                chunk_str = json.dumps(chunk, default=str)
                assert "sk-must-not-leak" not in chunk_str, (
                    f"API key leaked into state chunk: {chunk_str}"
                )
