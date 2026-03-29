"""
Integration-style tests for the LangGraph graph wiring.
Uses MemorySaver (allowed in tests only) to avoid requiring a real PostgreSQL instance.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graphs.main_graph import build_graph


class TestGraphWiring:
    @pytest.mark.asyncio
    async def test_graph_routes_faq_to_faq_response(self):
        """triage returning faq intent should invoke faq_response then END."""
        config = {
            "configurable": {
                "thread_id": "test-user:thread-1",
                "openai_api_key": "sk-test",
            }
        }

        mock_triage = AsyncMock(return_value={"intent": "faq"})
        mock_faq = AsyncMock(
            return_value={"messages": [AIMessage(content="¡Hola! Puedo ayudarte con ventas, rastreo y más.")]}
        )

        with (
            patch("src.graphs.main_graph.triage_node", new=mock_triage),
            patch("src.graphs.main_graph.faq_response_node", new=mock_faq),
        ):
            graph = build_graph(MemorySaver())
            results = []
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="Hola")],
                    "thread_id": "test-user:thread-1",
                },
                config=config,
                stream_mode="updates",
            ):
                results.append(chunk)

        assert len(results) > 0
        node_names = [list(r.keys())[0] for r in results if isinstance(r, dict)]
        assert "faq_response" in node_names
        assert "sales_collect" not in node_names

    @pytest.mark.asyncio
    async def test_graph_routes_sales_to_sales_collect(self):
        """triage returning sales intent should invoke sales_collect."""
        config = {
            "configurable": {
                "thread_id": "test-user:thread-2",
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
            results = []
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="Quiero comprar algo")],
                    "thread_id": "test-user:thread-2",
                },
                config=config,
                stream_mode="updates",
            ):
                results.append(chunk)

        node_names = [list(r.keys())[0] for r in results if isinstance(r, dict)]
        assert "sales_collect" in node_names
        assert "faq_response" not in node_names

    @pytest.mark.asyncio
    async def test_state_does_not_contain_api_key(self):
        """Ensure openai_api_key never appears in any state update."""
        config = {
            "configurable": {
                "thread_id": "test-user:thread-3",
                "openai_api_key": "sk-super-secret",
            }
        }

        mock_triage = AsyncMock(return_value={"intent": "faq"})
        mock_faq = AsyncMock(return_value={"messages": [AIMessage(content="Puedo ayudarte.")]})

        with (
            patch("src.graphs.main_graph.triage_node", new=mock_triage),
            patch("src.graphs.main_graph.faq_response_node", new=mock_faq),
        ):
            graph = build_graph(MemorySaver())
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="test")],
                    "thread_id": "test-user:thread-3",
                },
                config=config,
                stream_mode="updates",
            ):
                chunk_str = json.dumps(chunk, default=str)
                assert "sk-super-secret" not in chunk_str, (
                    f"API key leaked into state update: {chunk_str}"
                )

    @pytest.mark.asyncio
    async def test_display_message_is_natural_language(self):
        """The AIMessage emitted by faq_response must not be raw JSON."""
        config = {
            "configurable": {
                "thread_id": "test-user:thread-4",
                "openai_api_key": "sk-test",
            }
        }

        mock_triage = AsyncMock(return_value={"intent": "faq"})
        mock_faq = AsyncMock(
            return_value={"messages": [AIMessage(content="¡Hola! ¿En qué puedo ayudarte?")]}
        )

        with (
            patch("src.graphs.main_graph.triage_node", new=mock_triage),
            patch("src.graphs.main_graph.faq_response_node", new=mock_faq),
        ):
            graph = build_graph(MemorySaver())
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="test")],
                    "thread_id": "test-user:thread-4",
                },
                config=config,
                stream_mode="updates",
            ):
                if isinstance(chunk, dict) and "faq_response" in chunk:
                    messages = chunk["faq_response"].get("messages", [])
                    for msg in messages:
                        content = msg.content if hasattr(msg, "content") else str(msg)
                        assert not (
                            content.strip().startswith("{") and '"intent"' in content
                        ), f"FAQ emitted raw JSON to user: {content}"
