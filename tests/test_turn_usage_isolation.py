"""The SSE `done` event must report ONLY the current turn's usage/products.

`turn_usage` is an operator.add channel and `mentioned_product_ids` /
`faq_used` are last-write-wins channels — all of them persist in the
checkpointer across turns on the same thread. Without per-run isolation the
done event re-reports prior turns' records: NestJS would persist duplicate
AiInvocationUsage rows under each new turnRequestId (billing over-count) and
re-attach stale product images on unrelated replies.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graphs.main_graph import build_graph
from src.main import _stream_graph


def _usage(node: str) -> dict:
    return {
        "node": node,
        "provider": "openai",
        "model": "gpt-test",
        "input_tokens": 10,
        "output_tokens": 5,
    }


def _config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            "turn_request_id": f"req-{thread_id}",
            "openai_api_key": "sk-test",
        }
    }


def _inputs(text: str) -> dict:
    # Mirrors the per-turn reset fields chat_stream puts on every request.
    # A known name keeps the routing away from the real name_capture node.
    return {
        "messages": [HumanMessage(content=text)],
        "user_context": {"name": "Ana"},
        "faqs": [],
        "attachments": [],
        "faq_used": None,
        "mentioned_product_ids": [],
    }


async def _run_turn(graph, text: str, thread_id: str) -> dict:
    """Drive _stream_graph for one turn; return the parsed done event."""
    done = None
    async for sse in _stream_graph(
        _inputs(text), _config(thread_id), graph=graph, agent_code_name="helena"
    ):
        payload = json.loads(sse.removeprefix("data: ").strip())
        if payload.get("type") == "done":
            done = payload
    assert done is not None, "stream never emitted a done event"
    return done


class TestPerTurnIsolation:
    async def test_turn_usage_reports_only_current_turn(self):
        thread = "test-user:usage-iso"

        mock_triage = AsyncMock(
            side_effect=lambda *a, **k: {
                "intent": "faq",
                "turn_usage": [_usage("triage")],
            }
        )
        mock_faq = AsyncMock(
            side_effect=lambda *a, **k: {
                "messages": [AIMessage(content="respuesta")],
                "turn_usage": [_usage("faq_response")],
            }
        )

        with (
            patch("src.graphs.sales_graph.triage_node", new=mock_triage),
            patch("src.graphs.sales_graph.faq_response_node", new=mock_faq),
        ):
            graph = build_graph(MemorySaver())

            done1 = await _run_turn(graph, "hola", thread)
            assert [u["node"] for u in done1["turn_usage"]] == [
                "triage",
                "faq_response",
            ]

            done2 = await _run_turn(graph, "¿horarios?", thread)
            # Turn 2 must NOT re-report turn 1's two records.
            assert [u["node"] for u in done2["turn_usage"]] == [
                "triage",
                "faq_response",
            ]

    async def test_mentioned_product_ids_do_not_leak_across_turns(self):
        thread = "test-user:product-iso"

        mock_triage = AsyncMock(
            side_effect=lambda *a, **k: {"intent": "faq", "turn_usage": []}
        )
        # Turn 1's node mentions a product; turn 2's node doesn't touch the field.
        mock_faq = AsyncMock(
            side_effect=[
                {
                    "messages": [AIMessage(content="producto X")],
                    "mentioned_product_ids": ["prod-1"],
                    "turn_usage": [],
                },
                {
                    "messages": [AIMessage(content="horarios")],
                    "turn_usage": [],
                },
            ]
        )

        with (
            patch("src.graphs.sales_graph.triage_node", new=mock_triage),
            patch("src.graphs.sales_graph.faq_response_node", new=mock_faq),
        ):
            graph = build_graph(MemorySaver())

            done1 = await _run_turn(graph, "info del producto", thread)
            assert done1["mentioned_product_ids"] == ["prod-1"]

            done2 = await _run_turn(graph, "¿horarios?", thread)
            assert done2["mentioned_product_ids"] == []

    async def test_faq_used_does_not_leak_across_turns(self):
        thread = "test-user:faq-used-iso"

        mock_triage = AsyncMock(
            side_effect=lambda *a, **k: {"intent": "faq", "turn_usage": []}
        )
        mock_faq = AsyncMock(
            side_effect=[
                {
                    "messages": [AIMessage(content="r1")],
                    "faq_used": [{"id": "f-1", "question": "q", "confidence": 0.5}],
                    "turn_usage": [],
                },
                {
                    "messages": [AIMessage(content="r2")],
                    "turn_usage": [],
                },
            ]
        )

        with (
            patch("src.graphs.sales_graph.triage_node", new=mock_triage),
            patch("src.graphs.sales_graph.faq_response_node", new=mock_faq),
        ):
            graph = build_graph(MemorySaver())

            done1 = await _run_turn(graph, "¿envíos?", thread)
            assert done1["faq_used"] == [
                {"id": "f-1", "question": "q", "confidence": 0.5}
            ]

            done2 = await _run_turn(graph, "gracias", thread)
            assert done2["faq_used"] == []
