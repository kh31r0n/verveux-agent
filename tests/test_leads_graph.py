"""Tests for the leads graph (veronica) — routing, auto-chain, and the
execute_lead idempotency contract.

Mirrors test_appointments_graph.py: MemorySaver, mock the individual nodes,
drive astream(), inspect the sequence of node updates.
"""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import Request, Response, HTTPStatusError
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agents.execute_lead import execute_lead_node
from src.agents.lead_collect import _sanitize
from src.graphs.leads_graph import build_leads_graph


# ── Helpers ─────────────────────────────────────────────────────────────────


def _node_names(chunks: list) -> list[str]:
    out: list[str] = []
    for c in chunks:
        if not isinstance(c, dict):
            continue
        for key in c.keys():
            if key.startswith("__"):
                continue
            out.append(key)
    return out


def _config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            "openai_api_key": "sk-test",
        }
    }


async def _run(graph, inputs: dict, thread_id: str) -> list[str]:
    chunks = []
    async for c in graph.astream(
        inputs, config=_config(thread_id), stream_mode="updates"
    ):
        chunks.append(c)
    return _node_names(chunks)


# ── Field sanitization ──────────────────────────────────────────────────────


class TestSanitize:
    def test_rejects_invalid_email(self):
        assert "email" not in _sanitize({"email": "not-an-email"})
        assert _sanitize({"email": "a@b.co"})["email"] == "a@b.co"

    def test_normalizes_service_interest_case(self):
        assert _sanitize({"serviceInterest": "crm"})["serviceInterest"] == "CRM"

    def test_rejects_unknown_service_interest(self):
        assert "serviceInterest" not in _sanitize({"serviceInterest": "OTRO"})

    def test_drops_blank_and_non_string_values(self):
        clean = _sanitize({"fullName": "  ", "company": 42, "email": None})
        assert clean == {}


# ── Routing matrix ─────────────────────────────────────────────────────────


class TestLeadsGraphRouting:
    @pytest.mark.asyncio
    async def test_faq_intent_routes_to_faq_response(self):
        mock_triage = AsyncMock(return_value={"intent": "faq"})
        mock_faq = AsyncMock(
            return_value={"messages": [AIMessage(content="Claro, te cuento…")]}
        )
        with (
            patch("src.graphs.leads_graph.triage_node", new=mock_triage),
            patch("src.graphs.leads_graph.faq_response_node", new=mock_faq),
        ):
            graph = build_leads_graph(MemorySaver())
            names = await _run(
                graph,
                {
                    "messages": [HumanMessage(content="¿qué hace la plataforma?")],
                    "thread_id": "u:faq",
                    "user_context": {"name": "Ana"},
                },
                "u:faq",
            )
        assert "faq_response" in names
        assert "lead_collect" not in names

    @pytest.mark.asyncio
    async def test_lead_capture_intent_routes_to_lead_collect(self):
        mock_triage = AsyncMock(return_value={"intent": "lead_capture"})
        mock_collect = AsyncMock(
            return_value={
                "messages": [AIMessage(content="¿Me compartes tu correo?")],
                "lead_data": {"fullName": "Ana"},
                "lead_collection_complete": False,
            }
        )
        with (
            patch("src.graphs.leads_graph.triage_node", new=mock_triage),
            patch("src.graphs.leads_graph.lead_collect_node", new=mock_collect),
        ):
            graph = build_leads_graph(MemorySaver())
            names = await _run(
                graph,
                {
                    "messages": [HumanMessage(content="me interesa el CRM")],
                    "thread_id": "u:lead",
                    "user_context": {"name": "Ana"},
                },
                "u:lead",
            )
        assert "lead_collect" in names
        assert "execute_lead" not in names  # incomplete → wait for user

    @pytest.mark.asyncio
    async def test_unknown_visitor_routes_to_name_capture_first(self):
        mock_triage = AsyncMock(return_value={"intent": "lead_capture"})
        mock_name = AsyncMock(
            return_value={
                "messages": [AIMessage(content="¿Cómo te llamas?")],
                "name_capture_reply_sent": True,
                "name_capture_attempts": 1,
            }
        )
        with (
            patch("src.graphs.leads_graph.triage_node", new=mock_triage),
            patch("src.graphs.leads_graph.name_capture_node", new=mock_name),
        ):
            graph = build_leads_graph(MemorySaver())
            names = await _run(
                graph,
                {
                    "messages": [HumanMessage(content="quiero info")],
                    "thread_id": "u:anon",
                    "user_context": {},
                },
                "u:anon",
            )
        assert "name_capture" in names
        assert "lead_collect" not in names

    @pytest.mark.asyncio
    async def test_greeting_with_known_name_routes_to_greeting(self):
        mock_triage = AsyncMock(return_value={"intent": "greeting"})
        mock_greet = AsyncMock(
            return_value={"messages": [AIMessage(content="¡Hola Ana!")]}
        )
        with (
            patch("src.graphs.leads_graph.triage_node", new=mock_triage),
            patch(
                "src.graphs.leads_graph.greeting_response_node", new=mock_greet
            ),
        ):
            graph = build_leads_graph(MemorySaver())
            names = await _run(
                graph,
                {
                    "messages": [HumanMessage(content="hola")],
                    "thread_id": "u:greet",
                    "user_context": {"name": "Ana"},
                },
                "u:greet",
            )
        assert "greeting_response" in names
        assert "name_capture" not in names


# ── Auto-chain into submission ─────────────────────────────────────────────


class TestLeadsAutoChain:
    @pytest.mark.asyncio
    async def test_complete_collection_chains_into_execute_lead(self):
        mock_triage = AsyncMock(return_value={"intent": "lead_capture"})
        mock_collect = AsyncMock(
            return_value={
                "messages": [AIMessage(content="¡Gracias! Un asesor te contactará.")],
                "lead_data": {
                    "fullName": "Ana García",
                    "email": "ana@empresa.com",
                    "serviceInterest": "CRM",
                },
                "lead_collection_complete": True,
                "lead_submission_id": "sub-1",
            }
        )
        mock_execute = AsyncMock(return_value={"lead_submitted": True})
        with (
            patch("src.graphs.leads_graph.triage_node", new=mock_triage),
            patch("src.graphs.leads_graph.lead_collect_node", new=mock_collect),
            patch("src.graphs.leads_graph.execute_lead_node", new=mock_execute),
        ):
            graph = build_leads_graph(MemorySaver())
            names = await _run(
                graph,
                {
                    "messages": [HumanMessage(content="ana@empresa.com, CRM")],
                    "thread_id": "u:chain",
                    "user_context": {"name": "Ana"},
                },
                "u:chain",
            )
        assert names.count("execute_lead") == 1
        mock_execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_completed_but_unsubmitted_lead_retries_execute_from_triage(self):
        """A failed POST last turn leaves complete+unsubmitted state — the
        next turn goes straight to execute_lead with the same submission id."""
        mock_triage = AsyncMock(return_value={"intent": "lead_capture"})
        mock_execute = AsyncMock(return_value={"lead_submitted": True})
        with (
            patch("src.graphs.leads_graph.triage_node", new=mock_triage),
            patch("src.graphs.leads_graph.execute_lead_node", new=mock_execute),
        ):
            graph = build_leads_graph(MemorySaver())
            names = await _run(
                graph,
                {
                    "messages": [HumanMessage(content="¿quedó registrado?")],
                    "thread_id": "u:retry",
                    "user_context": {"name": "Ana"},
                    "lead_data": {
                        "fullName": "Ana",
                        "email": "ana@empresa.com",
                        "serviceInterest": "CRM",
                    },
                    "lead_collection_complete": True,
                    "lead_submission_id": "sub-retry",
                    "lead_submitted": False,
                },
                "u:retry",
            )
        assert "execute_lead" in names
        assert "lead_collect" not in names


# ── execute_lead idempotency contract ──────────────────────────────────────


def _lead_state(**overrides) -> dict:
    state = {
        "thread_id": "u:exec",
        "tenant_id": "tenant-1",
        "conversation_id": "conv-1",
        "language": "es",
        "lead_submission_id": "sub-1",
        "lead_data": {
            "fullName": "Ana García",
            "email": "ana@empresa.com",
            "serviceInterest": "CRM",
        },
        "lead_submitted": False,
    }
    state.update(overrides)
    return state


class TestExecuteLeadNode:
    @pytest.mark.asyncio
    async def test_replay_with_lead_submitted_makes_no_http_call(self):
        mock_submit = AsyncMock()
        with patch(
            "src.agents.backend_client.submit_inquiry", new=mock_submit
        ):
            result = await execute_lead_node(
                _lead_state(lead_submitted=True), {}
            )
        assert result == {}
        mock_submit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_success_posts_idempotency_key_and_latches(self):
        mock_submit = AsyncMock(
            return_value={"ok": True, "inquiryId": "inq-1", "deduped": False}
        )
        with patch(
            "src.agents.backend_client.submit_inquiry", new=mock_submit
        ):
            result = await execute_lead_node(_lead_state(), {})

        assert result == {"lead_submitted": True}
        mock_submit.assert_awaited_once_with(
            tenant_id="tenant-1",
            conversation_id="conv-1",
            idempotency_key="sub-1",
            lead_data=_lead_state()["lead_data"],
        )

    @pytest.mark.asyncio
    async def test_backend_failure_does_not_latch_and_apologizes(self):
        error = HTTPStatusError(
            "boom",
            request=Request("POST", "http://x/api/v1/internal/inquiries"),
            response=Response(500, text="oops"),
        )
        mock_submit = AsyncMock(side_effect=error)
        writes: list[dict] = []
        with (
            patch("src.agents.backend_client.submit_inquiry", new=mock_submit),
            patch(
                "src.agents.execute_lead.get_stream_writer",
                new=lambda: writes.append,
            ),
        ):
            result = await execute_lead_node(_lead_state(), {})

        assert "lead_submitted" not in result
        assert result["messages"], "failure must append an apology message"
        assert "problema" in result["messages"][0].content
        assert writes and writes[0]["type"] == "token"

    @pytest.mark.asyncio
    async def test_missing_submission_id_is_a_silent_noop(self):
        mock_submit = AsyncMock()
        with patch(
            "src.agents.backend_client.submit_inquiry", new=mock_submit
        ):
            result = await execute_lead_node(
                _lead_state(lead_submission_id=None), {}
            )
        assert result == {}
        mock_submit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_double_invocation_only_posts_once(self):
        """The full replay contract: first call posts + latches, a second call
        with the latched state is a no-op."""
        mock_submit = AsyncMock(return_value={"ok": True, "inquiryId": "inq-1"})
        with patch(
            "src.agents.backend_client.submit_inquiry", new=mock_submit
        ):
            state = _lead_state()
            first = await execute_lead_node(state, {})
            state.update(first)
            second = await execute_lead_node(state, {})

        assert first == {"lead_submitted": True}
        assert second == {}
        assert mock_submit.await_count == 1
