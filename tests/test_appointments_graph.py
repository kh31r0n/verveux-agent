"""Tests for the restructured appointments graph (Phase 3).

Mirrors the patterns in test_graph.py / test_interrupt.py: MemorySaver, mock
the individual nodes, drive astream(), inspect the sequence of node updates.
"""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agents.appointments.reservation_propose import _parse_choice
from src.graphs.appointments_graph import build_appointments_graph


# ── Helpers ─────────────────────────────────────────────────────────────────


def _node_names(chunks: list) -> list[str]:
    """Pull the leading node key from each `updates`-mode chunk."""
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


# ── parse_choice helper ────────────────────────────────────────────────────


class TestParseChoice:
    def test_numeric_choice_picks_indexed_slot(self):
        slots = [{"startsAt": "2026-07-01T10:00:00Z"}, {"startsAt": "2026-07-01T14:00:00Z"}]
        assert _parse_choice("2", slots) is slots[1]

    def test_time_substring_matches_unique_slot(self):
        slots = [{"startsAt": "2026-07-01T10:00:00Z"}, {"startsAt": "2026-07-01T14:00:00Z"}]
        assert _parse_choice("a las 14:00 por favor", slots) is slots[1]

    def test_ambiguous_returns_none(self):
        slots = [{"startsAt": "2026-07-01T10:00:00Z"}, {"startsAt": "2026-07-02T10:00:00Z"}]
        # Two slots both at 10:00 — caller cannot disambiguate.
        assert _parse_choice("a las 10:00", slots) is None

    def test_unparseable_returns_none(self):
        slots = [{"startsAt": "2026-07-01T10:00:00Z"}]
        assert _parse_choice("hola", slots) is None


# ── Routing matrix ─────────────────────────────────────────────────────────


class TestAppointmentsGraphRouting:
    @pytest.mark.asyncio
    async def test_booking_intent_routes_to_appointment_collect(self):
        mock_triage = AsyncMock(return_value={"intent": "booking"})
        mock_collect = AsyncMock(
            return_value={
                "messages": [AIMessage(content="¿Qué tipo de cita necesitas?")],
                "booking_intent": "book",
            }
        )
        with (
            patch("src.graphs.appointments_graph.triage_node", new=mock_triage),
            patch(
                "src.graphs.appointments_graph.appointment_collect_node",
                new=mock_collect,
            ),
        ):
            graph = build_appointments_graph(MemorySaver())
            chunks = []
            async for c in graph.astream(
                {
                    "messages": [HumanMessage(content="quiero una cita")],
                    "thread_id": "u:1",
                    "user_context": {"name": "Ana"},
                },
                config=_config("u:1"),
                stream_mode="updates",
            ):
                chunks.append(c)
        names = _node_names(chunks)
        assert "appointment_collect" in names
        assert "appointment_cancel" not in names
        assert "faq_response" not in names

    @pytest.mark.asyncio
    async def test_cancel_intent_routes_to_appointment_cancel(self):
        mock_triage = AsyncMock(return_value={"intent": "appointment_cancel"})
        mock_cancel = AsyncMock(
            return_value={"messages": [AIMessage(content="Cancelando...")]}
        )
        with (
            patch("src.graphs.appointments_graph.triage_node", new=mock_triage),
            patch(
                "src.graphs.appointments_graph.appointment_cancel_node",
                new=mock_cancel,
            ),
        ):
            graph = build_appointments_graph(MemorySaver())
            chunks = []
            async for c in graph.astream(
                {
                    "messages": [HumanMessage(content="cancela mi cita")],
                    "thread_id": "u:cancel",
                },
                config=_config("u:cancel"),
                stream_mode="updates",
            ):
                chunks.append(c)
        names = _node_names(chunks)
        assert "appointment_cancel" in names
        assert "appointment_collect" not in names

    @pytest.mark.asyncio
    async def test_reschedule_intent_routes_to_appointment_reschedule(self):
        mock_triage = AsyncMock(
            return_value={"intent": "appointment_reschedule"}
        )
        # Reschedule node returns without a source id → graph ends at this node.
        mock_resched = AsyncMock(
            return_value={
                "messages": [AIMessage(content="¿Cuál cita?")],
            }
        )
        with (
            patch("src.graphs.appointments_graph.triage_node", new=mock_triage),
            patch(
                "src.graphs.appointments_graph.appointment_reschedule_node",
                new=mock_resched,
            ),
        ):
            graph = build_appointments_graph(MemorySaver())
            chunks = []
            async for c in graph.astream(
                {
                    "messages": [HumanMessage(content="muévela al jueves")],
                    "thread_id": "u:r",
                    "user_context": {"name": "Ana"},
                },
                config=_config("u:r"),
                stream_mode="updates",
            ):
                chunks.append(c)
        names = _node_names(chunks)
        assert "appointment_reschedule" in names
        assert "availability_lookup" not in names

    @pytest.mark.asyncio
    async def test_faq_default_routes_to_faq_response(self):
        mock_triage = AsyncMock(return_value={"intent": "faq"})
        mock_faq = AsyncMock(
            return_value={"messages": [AIMessage(content="Atendemos 9-5.")]}
        )
        with (
            patch("src.graphs.appointments_graph.triage_node", new=mock_triage),
            patch(
                "src.graphs.appointments_graph.faq_response_node", new=mock_faq
            ),
        ):
            graph = build_appointments_graph(MemorySaver())
            chunks = []
            async for c in graph.astream(
                {
                    "messages": [HumanMessage(content="¿qué horarios atienden?")],
                    "thread_id": "u:faq",
                    "user_context": {"name": "Ana"},
                },
                config=_config("u:faq"),
                stream_mode="updates",
            ):
                chunks.append(c)
        names = _node_names(chunks)
        assert "faq_response" in names

    @pytest.mark.asyncio
    async def test_booking_complete_chains_into_availability_lookup(self):
        # Once the collect node says "booking_complete" the conditional edge
        # advances in the same turn.
        mock_triage = AsyncMock(return_value={"intent": "booking"})
        mock_collect = AsyncMock(
            return_value={
                "messages": [AIMessage(content="Listo.")],
                "booking_intent": "book",
                "booking_complete": True,
            }
        )
        mock_avail = AsyncMock(
            return_value={
                "candidate_slots": [
                    {"startsAt": "2026-07-01T10:00:00Z", "endsAt": "2026-07-01T10:30:00Z", "resources": []}
                ],
                "messages": [AIMessage(content="Tengo este horario...")],
            }
        )
        mock_propose = AsyncMock(
            return_value={
                "chosen_slot": None,
                "messages": [AIMessage(content="¿Cuál prefieres?")],
            }
        )
        with (
            patch("src.graphs.appointments_graph.triage_node", new=mock_triage),
            patch(
                "src.graphs.appointments_graph.appointment_collect_node",
                new=mock_collect,
            ),
            patch(
                "src.graphs.appointments_graph.availability_lookup_node",
                new=mock_avail,
            ),
            patch(
                "src.graphs.appointments_graph.reservation_propose_node",
                new=mock_propose,
            ),
        ):
            graph = build_appointments_graph(MemorySaver())
            chunks = []
            async for c in graph.astream(
                {
                    "messages": [HumanMessage(content="cita médica jueves 9am")],
                    "thread_id": "u:chain",
                    "user_context": {"name": "Ana"},
                },
                config=_config("u:chain"),
                stream_mode="updates",
            ):
                chunks.append(c)
        names = _node_names(chunks)
        assert "appointment_collect" in names
        assert "availability_lookup" in names
        assert "reservation_propose" in names

    @pytest.mark.asyncio
    async def test_slot_conflict_in_confirmation_loops_back_to_availability(self):
        # When confirmation reports slot_conflict the graph routes back to
        # availability_lookup so the user can pick again.
        mock_triage = AsyncMock(return_value={"intent": "booking"})
        mock_collect = AsyncMock(
            return_value={
                "messages": [AIMessage(content="ok")],
                "booking_intent": "book",
                "booking_complete": True,
            }
        )
        mock_avail = AsyncMock(
            return_value={
                "candidate_slots": [
                    {
                        "startsAt": "2026-07-01T10:00:00Z",
                        "endsAt": "2026-07-01T10:30:00Z",
                        "resources": [],
                    }
                ],
                "messages": [AIMessage(content="opciones")],
            }
        )
        mock_propose = AsyncMock(
            return_value={
                "chosen_slot": {
                    "startsAt": "2026-07-01T10:00:00Z",
                    "endsAt": "2026-07-01T10:30:00Z",
                    "resources": [],
                },
                "messages": [AIMessage(content="ok")],
            }
        )
        # First confirmation call reports a conflict (sends us back to
        # availability_lookup). The second succeeds so the graph terminates.
        mock_confirm = AsyncMock(
            side_effect=[
                {
                    "slot_conflict": True,
                    "chosen_slot": None,
                    "messages": [AIMessage(content="Justo lo tomaron...")],
                },
                {
                    "slot_conflict": False,
                    "reservation_appointment_id": "appt-1",
                    "messages": [AIMessage(content="Confirmado.")],
                },
            ]
        )
        with (
            patch("src.graphs.appointments_graph.triage_node", new=mock_triage),
            patch(
                "src.graphs.appointments_graph.appointment_collect_node",
                new=mock_collect,
            ),
            patch(
                "src.graphs.appointments_graph.availability_lookup_node",
                new=mock_avail,
            ),
            patch(
                "src.graphs.appointments_graph.reservation_propose_node",
                new=mock_propose,
            ),
            patch(
                "src.graphs.appointments_graph.confirmation_node",
                new=mock_confirm,
            ),
        ):
            graph = build_appointments_graph(MemorySaver())
            chunks = []
            async for c in graph.astream(
                {
                    "messages": [HumanMessage(content="cita")],
                    "thread_id": "u:conflict",
                    "user_context": {"name": "Ana"},
                },
                config=_config("u:conflict"),
                stream_mode="updates",
            ):
                chunks.append(c)
        names = _node_names(chunks)
        # availability_lookup appears twice — initial + after slot_conflict
        # bounceback.
        assert names.count("availability_lookup") >= 2
        assert "confirmation" in names

    @pytest.mark.asyncio
    async def test_in_progress_booking_intent_keeps_routing_to_collect(self):
        # When booking_intent="book" is already on state, the conditional
        # edge keeps using appointment_collect even if `intent` field is
        # blank (e.g. follow-up reply to interrupt).
        mock_triage = AsyncMock(return_value={})  # early-exit no-op
        mock_collect = AsyncMock(
            return_value={
                "messages": [AIMessage(content="...")],
            }
        )
        with (
            patch("src.graphs.appointments_graph.triage_node", new=mock_triage),
            patch(
                "src.graphs.appointments_graph.appointment_collect_node",
                new=mock_collect,
            ),
        ):
            graph = build_appointments_graph(MemorySaver())
            chunks = []
            async for c in graph.astream(
                {
                    "messages": [HumanMessage(content="dentista")],
                    "thread_id": "u:resume",
                    "booking_intent": "book",
                },
                config=_config("u:resume"),
                stream_mode="updates",
            ):
                chunks.append(c)
        names = _node_names(chunks)
        assert "appointment_collect" in names
