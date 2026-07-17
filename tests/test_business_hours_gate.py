"""Business-hours gating on the Python side.

Outside the tenant's business hours (backend-resolved `within_business_hours`
flag, forwarded per turn) every graph routes non-urgent turns to faq_response,
which splices the tenant-editable `{AGENT_TYPE}_OUTSIDE_HOURS` prompt so the
customer is told the business is closed and the model stays FAQ-only. Urgent
intents (URGENT_INTENTS) and camila's handoff branches are exempt. Absent flag
(old checkpoints / older backend) behaves as today.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agents.business_hours_gate import (
    DEFAULT_OUTSIDE_HOURS_INSTRUCTION,
    within_business_hours,
)
from src.agents.faq_response import faq_response_node
from src.graphs.shared_routing import should_restrict_to_faq
from src.providers.base import UsageInfo


# ── Predicate + routing policy (pure functions) ─────────────────────────────


class TestPredicate:
    def test_absent_flag_is_within_hours(self):
        assert within_business_hours({}) is True

    def test_explicit_true(self):
        assert within_business_hours({"within_business_hours": True}) is True

    def test_explicit_false(self):
        assert within_business_hours({"within_business_hours": False}) is False


class TestShouldRestrictToFaq:
    def test_within_hours_never_restricts(self):
        assert should_restrict_to_faq({"within_business_hours": True, "intent": "sales"}) is False
        assert should_restrict_to_faq({"intent": "sales"}) is False  # absent flag

    def test_outside_hours_restricts_operational_intents(self):
        for intent in ("sales", "order", "booking", "lead_capture", "faq", "greeting", ""):
            state = {"within_business_hours": False, "intent": intent}
            assert should_restrict_to_faq(state) is True, intent

    def test_outside_hours_exempts_urgent_intents(self):
        for intent in ("complaint", "escalation", "payment_proof", "appointment_cancel"):
            state = {"within_business_hours": False, "intent": intent}
            assert should_restrict_to_faq(state) is False, intent

    def test_explicit_intent_argument_wins_over_state(self):
        state = {"within_business_hours": False, "intent": "sales"}
        assert should_restrict_to_faq(state, "escalation") is False


# ── faq_response prompt splice ──────────────────────────────────────────────


class _CapturingProvider:
    name = "fake"

    def __init__(self) -> None:
        self.captured_messages: list[dict] | None = None
        self.last_usage = UsageInfo(input_tokens=10, output_tokens=5)

    def stream_chat(self, *, model: str, messages: list[dict]):
        self.captured_messages = messages

        async def _gen():
            yield "ok"

        return _gen()


def _faq_state(*, within_hours, catalog_access=True) -> dict:
    state: dict = {
        "messages": [HumanMessage(content="¿Tienen envíos a Cali?")],
        "thread_id": "t:helena:1",
        "agent_type": "sales",
        "faqs": [],
        "user_context": {},
        "product_catalog": [],
        "catalog_access_enabled": catalog_access,
    }
    if within_hours is not None:
        state["within_business_hours"] = within_hours
    return state


async def _run_faq(state: dict, config: dict | None = None) -> _CapturingProvider:
    provider = _CapturingProvider()
    with (
        patch("src.agents.faq_response.get_stream_writer", return_value=(lambda e: None)),
        patch("src.agents.faq_response.get_provider", return_value=provider),
        patch("src.agents.faq_response.resolve_model", return_value="m"),
    ):
        await faq_response_node(state, config or {"configurable": {}})
    assert provider.captured_messages is not None
    return provider


class TestFaqResponseSplice:
    async def test_outside_hours_appends_default_instruction(self):
        provider = await _run_faq(_faq_state(within_hours=False))
        sys = provider.captured_messages[0]["content"]
        assert DEFAULT_OUTSIDE_HOURS_INSTRUCTION in sys

    async def test_within_hours_omits_instruction(self):
        provider = await _run_faq(_faq_state(within_hours=True))
        assert "FUERA DE HORARIO" not in provider.captured_messages[0]["content"]

    async def test_absent_flag_omits_instruction(self):
        provider = await _run_faq(_faq_state(within_hours=None))
        assert "FUERA DE HORARIO" not in provider.captured_messages[0]["content"]

    async def test_tenant_custom_prompt_wins_over_default(self):
        config = {
            "configurable": {
                "prompts": {
                    "SALES_OUTSIDE_HOURS": {"content": "CERRADO — solo FAQs personalizado."}
                }
            }
        }
        provider = await _run_faq(_faq_state(within_hours=False), config)
        sys = provider.captured_messages[0]["content"]
        assert "CERRADO — solo FAQs personalizado." in sys
        assert DEFAULT_OUTSIDE_HOURS_INSTRUCTION not in sys

    async def test_coexists_with_catalog_degraded_block(self):
        provider = await _run_faq(
            _faq_state(within_hours=False, catalog_access=False)
        )
        sys = provider.captured_messages[0]["content"]
        assert "SIN ACCESO AL CATÁLOGO" in sys
        assert DEFAULT_OUTSIDE_HOURS_INSTRUCTION in sys


# ── Graph routing (mirrors test_leads_graph.py) ─────────────────────────────


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
    return {"configurable": {"thread_id": thread_id, "openai_api_key": "sk-test"}}


async def _run(graph, inputs: dict, thread_id: str) -> list[str]:
    chunks = []
    async for c in graph.astream(inputs, config=_config(thread_id), stream_mode="updates"):
        chunks.append(c)
    return _node_names(chunks)


def _inputs(text: str, thread_id: str, **extra) -> dict:
    return {
        "messages": [HumanMessage(content=text)],
        "thread_id": thread_id,
        "user_context": {"name": "Ana"},
        "within_business_hours": False,
        **extra,
    }


_FAQ_REPLY = AsyncMock(return_value={"messages": [AIMessage(content="Estamos fuera de horario.")]})


class TestSalesGraphGating:
    async def test_outside_hours_sales_intent_goes_to_faq(self):
        from src.graphs.sales_graph import build_sales_graph

        with (
            patch("src.graphs.sales_graph.triage_node", new=AsyncMock(return_value={"intent": "sales"})),
            patch("src.graphs.sales_graph.faq_response_node", new=_FAQ_REPLY),
        ):
            graph = build_sales_graph(MemorySaver())
            names = await _run(graph, _inputs("quiero comprar", "s:1"), "s:1")
        assert "faq_response" in names
        assert "sales_collect" not in names

    async def test_within_hours_sales_intent_unchanged(self):
        from src.graphs.sales_graph import build_sales_graph

        mock_collect = AsyncMock(return_value={"messages": [AIMessage(content="¿Qué deseas?")]})
        with (
            patch("src.graphs.sales_graph.triage_node", new=AsyncMock(return_value={"intent": "sales"})),
            patch("src.graphs.sales_graph.sales_collect_node", new=mock_collect),
        ):
            graph = build_sales_graph(MemorySaver())
            names = await _run(
                graph, _inputs("quiero comprar", "s:2", within_business_hours=True), "s:2"
            )
        assert "sales_collect" in names
        assert "faq_response" not in names

    async def test_outside_hours_complaint_still_routes_to_complaint(self):
        from src.graphs.sales_graph import build_sales_graph

        mock_complaint = AsyncMock(return_value={"messages": [AIMessage(content="Lo siento…")]})
        with (
            patch("src.graphs.sales_graph.triage_node", new=AsyncMock(return_value={"intent": "complaint"})),
            patch("src.graphs.sales_graph.complaint_collect_node", new=mock_complaint),
        ):
            graph = build_sales_graph(MemorySaver())
            names = await _run(graph, _inputs("tengo un reclamo", "s:3"), "s:3")
        assert "complaint_collect" in names
        assert "faq_response" not in names


class TestRestaurantGraphGating:
    async def test_outside_hours_cuts_off_mid_order(self):
        # Immediate cutoff: the cart is populated (in-progress order) and the
        # stale intent is "order", yet the turn lands in faq_response. The
        # cart survives in the checkpoint for when hours reopen.
        from src.graphs.restaurant_graph import build_restaurant_graph

        with (
            patch("src.graphs.restaurant_graph.triage_node", new=AsyncMock(return_value={})),
            patch("src.graphs.restaurant_graph.faq_response_node", new=_FAQ_REPLY),
        ):
            graph = build_restaurant_graph(MemorySaver())
            names = await _run(
                graph,
                _inputs(
                    "y también unas papas",
                    "r:1",
                    intent="order",
                    cart=[{"product_id": "p1", "qty": 1}],
                ),
                "r:1",
            )
        assert "faq_response" in names
        assert "order_collect" not in names


class TestAppointmentsGraphGating:
    async def test_outside_hours_cuts_off_mid_booking(self):
        from src.graphs.appointments_graph import build_appointments_graph

        with (
            patch("src.graphs.appointments_graph.triage_node", new=AsyncMock(return_value={})),
            patch("src.graphs.appointments_graph.faq_response_node", new=_FAQ_REPLY),
        ):
            graph = build_appointments_graph(MemorySaver())
            names = await _run(
                graph,
                _inputs("la de las 3pm", "a:1", intent="booking", booking_intent="book"),
                "a:1",
            )
        assert "faq_response" in names
        assert "appointment_collect" not in names

    async def test_outside_hours_escalation_still_routes(self):
        from src.graphs.appointments_graph import build_appointments_graph

        mock_esc = AsyncMock(return_value={"messages": [AIMessage(content="Te comunico…")]})
        with (
            patch("src.graphs.appointments_graph.triage_node", new=AsyncMock(return_value={"intent": "escalation"})),
            patch("src.graphs.appointments_graph.escalation_node", new=mock_esc),
        ):
            graph = build_appointments_graph(MemorySaver())
            names = await _run(graph, _inputs("quiero hablar con un humano", "a:2"), "a:2")
        assert "escalation" in names
        assert "faq_response" not in names


class TestSchoolGraphGating:
    async def test_outside_hours_course_inquiry_goes_to_faq(self):
        from src.graphs.school_graph import build_school_graph

        with (
            patch("src.graphs.school_graph.triage_node", new=AsyncMock(return_value={"intent": "course_inquiry"})),
            patch("src.graphs.school_graph.faq_response_node", new=_FAQ_REPLY),
        ):
            graph = build_school_graph(MemorySaver())
            names = await _run(graph, _inputs("¿qué cursos tienen?", "e:1"), "e:1")
        assert "faq_response" in names
        assert "course_inquiry" not in names


class TestLeadsGraphGating:
    async def test_outside_hours_lead_capture_goes_to_faq(self):
        from src.graphs.leads_graph import build_leads_graph

        with (
            patch("src.graphs.leads_graph.triage_node", new=AsyncMock(return_value={"intent": "lead_capture"})),
            patch("src.graphs.leads_graph.faq_response_node", new=_FAQ_REPLY),
        ):
            graph = build_leads_graph(MemorySaver())
            names = await _run(graph, _inputs("me interesa el CRM", "l:1"), "l:1")
        assert "faq_response" in names
        assert "lead_collect" not in names


class TestCrossTurnIsolation:
    async def test_hours_reopening_unsticks_the_conversation(self):
        # Turn 1 runs outside hours and persists within_business_hours=False
        # in the checkpoint. Turn 2 arrives within hours: chat_stream always
        # overwrites the flag in the inputs dict, and that overwrite must win
        # over the checkpointed False — otherwise the conversation would stay
        # stuck in FAQ-only mode after the business reopens.
        from src.graphs.sales_graph import build_sales_graph

        mock_collect = AsyncMock(return_value={"messages": [AIMessage(content="¿Qué deseas?")]})
        with (
            patch("src.graphs.sales_graph.triage_node", new=AsyncMock(return_value={"intent": "sales"})),
            patch("src.graphs.sales_graph.sales_collect_node", new=mock_collect),
            patch("src.graphs.sales_graph.faq_response_node", new=_FAQ_REPLY),
        ):
            graph = build_sales_graph(MemorySaver())
            thread = "s:cross"
            names1 = await _run(graph, _inputs("quiero comprar", thread), thread)
            assert "faq_response" in names1
            assert "sales_collect" not in names1

            names2 = await _run(
                graph,
                _inputs("sigo aquí, quiero comprar", thread, within_business_hours=True),
                thread,
            )
            assert "sales_collect" in names2
            assert "faq_response" not in names2


class TestCamilaGraphGating:
    async def test_outside_hours_attachment_still_hands_off(self):
        from src.graphs.camila_graph import build_camila_graph

        mock_handoff = AsyncMock(return_value={"messages": [AIMessage(content="Lo reviso con un humano.")]})
        with (
            patch("src.graphs.camila_graph.camila_triage_node", new=AsyncMock(return_value={})),
            patch("src.graphs.camila_graph.handoff_node", new=mock_handoff),
        ):
            graph = build_camila_graph(MemorySaver())
            names = await _run(
                graph,
                _inputs("le envío el comprobante", "c:1", attachments=[{"type": "image"}]),
                "c:1",
            )
        assert "handoff" in names
        assert "faq_response" not in names

    async def test_outside_hours_plain_question_goes_to_faq(self):
        from src.graphs.camila_graph import build_camila_graph

        with (
            patch("src.graphs.camila_graph.camila_triage_node", new=AsyncMock(return_value={"intent": "faq"})),
            patch("src.graphs.camila_graph.faq_response_node", new=_FAQ_REPLY),
        ):
            graph = build_camila_graph(MemorySaver())
            names = await _run(graph, _inputs("¿cuándo inician clases?", "c:2"), "c:2")
        assert "faq_response" in names
