"""Wiring tests for the camila (academic-secretary) school graph.

All nodes are mocked; we only validate that the routing edges branch
correctly given representative triage outputs and inbound state. No real
LLM calls, no real backend HTTP — patched at module boundaries.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graphs.camila_graph import build_camila_graph
from src.schemas.intent import IntentType, StructuredIntent


def _node_names(results):
    return [list(r.keys())[0] for r in results if isinstance(r, dict)]


def _base_state(message: str, **overrides):
    state = {
        "messages": [HumanMessage(content=message)],
        "thread_id": "test:camila:1",
        "tenant_id": "tenant-1",
        "conversation_id": "conv-1",
        "contact_id": "contact-1",
        "user_context": {"name": "Patricia Hernández"},  # default: name known
        "school_name_captured": True,
        "attachments": [],
        "faqs": [],
        "agent_code_name": "camila",
        "agent_type": "school",
    }
    state.update(overrides)
    return state


def _config():
    return {
        "configurable": {
            "thread_id": "test:camila:1",
            "openai_api_key": "sk-test",
            "llm_provider": "openai",
            "llm_model": "gpt-test",
        }
    }


class TestCamilaRouting:
    @pytest.mark.asyncio
    async def test_payment_proof_routes_to_handoff(self):
        mock_triage = AsyncMock(
            return_value={
                "intent": IntentType.PAYMENT_PROOF.value,
                "structured_intent": StructuredIntent(
                    intent=IntentType.PAYMENT_PROOF, confidence=0.95
                ),
                "handoff_reason": IntentType.PAYMENT_PROOF.value,
            }
        )
        mock_handoff = AsyncMock(
            return_value={"messages": [AIMessage(content="Un asesor te contactará.")]}
        )
        mock_name = AsyncMock(return_value={})
        mock_faq = AsyncMock(return_value={})

        with (
            patch("src.graphs.camila_graph.camila_triage_node", new=mock_triage),
            patch("src.graphs.camila_graph.handoff_node", new=mock_handoff),
            patch("src.graphs.camila_graph.name_capture_node", new=mock_name),
            patch("src.graphs.camila_graph.faq_response_node", new=mock_faq),
        ):
            graph = build_camila_graph(MemorySaver())
            results = []
            async for chunk in graph.astream(
                _base_state("Envío transferencia FACTER 3"),
                config=_config(),
                stream_mode="updates",
            ):
                results.append(chunk)

        names = _node_names(results)
        assert "handoff" in names
        assert "faq_response" not in names
        assert "name_capture" not in names
        mock_handoff.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_attachment_routes_to_handoff_regardless_of_intent(self):
        # Even when triage returns FAQ, an inbound attachment forces handoff.
        mock_triage = AsyncMock(
            return_value={
                "intent": IntentType.FAQ.value,
                "structured_intent": StructuredIntent(
                    intent=IntentType.FAQ, confidence=0.9
                ),
            }
        )
        mock_handoff = AsyncMock(
            return_value={"messages": [AIMessage(content="Un asesor te contactará.")]}
        )
        mock_faq = AsyncMock(return_value={})

        with (
            patch("src.graphs.camila_graph.camila_triage_node", new=mock_triage),
            patch("src.graphs.camila_graph.handoff_node", new=mock_handoff),
            patch("src.graphs.camila_graph.faq_response_node", new=mock_faq),
        ):
            graph = build_camila_graph(MemorySaver())
            results = []
            async for chunk in graph.astream(
                _base_state(
                    "Aquí va mi recibo",
                    attachments=[{"type": "image", "mimeType": "image/jpeg"}],
                ),
                config=_config(),
                stream_mode="updates",
            ):
                results.append(chunk)

        names = _node_names(results)
        assert "handoff" in names
        assert "faq_response" not in names

    @pytest.mark.asyncio
    async def test_missing_name_routes_to_name_capture(self):
        mock_triage = AsyncMock(
            return_value={
                "intent": IntentType.GREETING.value,
                "structured_intent": StructuredIntent(
                    intent=IntentType.GREETING, confidence=0.99
                ),
            }
        )
        mock_name = AsyncMock(
            return_value={
                "messages": [AIMessage(content="¿Cuál es tu nombre completo?")]
            }
        )
        mock_handoff = AsyncMock(return_value={})
        mock_faq = AsyncMock(return_value={})

        with (
            patch("src.graphs.camila_graph.camila_triage_node", new=mock_triage),
            patch("src.graphs.camila_graph.name_capture_node", new=mock_name),
            patch("src.graphs.camila_graph.handoff_node", new=mock_handoff),
            patch("src.graphs.camila_graph.faq_response_node", new=mock_faq),
        ):
            graph = build_camila_graph(MemorySaver())
            results = []
            async for chunk in graph.astream(
                _base_state(
                    "Hola",
                    user_context={},
                    school_name_captured=False,
                ),
                config=_config(),
                stream_mode="updates",
            ):
                results.append(chunk)

        names = _node_names(results)
        assert "name_capture" in names
        assert "faq_response" not in names
        assert "handoff" not in names

    @pytest.mark.asyncio
    async def test_known_name_general_question_routes_to_faq(self):
        mock_triage = AsyncMock(
            return_value={
                "intent": IntentType.FAQ.value,
                "structured_intent": StructuredIntent(
                    intent=IntentType.FAQ, confidence=0.95
                ),
            }
        )
        mock_faq = AsyncMock(
            return_value={"messages": [AIMessage(content="Nuestro horario es...")]}
        )
        mock_handoff = AsyncMock(return_value={})
        mock_name = AsyncMock(return_value={})

        with (
            patch("src.graphs.camila_graph.camila_triage_node", new=mock_triage),
            patch("src.graphs.camila_graph.faq_response_node", new=mock_faq),
            patch("src.graphs.camila_graph.handoff_node", new=mock_handoff),
            patch("src.graphs.camila_graph.name_capture_node", new=mock_name),
        ):
            graph = build_camila_graph(MemorySaver())
            results = []
            async for chunk in graph.astream(
                _base_state("¿Cuál es el horario?"),
                config=_config(),
                stream_mode="updates",
            ):
                results.append(chunk)

        names = _node_names(results)
        assert "faq_response" in names
        assert "handoff" not in names
        assert "name_capture" not in names

    @pytest.mark.asyncio
    async def test_secondary_handoff_intent_promotes_to_handoff(self):
        # Simulates multi-intent: triage already promoted a secondary
        # CORRECTION_REQUEST to be the routing intent.
        mock_triage = AsyncMock(
            return_value={
                "intent": IntentType.CORRECTION_REQUEST.value,
                "structured_intent": StructuredIntent(
                    intent=IntentType.FAQ,
                    confidence=0.7,
                    secondary_intents=[IntentType.CORRECTION_REQUEST],
                ),
                "handoff_reason": IntentType.CORRECTION_REQUEST.value,
            }
        )
        mock_handoff = AsyncMock(
            return_value={"messages": [AIMessage(content="Un asesor te contactará.")]}
        )

        with (
            patch("src.graphs.camila_graph.camila_triage_node", new=mock_triage),
            patch("src.graphs.camila_graph.handoff_node", new=mock_handoff),
        ):
            graph = build_camila_graph(MemorySaver())
            results = []
            async for chunk in graph.astream(
                _base_state(
                    "¿Cuál es el horario? Y también necesito corregir mi apellido"
                ),
                config=_config(),
                stream_mode="updates",
            ):
                results.append(chunk)

        names = _node_names(results)
        assert "handoff" in names


class TestIdentityValidator:
    def test_detects_two_distinct_full_names(self):
        from langchain_core.messages import HumanMessage

        from src.agents.identity_validator import detect_identity_conflict

        msgs = [
            HumanMessage(content="Mi nombre es Fabián Alberto Neiva Cortés"),
            HumanMessage(content="Disculpe, en el documento aparece Neiva Torres"),
        ]
        assert detect_identity_conflict(msgs) is True

    def test_no_conflict_for_single_name(self):
        from langchain_core.messages import HumanMessage

        from src.agents.identity_validator import detect_identity_conflict

        msgs = [HumanMessage(content="Soy Patricia Hernández, cédula 1023456789")]
        assert detect_identity_conflict(msgs) is False

    def test_detects_two_distinct_documentos(self):
        from langchain_core.messages import HumanMessage

        from src.agents.identity_validator import detect_identity_conflict

        msgs = [
            HumanMessage(content="Mi cédula es 80844411"),
            HumanMessage(content="Perdón, la correcta es 1023456789"),
        ]
        assert detect_identity_conflict(msgs) is True
