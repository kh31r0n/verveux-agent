"""Tests for the query_normalizer node and its deterministic trigger.

The trigger (`should_normalize`) is a pure function — tested as a matrix.
The node is tested with a fake provider and a patched backend client: no
real LLM calls, no real HTTP.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agents.query_normalizer import query_normalizer_node
from src.config import settings
from src.graphs.shared_routing import (
    FAQ_SCORE_TRIGGER_THRESHOLD,
    contains_escalation_keywords,
    should_normalize,
)
from src.providers.base import UsageInfo


class _FakeProvider:
    name = "fake"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0
        self.last_usage = UsageInfo(input_tokens=5, output_tokens=5)

    def stream_chat(self, *, model: str, messages: list):
        self.calls += 1

        async def _gen():
            yield self.reply

        return _gen()


def _state(message: str = "quales son los orarios?", **overrides) -> dict:
    state = {
        "messages": [HumanMessage(content=message)],
        "thread_id": "test:norm:1",
        "tenant_id": "tenant-1",
        "conversation_id": "conv-1",
        "faqs": [],
    }
    state.update(overrides)
    return state


def _config(enabled: bool = True) -> dict:
    return {"configurable": {"normalization_enabled": enabled}}


def _llm_reply(
    corrected: str = "¿cuáles son los horarios?",
    confidence: float = 0.95,
    risk: str = "LOW",
) -> str:
    return json.dumps(
        {
            "corrected_text": corrected,
            "confidence": confidence,
            "changed_meaning_risk": risk,
            "reason": "typos",
        }
    )


# ── Trigger matrix ───────────────────────────────────────────────────────────


class TestShouldNormalize:
    @pytest.mark.parametrize(
        "text,faqs,flag,expected",
        [
            # Flag off → never
            ("quales son los orarios", [], False, False),
            # Too short
            ("ok", [], True, False),
            ("sí", [], True, False),
            # Escalation keywords suppress
            ("quiero hablar con un humano", [], True, False),
            ("necesito un asesor urgente", [], True, False),
            ("i want a refund now", [], True, False),
            # Strong retrieval → no need
            ("quales son los orarios", [{"score": 0.5}], True, False),
            (
                "quales son los orarios",
                [{"score": FAQ_SCORE_TRIGGER_THRESHOLD}],
                True,
                False,
            ),
            # Empty or marginal retrieval → trigger
            ("quales son los orarios", [], True, True),
            ("quales son los orarios", [{"score": 0.05}], True, True),
            # Missing score treated as 0
            ("quales son los orarios", [{"question": "q"}], True, True),
        ],
    )
    def test_matrix(self, text, faqs, flag, expected):
        assert should_normalize(text, faqs, flag) is expected

    def test_global_kill_switch(self, monkeypatch):
        monkeypatch.setattr(settings, "query_normalization_enabled", False)
        assert should_normalize("quales son los orarios", [], True) is False

    def test_escalation_keyword_detection(self):
        assert contains_escalation_keywords("Necesito hablar con un HUMANO") is True
        assert contains_escalation_keywords("¿Cuáles son los horarios?") is False
        assert contains_escalation_keywords("") is False
        assert contains_escalation_keywords(None) is False


# ── Node behavior ────────────────────────────────────────────────────────────


class TestQueryNormalizerNode:
    async def test_no_trigger_fast_path_skips_llm(self):
        provider = _FakeProvider(_llm_reply())
        state = _state(faqs=[{"score": 0.9, "question": "q", "answer": "a"}])
        with patch(
            "src.agents.query_normalizer.get_provider", return_value=provider
        ) as get_provider_mock:
            update = await query_normalizer_node(state, _config(True))

        get_provider_mock.assert_not_called()
        assert provider.calls == 0
        assert update["original_text"] == "quales son los orarios?"
        assert update["normalization"]["applied"] is False
        assert update["normalization"]["reason"] == "no_trigger"
        assert "turn_usage" not in update
        assert "faqs" not in update
        json.dumps(update["normalization"])

    async def test_flag_off_is_no_trigger(self):
        update = await query_normalizer_node(_state(), _config(False))
        assert update["normalization"]["enabled"] is False
        assert update["normalization"]["applied"] is False

    async def test_applied_path_overwrites_faqs_and_records_usage(self):
        provider = _FakeProvider(_llm_reply())
        backend_faqs = [
            {
                "id": "f1",
                "question": "¿Cuáles son los horarios?",
                "answer": "L-V 8-17",
                "category": "general",
                "score": 0.42,
            }
        ]
        with (
            patch("src.agents.query_normalizer.get_provider", return_value=provider),
            patch("src.agents.query_normalizer.resolve_model", return_value="m"),
            patch(
                "src.agents.backend_client.search_faqs",
                new=AsyncMock(return_value=backend_faqs),
            ) as search_mock,
        ):
            update = await query_normalizer_node(_state(), _config(True))

        search_mock.assert_awaited_once_with(
            "conv-1", "¿cuáles son los horarios?", limit=5
        )
        norm = update["normalization"]
        assert norm["applied"] is True
        assert norm["corrected_text"] == "¿cuáles son los horarios?"
        assert norm["model"] == "m"
        assert update["faqs"][0]["question"] == "¿Cuáles son los horarios?"
        assert update["faqs"][0]["score"] == 0.42
        assert len(update["turn_usage"]) == 1
        assert update["turn_usage"][0]["node"] == "query_normalizer"
        json.dumps(norm)

    @pytest.mark.parametrize(
        "reply,expected_reason",
        [
            (_llm_reply(risk="HIGH"), None),
            (_llm_reply(confidence=0.3), None),
            ("not json at all", "parse_error"),
        ],
    )
    async def test_discard_paths_keep_faqs_but_record_usage(
        self, reply, expected_reason
    ):
        provider = _FakeProvider(reply)
        original_faqs = [{"question": "q", "answer": "a", "score": 0.05}]
        with (
            patch("src.agents.query_normalizer.get_provider", return_value=provider),
            patch("src.agents.query_normalizer.resolve_model", return_value="m"),
            patch(
                "src.agents.backend_client.search_faqs", new=AsyncMock()
            ) as search_mock,
        ):
            update = await query_normalizer_node(
                _state(faqs=original_faqs), _config(True)
            )

        search_mock.assert_not_awaited()
        assert update["normalization"]["applied"] is False
        if expected_reason:
            assert update["normalization"]["reason"] == expected_reason
        assert "faqs" not in update  # original state faqs untouched
        assert len(update["turn_usage"]) == 1  # tokens were spent

    async def test_unchanged_text_not_applied(self):
        provider = _FakeProvider(
            _llm_reply(corrected="quales son los orarios?", confidence=1.0)
        )
        with (
            patch("src.agents.query_normalizer.get_provider", return_value=provider),
            patch("src.agents.query_normalizer.resolve_model", return_value="m"),
        ):
            update = await query_normalizer_node(_state(), _config(True))
        assert update["normalization"]["applied"] is False
        assert update["normalization"]["reason"] == "unchanged"

    async def test_retrieval_failure_keeps_original_faqs(self):
        provider = _FakeProvider(_llm_reply())
        with (
            patch("src.agents.query_normalizer.get_provider", return_value=provider),
            patch("src.agents.query_normalizer.resolve_model", return_value="m"),
            patch(
                "src.agents.backend_client.search_faqs",
                new=AsyncMock(side_effect=RuntimeError("backend down")),
            ),
        ):
            update = await query_normalizer_node(_state(), _config(True))

        norm = update["normalization"]
        assert norm["applied"] is False
        assert norm["reason"] == "retrieval_failed"
        assert "faqs" not in update
        assert len(update["turn_usage"]) == 1

    async def test_retrieval_empty_keeps_original_faqs_but_stays_applied(self):
        provider = _FakeProvider(_llm_reply())
        with (
            patch("src.agents.query_normalizer.get_provider", return_value=provider),
            patch("src.agents.query_normalizer.resolve_model", return_value="m"),
            patch(
                "src.agents.backend_client.search_faqs",
                new=AsyncMock(return_value=[]),
            ),
        ):
            update = await query_normalizer_node(_state(), _config(True))

        assert update["normalization"]["applied"] is True
        assert "faqs" not in update

    async def test_legacy_state_without_new_keys(self):
        # Old checkpoints carry neither original_text nor normalization.
        state = {
            "messages": [HumanMessage(content="hola necesito ayuda")],
            "faqs": [{"question": "q", "answer": "a", "score": 0.9}],
        }
        update = await query_normalizer_node(state, {"configurable": {}})
        assert update["normalization"]["applied"] is False
        json.dumps(update)

    async def test_reads_trailing_burst_not_only_last_message(self):
        # Coalesced fragments that landed as separate checkpoint entries.
        state = _state()
        state["messages"] = [
            AIMessage(content="¡Hola! ¿En qué te ayudo?"),
            HumanMessage(content="quales son"),
            HumanMessage(content="los orarios?"),
        ]
        update = await query_normalizer_node(state, _config(False))
        assert "quales son" in update["original_text"]
        assert "los orarios?" in update["original_text"]


# ── Write-once provenance: no other node writes the new keys ────────────────


class TestProvenanceOwnership:
    async def test_triage_never_writes_normalization_keys(self):
        from src.agents.triage import triage_node

        provider = _FakeProvider(
            '{"intent": "faq", "confidence": 0.9, "raw_text": "horarios"}'
        )
        state = _state(
            normalization={"applied": True, "corrected_text": "x"},
            original_text="y",
            agent_type="sales",
            agent_code_name="helena",
            contact_id="",
        )
        with (
            patch("src.agents.triage.get_provider", return_value=provider),
            patch("src.agents.triage.resolve_model", return_value="m"),
            patch(
                "src.agents.triage.get_stream_writer", return_value=lambda e: None
            ),
        ):
            update = await triage_node(state, {"configurable": {}})

        assert "normalization" not in update
        assert "original_text" not in update


# ── Graph wiring: normalizer runs first in every graph ──────────────────────


class TestGraphWiring:
    @pytest.mark.parametrize(
        "module,builder,triage_attr",
        [
            ("src.graphs.camila_graph", "build_camila_graph", "camila_triage_node"),
            ("src.graphs.sales_graph", "build_sales_graph", "triage_node"),
            ("src.graphs.school_graph", "build_school_graph", "triage_node"),
            ("src.graphs.restaurant_graph", "build_restaurant_graph", "triage_node"),
            (
                "src.graphs.appointments_graph",
                "build_appointments_graph",
                "triage_node",
            ),
        ],
    )
    async def test_normalizer_runs_before_triage(self, module, builder, triage_attr):
        import importlib

        mod = importlib.import_module(module)
        mock_norm = AsyncMock(return_value={"original_text": "hola"})
        # Triage replies with an AI message and ends the turn via faq/greeting
        # routing being mocked too — we only assert ordering of the first two.
        mock_triage = AsyncMock(
            return_value={
                "intent": "faq",
                "messages": [AIMessage(content="ok")],
            }
        )
        mock_faq = AsyncMock(return_value={"messages": [AIMessage(content="resp")]})

        with (
            patch(f"{module}.query_normalizer_node", new=mock_norm),
            patch(f"{module}.{triage_attr}", new=mock_triage),
            patch(f"{module}.faq_response_node", new=mock_faq),
        ):
            graph = getattr(mod, builder)(MemorySaver())
            results = []
            async for chunk in graph.astream(
                _state(
                    user_context={"name": "Ana"},
                    school_name_captured=True,
                    agent_type="sales",
                    agent_code_name="test",
                ),
                config={
                    "configurable": {
                        "thread_id": f"test:{builder}",
                        "openai_api_key": "sk-test",
                    }
                },
                stream_mode="updates",
            ):
                results.append(chunk)

        node_order = [list(r.keys())[0] for r in results if isinstance(r, dict)]
        assert node_order[0] == "query_normalizer"
        assert mock_norm.await_count == 1
