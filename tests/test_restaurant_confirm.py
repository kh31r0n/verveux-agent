"""Unit tests for restaurant_confirm — LLM decision handling, keyword fallback
on LLM failure, and the state contract each branch returns."""

from __future__ import annotations

import json
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.restaurant_confirm import restaurant_confirm_node
from src.providers.base import UsageInfo

_CONFIG = {"configurable": {"thread_id": "t:1", "openai_api_key": "sk-test"}}

_CART = [
    {"product_id": "p-tacos", "name": "Tacos al pastor", "qty": 2, "price": 12.5, "notes": ""},
]


class _FakeProvider:
    """Call 1 = classification, call 2 = user-facing reply."""

    name = "fake"

    def __init__(self, classification: str | Exception, reply: str = "¡Gracias! Tu pedido va en camino."):
        self.classification = classification
        self.reply = reply
        self._calls = 0
        self.last_usage = UsageInfo(input_tokens=10, output_tokens=5)

    def stream_chat(self, *, model: str, messages: list[dict]):
        self._calls += 1
        is_classify = self._calls == 1
        classification = self.classification
        reply = self.reply

        async def _gen():
            if is_classify and isinstance(classification, Exception):
                raise classification
            yield classification if is_classify else reply

        return _gen()


def _state(message: str, **overrides) -> dict:
    state = {
        "messages": [
            AIMessage(content="Resumen del pedido... Responde confirmar para enviarlo."),
            HumanMessage(content=message),
        ],
        "thread_id": "t:1",
        "conversation_id": "conv-1",
        "contact_id": "contact-1",
        "cart": list(_CART),
        "restaurant_order_data": {"service_type": "pickup"},
        "restaurant_phase": "confirmation",
        "user_context": {"name": "Ana"},
        "language": "es",
        "agent_type": "restaurant",
        "agent_code_name": "giulia",
    }
    state.update(overrides)
    return state


async def _run(state: dict, classification: str | Exception) -> dict:
    provider = _FakeProvider(classification)
    with (
        patch("src.agents.restaurant_confirm.get_provider", return_value=provider),
        patch("src.agents.restaurant_confirm.resolve_model", return_value="test-model"),
        patch("src.agents.restaurant_confirm.get_stream_writer", return_value=lambda *_: None),
    ):
        return await restaurant_confirm_node(state, _CONFIG)


class TestLlmDecisions:
    async def test_fenced_confirm_decision(self):
        """Gemini-style fenced JSON classification must parse."""
        raw = '```json\n' + json.dumps({"decision": "confirm", "confidence": 0.95}) + "\n```"
        result = await _run(_state("confirmar"), raw)

        assert result["restaurant_order_confirmed"] is True
        assert result["messages"], "confirm branch replies before checkout"
        assert len(result["turn_usage"]) == 2  # classify + reply

    async def test_modify_returns_to_collect_without_reply(self):
        result = await _run(_state("mejor quita el flan"), json.dumps({"decision": "modify"}))

        assert result["restaurant_order_confirmed"] is False
        assert result["restaurant_phase"] == "collect"
        assert result["restaurant_order_complete"] is False
        assert "messages" not in result, "order_collect answers the same turn"
        assert len(result["turn_usage"]) == 1  # classify only

    async def test_unclear_reasks_without_flag_changes(self):
        result = await _run(_state("¿tienen salsa verde?"), json.dumps({"decision": "unclear"}))

        assert "restaurant_order_confirmed" not in result
        assert "restaurant_phase" not in result
        assert result["messages"], "unclear branch re-asks"

    async def test_invalid_decision_falls_back_to_keywords(self):
        result = await _run(_state("sí, confirmo"), json.dumps({"decision": "banana"}))
        assert result["restaurant_order_confirmed"] is True


class TestKeywordFallback:
    async def test_llm_failure_with_yes_keywords_confirms(self):
        result = await _run(_state("sí, confirmo"), RuntimeError("provider down"))
        assert result["restaurant_order_confirmed"] is True

    async def test_llm_failure_with_no_keywords_modifies(self):
        result = await _run(_state("no, quiero cambiar algo"), RuntimeError("provider down"))
        assert result["restaurant_phase"] == "collect"
        assert result["restaurant_order_confirmed"] is False

    async def test_llm_failure_ambiguous_is_unclear(self):
        result = await _run(_state("mmm 🤔"), RuntimeError("provider down"))
        assert "restaurant_order_confirmed" not in result
        assert result["messages"]
