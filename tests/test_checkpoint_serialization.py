"""Checkpoint state must stay JSON-native.

LangGraph will block deserializing unregistered custom Python types from
checkpoints. Triage nodes therefore store `structured_intent` as a plain
`model_dump(mode="json")` dict — never a Pydantic instance. Legacy
checkpoints written before this change are allow-listed in main.py.
"""

import json
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from src.agents.camila_triage import camila_triage_node
from src.agents.triage import triage_node
from src.providers.base import UsageInfo


class _FakeProvider:
    name = "fake"

    def __init__(self, reply: str):
        self.reply = reply
        self.last_usage = UsageInfo(input_tokens=5, output_tokens=5)

    def stream_chat(self, *, model: str, messages: list):
        async def _gen():
            yield self.reply

        return _gen()


def _assert_json_native(value):
    # json.dumps raises on anything that isn't a plain JSON value —
    # exactly the property checkpoints need.
    json.dumps(value)


async def test_triage_stores_structured_intent_as_plain_dict():
    provider = _FakeProvider(
        '{"intent": "sales", "confidence": 0.9, "raw_text": "quiero comprar"}'
    )
    state = {
        "messages": [HumanMessage(content="quiero comprar")],
        "thread_id": "t1",
        "agent_type": "sales",
        "agent_code_name": "helena",
        "contact_id": "",
        "faqs": [],
    }
    with (
        patch("src.agents.triage.get_provider", return_value=provider),
        patch("src.agents.triage.resolve_model", return_value="m"),
        patch("src.agents.triage.get_stream_writer", return_value=lambda e: None),
    ):
        update = await triage_node(state, {"configurable": {}})

    structured = update["structured_intent"]
    assert isinstance(structured, dict)
    assert structured["intent"] == "sales"  # str, not IntentType
    _assert_json_native(structured)


async def test_camila_triage_stores_structured_intent_as_plain_dict():
    provider = _FakeProvider(
        '{"intent": "greeting", "confidence": 0.9, "secondary_intents": [], "raw_text": "hola"}'
    )
    state = {
        "messages": [HumanMessage(content="hola")],
        "thread_id": "t2",
        "faqs": [],
    }
    with (
        patch("src.agents.camila_triage.get_provider", return_value=provider),
        patch("src.agents.camila_triage.resolve_model", return_value="m"),
        patch(
            "src.agents.camila_triage.get_stream_writer",
            return_value=lambda e: None,
        ),
    ):
        update = await camila_triage_node(state, {"configurable": {}})

    structured = update["structured_intent"]
    assert isinstance(structured, dict)
    assert structured["intent"] == "greeting"
    _assert_json_native(structured)


async def test_camila_identity_conflict_shortcircuit_is_plain_dict():
    # Two distinct full names across user messages → deterministic
    # IDENTITY_CONFLICT, no LLM call — must also store a plain dict.
    state = {
        "messages": [
            HumanMessage(content="El estudiante es Neiva Cortés"),
            HumanMessage(content="Perdón, es Neiva Torres"),
        ],
        "thread_id": "t3",
        "faqs": [],
    }
    with patch(
        "src.agents.camila_triage.get_stream_writer", return_value=lambda e: None
    ):
        update = await camila_triage_node(state, {"configurable": {}})

    structured = update["structured_intent"]
    assert isinstance(structured, dict)
    assert structured["intent"] == "identity_conflict"
    _assert_json_native(structured)
