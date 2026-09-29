"""faq_response_node must report the FAQs it injected as `faq_used`,
carrying the backend-provided id + retrieval score so NestJS can log usage
by stable id instead of matching question text against LLM output.
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import HumanMessage

from src.agents.faq_response import faq_response_node
from src.providers.base import UsageInfo


class _FakeProvider:
    name = "fake"

    def __init__(self, chunks: tuple[str, ...] = ("Claro, con gusto.",)):
        self.chunks = chunks
        self.last_usage = UsageInfo(input_tokens=10, output_tokens=5)

    def stream_chat(self, *, model: str, messages: list[dict]):
        async def _gen():
            for chunk in self.chunks:
                yield chunk

        return _gen()


def _faq(faq_id: str, question: str, score: float) -> dict:
    return {
        "id": faq_id,
        "question": question,
        "answer": f"respuesta de {question}",
        "category": "general",
        "priority": 0,
        "score": score,
    }


def _state(faqs: list[dict]) -> dict:
    return {
        "messages": [HumanMessage(content="¿Cuáles son los horarios?")],
        "thread_id": "test:helena:1",
        "agent_type": "sales",
        "faqs": faqs,
        "user_context": {},
    }


async def _call_node(state: dict) -> dict:
    events: list[dict] = []
    with (
        patch(
            "src.agents.faq_response.get_stream_writer",
            return_value=(lambda evt: events.append(evt)),
        ),
        patch(
            "src.agents.faq_response.get_provider",
            return_value=_FakeProvider(),
        ),
        patch("src.agents.faq_response.resolve_model", return_value="model-test"),
    ):
        return await faq_response_node(state, {"configurable": {}})


class TestFaqUsedReporting:
    async def test_reports_injected_faqs_with_id_and_score(self):
        state = _state(
            [
                _faq("f-1", "¿Horarios?", 0.42),
                _faq("f-2", "¿Envíos?", 0.17),
            ]
        )
        result = await _call_node(state)

        assert result["faq_used"] == [
            {"id": "f-1", "question": "¿Horarios?", "confidence": 0.42},
            {"id": "f-2", "question": "¿Envíos?", "confidence": 0.17},
        ]

    async def test_no_faqs_injected_reports_none(self):
        result = await _call_node(_state([]))
        assert result["faq_used"] is None

    async def test_missing_id_passes_through_without_inventing_one(self):
        faq = _faq("", "¿Horarios?", 0.3)
        faq["id"] = ""
        result = await _call_node(_state([faq]))

        assert result["faq_used"] == [
            {"id": "", "question": "¿Horarios?", "confidence": 0.3}
        ]
