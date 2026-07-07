"""Unit tests for the generic name_capture node.

The extraction LLM is replaced by a fake provider that returns a canned JSON
verdict; backend_client calls are AsyncMock-patched. Each test asserts the
node's turn contract: `name_capture_reply_sent`, latches, and state sync
driven strictly by the backend API result.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from langchain_core.messages import HumanMessage

from src.agents.name_capture import MAX_NAME_CAPTURE_ATTEMPTS, name_capture_node
from src.providers.base import UsageInfo


def _state(message: str = "Hola", **overrides) -> dict:
    state = {
        "messages": [HumanMessage(content=message)],
        "thread_id": "test:thread:1",
        "tenant_id": "tenant-1",
        "conversation_id": "conv-1",
        "contact_id": "contact-1",
        "user_context": {},
        "language": "es",
        "agent_code_name": "helena",
        "agent_type": "sales",
    }
    state.update(overrides)
    return state


class _FakeProvider:
    name = "fake"

    def __init__(self, verdict: dict):
        self.verdict = verdict
        self.last_usage = UsageInfo(input_tokens=10, output_tokens=5)

    def stream_chat(self, *, model: str, messages: list[dict]):
        async def _gen():
            yield json.dumps(self.verdict)

        return _gen()


def _writer_recorder():
    events: list[dict] = []
    return events, (lambda evt: events.append(evt))


async def _run(state: dict, verdict: dict, **client_mocks) -> tuple[dict, list[dict]]:
    """Run the node with a canned extraction verdict and mocked backend."""
    provider = _FakeProvider(verdict)
    events, write = _writer_recorder()
    update = client_mocks.get(
        "update",
        AsyncMock(return_value={"ok": True, "applied": True, "name": None}),
    )
    defer = client_mocks.get("defer", AsyncMock(return_value={"ok": True}))
    with (
        patch("src.agents.name_capture.get_provider", return_value=provider),
        patch("src.agents.name_capture.resolve_model", return_value="model-test"),
        patch("src.agents.name_capture.get_stream_writer", return_value=write),
        patch(
            "src.agents.name_capture.backend_client.update_contact_name", new=update
        ),
        patch(
            "src.agents.name_capture.backend_client.defer_contact_name_capture",
            new=defer,
        ),
    ):
        result = await name_capture_node(state, {"configurable": {}})
    return result, events


class TestFoundName:
    async def test_persists_latches_and_greets(self):
        update = AsyncMock(
            return_value={"ok": True, "applied": True, "name": "Patricia Hernández"}
        )
        result, events = await _run(
            _state("Soy Patricia Hernández"),
            {"found": True, "name": "Patricia Hernández"},
            update=update,
        )

        update.assert_awaited_once_with(
            contact_id="contact-1", name="Patricia Hernández", tenant_id="tenant-1"
        )
        assert result["name_captured"] is True
        assert result["school_name_captured"] is True  # legacy camila compat
        assert result["user_context"]["name"] == "Patricia Hernández"
        assert result["name_capture_reply_sent"] is True
        assert "Patricia Hernández" in events[0]["content"]

    async def test_adopts_backend_sanitized_name(self):
        # Backend strips HTML/whitespace — its canonical form wins.
        update = AsyncMock(return_value={"ok": True, "applied": True, "name": "Ana"})
        result, _ = await _run(
            _state("Soy  Ana "), {"found": True, "name": "Ana <b>x</b>"}, update=update
        )
        assert result["user_context"]["name"] == "Ana"

    async def test_manual_conflict_adopts_human_name(self):
        # applied=false → a CRM-edited (MANUAL) name exists; the agent greets
        # with the human's version, not the customer-offered one.
        update = AsyncMock(
            return_value={"ok": True, "applied": False, "name": "Doña Patricia"}
        )
        result, events = await _run(
            _state("Me llamo Paty"), {"found": True, "name": "Paty"}, update=update
        )
        assert result["user_context"]["name"] == "Doña Patricia"
        assert result["name_captured"] is True
        assert "Doña Patricia" in events[0]["content"]

    async def test_backend_error_greets_but_does_not_latch(self):
        update = AsyncMock(side_effect=httpx.RequestError("boom"))
        result, events = await _run(
            _state("Soy Ana"), {"found": True, "name": "Ana"}, update=update
        )
        # Personalizes this turn only — no latch, backend snapshot stays
        # authoritative next turn.
        assert "name_captured" not in result
        assert "school_name_captured" not in result
        assert result["user_context"]["name"] == "Ana"
        assert result["name_capture_reply_sent"] is True
        assert "Ana" in events[0]["content"]

    async def test_found_with_question_continues_routing(self):
        result, events = await _run(
            _state("Soy Ana, ¿tienen envíos?"),
            {"found": True, "name": "Ana", "has_question": True},
        )
        # Greeting prefix emitted, but the turn continues so the graph
        # answers the question.
        assert result["name_capture_reply_sent"] is False
        assert "Ana" in events[0]["content"]


class TestAskAndAttempts:
    async def test_no_name_first_attempt_asks(self):
        result, events = await _run(_state("Hola"), {"found": False, "name": ""})
        assert result["name_capture_attempts"] == 1
        assert result["name_capture_reply_sent"] is True
        assert "name_captured" not in result
        # Persona-aware ask
        assert "Helena" in events[0]["content"]

    async def test_second_miss_defers_implicitly_and_continues(self):
        defer = AsyncMock(return_value={"ok": True})
        result, events = await _run(
            _state("¿tienen envíos a Cali?", name_capture_attempts=1),
            {"found": False, "name": ""},
            defer=defer,
        )
        defer.assert_awaited_once()
        assert result["name_capture_deferred"] is True
        assert result["name_capture_reply_sent"] is False
        assert events == []  # nothing user-facing; downstream node answers

    async def test_max_attempts_constant(self):
        assert MAX_NAME_CAPTURE_ATTEMPTS == 2


class TestRefusal:
    async def test_refusal_defers_on_backend(self):
        defer = AsyncMock(return_value={"ok": True})
        result, events = await _run(
            _state("prefiero no decirlo"),
            {"found": False, "name": "", "refused": True},
            defer=defer,
        )
        defer.assert_awaited_once_with(contact_id="contact-1", tenant_id="tenant-1")
        assert result["name_capture_deferred"] is True
        # No question in the message → acknowledge and end the turn.
        assert result["name_capture_reply_sent"] is True
        assert "Sin problema" in events[0]["content"]

    async def test_refusal_with_question_continues_silently(self):
        result, events = await _run(
            _state("no te lo diré, ¿cuál es el horario?"),
            {"found": False, "name": "", "refused": True, "has_question": True},
        )
        assert result["name_capture_deferred"] is True
        assert result["name_capture_reply_sent"] is False
        assert events == []

    async def test_defer_backend_failure_still_defers_locally(self):
        defer = AsyncMock(side_effect=httpx.RequestError("down"))
        result, _ = await _run(
            _state("no gracias"),
            {"found": False, "name": "", "refused": True},
            defer=defer,
        )
        # A backend hiccup must never turn into repeated nagging.
        assert result["name_capture_deferred"] is True


class TestLanguageAndPersona:
    async def test_english_ask(self):
        _, events = await _run(
            _state("Hi", language="en"), {"found": False, "name": ""}
        )
        assert "could you tell me your name" in events[0]["content"]

    async def test_portuguese_greeting(self):
        result, events = await _run(
            _state("Sou a Maria", language="pt"), {"found": True, "name": "Maria"}
        )
        assert "Muito prazer, Maria!" in events[0]["content"]

    async def test_channel_persona_overrides_default(self):
        _, events = await _run(
            _state("Hola", agent_persona_name="Doña Rosa"),
            {"found": False, "name": ""},
        )
        assert "Doña Rosa" in events[0]["content"]
        assert "Helena" not in events[0]["content"]

    async def test_parse_failure_falls_back_to_ask(self):
        provider = _FakeProvider({})
        provider.stream_chat = lambda *, model, messages: _bad_json_gen()
        events, write = _writer_recorder()
        with (
            patch("src.agents.name_capture.get_provider", return_value=provider),
            patch("src.agents.name_capture.resolve_model", return_value="m"),
            patch("src.agents.name_capture.get_stream_writer", return_value=write),
            patch(
                "src.agents.name_capture.backend_client.update_contact_name",
                new=AsyncMock(),
            ),
            patch(
                "src.agents.name_capture.backend_client.defer_contact_name_capture",
                new=AsyncMock(),
            ),
        ):
            result = await name_capture_node(_state("Hola"), {"configurable": {}})
        assert result["name_capture_reply_sent"] is True
        assert result["name_capture_attempts"] == 1


async def _bad_json_gen():
    yield "not json at all"
