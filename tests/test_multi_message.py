"""Multi-message handling: trailing-burst helpers + per-thread coalescing.

WhatsApp users often split one thought across several rapid messages. Two
layers handle this:

1. `latest_user_messages` / `latest_user_text` (src/agents/utils.py) — every
   node reads the whole trailing run of human messages, not just the last one.
2. `_coalesced_stream` (src/main.py) — concurrent /chat/stream requests for
   the same thread are serialized; buffered fragments merge into one graph
   run and superseded requests finish with an empty `done` event.
"""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.utils import latest_user_messages, latest_user_text


class TestLatestUserMessages:
    def test_empty_state(self):
        assert latest_user_messages({"messages": []}) == []
        assert latest_user_messages({}) == []
        assert latest_user_text({"messages": []}) == ""

    def test_single_trailing_human(self):
        state = {
            "messages": [
                HumanMessage(content="Hola"),
                AIMessage(content="¡Hola! ¿En qué puedo ayudarte?"),
                HumanMessage(content="¿Tienen envíos?"),
            ]
        }
        assert latest_user_messages(state) == ["¿Tienen envíos?"]
        assert latest_user_text(state) == "¿Tienen envíos?"

    def test_burst_of_trailing_humans_joined_in_order(self):
        state = {
            "messages": [
                AIMessage(content="¿En qué puedo ayudarte?"),
                HumanMessage(content="Buenos días Patricia"),
                HumanMessage(content="¿Extendieron la fecha de matrículas?"),
                HumanMessage(content="Vi un post en Facebook"),
            ]
        }
        assert latest_user_messages(state) == [
            "Buenos días Patricia",
            "¿Extendieron la fecha de matrículas?",
            "Vi un post en Facebook",
        ]
        assert latest_user_text(state) == (
            "Buenos días Patricia\n"
            "¿Extendieron la fecha de matrículas?\n"
            "Vi un post en Facebook"
        )

    def test_stops_at_last_ai_message(self):
        state = {
            "messages": [
                HumanMessage(content="mensaje viejo"),
                AIMessage(content="respuesta vieja"),
                HumanMessage(content="mensaje nuevo"),
            ]
        }
        assert latest_user_messages(state) == ["mensaje nuevo"]

    def test_no_trailing_human_returns_empty(self):
        state = {
            "messages": [
                HumanMessage(content="Hola"),
                AIMessage(content="respuesta"),
            ]
        }
        assert latest_user_messages(state) == []
        assert latest_user_text(state) == ""

    def test_skips_empty_fragments(self):
        state = {
            "messages": [
                HumanMessage(content="Hola"),
                HumanMessage(content=""),
            ]
        }
        assert latest_user_messages(state) == ["Hola"]


class TestCoalescedStream:
    """Exercises src.main._coalesced_stream with a stubbed _stream_graph."""

    @pytest.fixture
    def main_module(self, monkeypatch):
        from src import main as main_mod

        monkeypatch.setattr(main_mod.settings, "message_settle_seconds", 0.05)
        monkeypatch.setattr(main_mod.settings, "message_settle_max_seconds", 0.5)
        # Ensure no state leaks between tests
        main_mod._thread_turns.clear()
        return main_mod

    def _make_run_recorder(self, main_module, run_duration=0.0):
        runs = []

        async def fake_stream_graph(inputs, config, graph=None, agent_code_name=""):
            runs.append([m.content for m in inputs["messages"]])
            if run_duration:
                await asyncio.sleep(run_duration)
            yield main_module._sse_event({"type": "done", "turn_usage": ["real"]})

        return runs, fake_stream_graph

    def _request(
        self,
        main_module,
        thread_id,
        message,
        turn_request_id,
        attachments=None,
        faqs=None,
    ):
        inputs = {
            "messages": [HumanMessage(content=message)],
            "attachments": attachments or [],
            "faqs": faqs or [],
        }
        config = {"configurable": {"thread_id": thread_id, "turn_request_id": turn_request_id}}
        return main_module._coalesced_stream(
            message,
            attachments or [],
            faqs or [],
            inputs,
            config,
            object(),
            "helena",
            thread_id,
        )

    async def _collect(self, gen):
        return [chunk async for chunk in gen]

    async def test_single_message_runs_graph(self, main_module, monkeypatch):
        runs, fake = self._make_run_recorder(main_module)
        monkeypatch.setattr(main_module, "_stream_graph", fake)

        events = await self._collect(
            self._request(main_module, "t1", "Hola", "req-1")
        )

        assert runs == [["Hola"]]
        assert any('"turn_usage": ["real"]' in e for e in events)
        assert main_module._thread_turns == {}

    async def test_rapid_burst_merges_into_one_run(self, main_module, monkeypatch):
        runs, fake = self._make_run_recorder(main_module)
        monkeypatch.setattr(main_module, "_stream_graph", fake)

        async def staggered(delay, message, req_id):
            await asyncio.sleep(delay)
            return await self._collect(
                self._request(main_module, "t2", message, req_id)
            )

        results = await asyncio.gather(
            staggered(0.0, "Buenos días", "req-1"),
            staggered(0.01, "¿Extendieron la fecha?", "req-2"),
            staggered(0.02, "Vi un post en Facebook", "req-3"),
        )

        # One graph run answering the whole burst as a single user turn
        assert runs == [["Buenos días\n¿Extendieron la fecha?\nVi un post en Facebook"]]

        # Exactly one stream carries the real run; the others are empty dones
        # that echo their own turn_request_id so NestJS releases reservations.
        real = [r for r in results if any('"real"' in e for e in r)]
        superseded = [r for r in results if any('"coalesced": true' in e for e in r)]
        assert len(real) == 1
        assert len(superseded) == 2
        superseded_ids = {
            json.loads(e.removeprefix("data: ").strip())["turn_request_id"]
            for r in superseded
            for e in r
        }
        assert superseded_ids <= {"req-1", "req-2", "req-3"}
        assert main_module._thread_turns == {}

    async def test_message_during_run_is_answered_by_next_run(
        self, main_module, monkeypatch
    ):
        runs, fake = self._make_run_recorder(main_module, run_duration=0.2)
        monkeypatch.setattr(main_module, "_stream_graph", fake)

        async def second_later():
            # Arrives after the first run started (0.05 settle + margin)
            await asyncio.sleep(0.15)
            return await self._collect(
                self._request(main_module, "t3", "segundo mensaje", "req-2")
            )

        first, second = await asyncio.gather(
            self._collect(self._request(main_module, "t3", "primer mensaje", "req-1")),
            second_later(),
        )

        assert runs == [["primer mensaje"], ["segundo mensaje"]]
        assert any('"real"' in e for e in first)
        assert any('"real"' in e for e in second)

    async def test_threads_do_not_block_each_other(self, main_module, monkeypatch):
        runs, fake = self._make_run_recorder(main_module)
        monkeypatch.setattr(main_module, "_stream_graph", fake)

        a, b = await asyncio.gather(
            self._collect(self._request(main_module, "thread-a", "hola a", "req-a")),
            self._collect(self._request(main_module, "thread-b", "hola b", "req-b")),
        )

        assert sorted(r[0] for r in runs) == ["hola a", "hola b"]
        assert any('"real"' in e for e in a)
        assert any('"real"' in e for e in b)

    async def test_attachments_merge_across_fragments(self, main_module, monkeypatch):
        captured = {}

        async def fake_stream_graph(inputs, config, graph=None, agent_code_name=""):
            captured["messages"] = [m.content for m in inputs["messages"]]
            captured["attachments"] = inputs["attachments"]
            yield main_module._sse_event({"type": "done", "turn_usage": ["real"]})

        monkeypatch.setattr(main_module, "_stream_graph", fake_stream_graph)

        async def staggered(delay, message, req_id, attachments):
            await asyncio.sleep(delay)
            return await self._collect(
                self._request(
                    main_module, "t4", message, req_id, attachments=attachments
                )
            )

        await asyncio.gather(
            staggered(0.0, "mira esta foto", "req-1", [{"type": "image"}]),
            staggered(0.01, "y este documento", "req-2", [{"type": "document"}]),
        )

        assert captured["messages"] == ["mira esta foto\ny este documento"]
        assert captured["attachments"] == [{"type": "image"}, {"type": "document"}]

    async def test_faqs_merge_and_dedupe_across_fragments(
        self, main_module, monkeypatch
    ):
        captured = {}

        async def fake_stream_graph(inputs, config, graph=None, agent_code_name=""):
            captured["faqs"] = inputs["faqs"]
            yield main_module._sse_event({"type": "done", "turn_usage": ["real"]})

        monkeypatch.setattr(main_module, "_stream_graph", fake_stream_graph)

        faq_horarios = {"question": "¿Cuál es el horario?", "answer": "L-V 9-18"}
        faq_envios = {"question": "¿Hacen envíos?", "answer": "Sí, 3-5 días"}
        faq_matriculas = {"question": "¿Fechas de matrículas?", "answer": "Hasta el 19"}

        async def staggered(delay, message, req_id, faqs):
            await asyncio.sleep(delay)
            return await self._collect(
                self._request(main_module, "t5", message, req_id, faqs=faqs)
            )

        await asyncio.gather(
            staggered(0.0, "hola, una pregunta", "req-1", [faq_horarios, faq_envios]),
            # Second fragment's request re-matched FAQs against the fuller
            # burst text and found one more, plus a duplicate.
            staggered(0.01, "¿extendieron matrículas?", "req-2", [faq_envios, faq_matriculas]),
        )

        assert captured["faqs"] == [faq_horarios, faq_envios, faq_matriculas]
