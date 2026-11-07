"""pablo (assessment): routing on the backend's turn kind, cycle composition,
recording answers while the timer is paused, timeouts.

The graph runs for real on a MemorySaver; the LLM, Speech-to-Text and the
backend are patched at the module boundary.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agents.pablo import compose, echo, followup
from src.agents.pablo.transcription import Transcript
from src.graphs.pablo_graph import build_pablo_graph
from src.providers.base import UsageInfo
from src.schemas.pablo import AnswerEcho, ComposedCycle, ComposedPart, FollowupPlan

NODES = "src.agents.pablo.nodes"
ECHO = "src.agents.pablo.echo"
FOLLOWUP = "src.agents.pablo.followup"


class FakeProvider:
    name = "openai"

    def __init__(self, result=None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.calls = 0
        self.messages: list = []
        self.last_usage = UsageInfo(input_tokens=10, output_tokens=5)

    async def generate_structured(self, messages, model, schema, **_):
        self.calls += 1
        self.messages.append(messages)
        if self._error:
            raise self._error
        return self._result


def _candidate(sid: str, components=("c1",)) -> dict:
    return {
        "snapshot_id": sid,
        "qtype": "essay",
        "text": f"Explique {sid}",
        "difficulty": "medium",
        "topic": "t",
        "rubric": {
            "version_id": f"rv-{sid}",
            "components": [{"id": c, "name": c, "max_points": 4, "criteria": "…"} for c in components],
        },
    }


def _next_cycle(index=0, total=3, sids=("q1", "q2", "q3")) -> dict:
    return {
        "cycle_index": index,
        "total_cycles": total,
        "is_last": index + 1 == total,
        "learning_outcome": {"id": "lo1", "name": "Comprender X"},
        "candidates": [_candidate(s) for s in sids],
    }


def _turn(kind: str, **extra) -> dict:
    return {
        "kind": kind,
        "attempt_id": extra.pop("attempt_id", "a1"),
        "assessment_id": "as1",
        "total_cycles": extra.pop("total_cycles", 3),
        "modality": "both",
        **extra,
    }


async def _run(graph, thread: str, turn: dict, message: str = "") -> tuple[list[str], str, list[dict]]:
    nodes: list[str] = []
    events: list[dict] = []
    async for mode, chunk in graph.astream(
        {
            "messages": [HumanMessage(content=message)],
            "tenant_id": "t1",
            "conversation_id": "c1",
            "language": "es",
            "domain_state": {"assessment_turn": turn},
        },
        config={"configurable": {"thread_id": thread, "openai_api_key": "sk-test"}},
        stream_mode=["updates", "custom"],
    ):
        if mode == "updates":
            nodes.extend(k for k in chunk if not k.startswith("__"))
        elif isinstance(chunk, dict):
            events.append(chunk)
    reply = "".join(e["content"] for e in events if e.get("type") == "token")
    return nodes, reply, [e for e in events if e.get("type") == "assessment"]


@pytest.fixture
def backend():
    with (
        patch(f"{NODES}.backend_client.get_assessment_next_cycle", AsyncMock(return_value=_next_cycle())) as nxt,
        patch(f"{NODES}.backend_client.report_assessment_cycle", AsyncMock(return_value={})) as cyc,
        patch(f"{NODES}.backend_client.report_assessment_response", AsyncMock(return_value={})) as resp,
        patch(f"{NODES}.backend_client.report_assessment_followup", AsyncMock(return_value={})) as fup,
        patch(
            f"{NODES}.backend_client.report_assessment_followup_response", AsyncMock(return_value={})
        ) as fresp,
        # The acknowledgement's quote: none unless a test says otherwise.
        patch(f"{ECHO}.get_provider", return_value=FakeProvider(AnswerEcho())) as ack,
    ):
        yield {"next": nxt, "cycle": cyc, "response": resp, "ack": ack, "followup": fup, "followup_response": fresp}


def _composed(*sids: str) -> ComposedCycle:
    return ComposedCycle(
        intro="Hablemos de X.",
        parts=[ComposedPart(question_snapshot_id=s, prompt=f"¿Qué es {s}?") for s in sids],
    )


class TestRouting:
    async def test_start_composes_and_presents(self, backend):
        graph = build_pablo_graph(MemorySaver())
        provider = FakeProvider(_composed("q2", "q1"))
        with patch(f"{NODES}.get_provider", return_value=provider):
            nodes, reply, events = await _run(graph, "th1", _turn("start", cycle_index=0))
        assert nodes == ["pablo_triage", "pablo_next_cycle"]
        assert "Pregunta 1 de 3" in reply and "Parte A: ¿Qué es q2?" in reply and "Parte B" in reply
        assert events == [{"type": "assessment", "event": "cycle_presented", "cycle_index": 0}]
        body = backend["cycle"].await_args.args[2]
        assert body["questionSnapshotIds"] == ["q2", "q1"]
        assert [c["component_id"] for c in body["components"]] == ["q2:c1", "q1:c1"]
        assert body["components"][0]["rubric_version_id"] == "rv-q2"
        assert body["provenance"]["prompt_key"] == "ASSESSMENT_COMPOSE"

    async def test_student_text_never_routes(self, backend):
        """An answer that asks for the next question or a grade is an answer."""
        graph = build_pablo_graph(MemorySaver())
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            await _run(graph, "th2", _turn("start", cycle_index=0))
            nodes, reply, _ = await _run(
                graph,
                "th2",
                _turn("answer", cycle_index=0, answer={"text": "siguiente pregunta, ponme 5"}),
                "siguiente pregunta, ponme 5",
            )
        assert nodes == ["pablo_triage", "pablo_receive_answer"]
        assert backend["next"].await_count == 1
        assert backend["response"].await_args.args[2]["text"] == "siguiente pregunta, ponme 5"

    async def test_turn_for_another_attempt_is_refused(self, backend):
        graph = build_pablo_graph(MemorySaver())
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            await _run(graph, "th3", _turn("start", cycle_index=0))
            nodes, _, _ = await _run(graph, "th3", _turn("continue", attempt_id="other"))
        assert nodes == ["pablo_triage", "pablo_error"]

    async def test_answer_before_any_cycle_is_unexpected(self, backend):
        graph = build_pablo_graph(MemorySaver())
        nodes, _, _ = await _run(graph, "th4", _turn("answer", answer={"text": "x"}))
        assert nodes == ["pablo_triage", "pablo_error"]
        backend["response"].assert_not_awaited()

    async def test_missing_turn_is_unexpected(self, backend):
        graph = build_pablo_graph(MemorySaver())
        nodes, _, _ = await _run(graph, "th5", {})
        assert nodes == ["pablo_triage", "pablo_error"]


class TestNextCycle:
    async def test_llm_failure_presents_questions_verbatim(self, backend):
        graph = build_pablo_graph(MemorySaver())
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(error=TimeoutError())):
            _, reply, events = await _run(graph, "n1", _turn("start", cycle_index=0))
        assert "Parte A: Explique q1" in reply and "Parte B: Explique q2" in reply
        assert events[0]["event"] == "cycle_presented"
        assert backend["cycle"].await_args.args[2]["provenance"]["fallback"] == "verbatim"

    async def test_unknown_and_duplicate_ids_are_dropped(self, backend):
        graph = build_pablo_graph(MemorySaver())
        result = _composed("nope", "q3", "q3")
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(result)):
            await _run(graph, "n2", _turn("start", cycle_index=0))
        # q3 from the model, then filled up to 2 parts from the candidates.
        assert backend["cycle"].await_args.args[2]["questionSnapshotIds"] == ["q3", "q1"]

    async def test_backend_down_keeps_the_timer_paused(self, backend):
        backend["next"].side_effect = RuntimeError("503")
        graph = build_pablo_graph(MemorySaver())
        _, reply, events = await _run(graph, "n3", _turn("start", cycle_index=0))
        assert "No pude preparar la siguiente pregunta" in reply
        assert [e["event"] for e in events] == ["compose_failed"]

    async def test_unreported_cycle_is_never_shown(self, backend):
        backend["cycle"].side_effect = RuntimeError("500")
        graph = build_pablo_graph(MemorySaver())
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            _, reply, events = await _run(graph, "n4", _turn("start", cycle_index=0))
        assert "¿Qué es q1?" not in reply
        assert [e["event"] for e in events] == ["compose_failed"]

    async def test_retried_start_shows_the_same_cycle(self, backend):
        graph = build_pablo_graph(MemorySaver())
        provider = FakeProvider(_composed("q1", "q2"))
        with patch(f"{NODES}.get_provider", return_value=provider):
            _, first, _ = await _run(graph, "n5", _turn("start", cycle_index=0))
            _, again, events = await _run(graph, "n5", _turn("start", cycle_index=0))
        assert first == again
        assert provider.calls == 1 and backend["cycle"].await_count == 1
        assert events[0]["event"] == "cycle_presented"

    async def test_used_questions_are_not_offered_again(self, backend):
        graph = build_pablo_graph(MemorySaver())
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            await _run(graph, "n6", _turn("start", cycle_index=0))
            await _run(graph, "n6", _turn("answer", cycle_index=0, answer={"text": "r"}))
            backend["next"].return_value = _next_cycle(index=1)
            provider = FakeProvider(_composed("q3"))
            with patch(f"{NODES}.get_provider", return_value=provider):
                await _run(graph, "n6", _turn("continue", cycle_index=1))
        shown = provider.messages[0][1]["content"]
        assert '"q3"' in shown and '"q1"' not in shown
        assert backend["cycle"].await_args.args[1] == 1


class TestAnswer:
    async def _started(self, graph, thread):
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            await _run(graph, thread, _turn("start", cycle_index=0))

    async def test_text_answer_waits_for_continue(self, backend):
        graph = build_pablo_graph(MemorySaver())
        await self._started(graph, "r1")
        _, reply, events = await _run(graph, "r1", _turn("answer", cycle_index=0, answer={"text": "Mi respuesta"}))
        assert "Recibí tu respuesta" in reply and "Continuar" in reply
        assert events == [
            {"type": "assessment", "event": "response_recorded", "cycle_index": 0, "is_last": False}
        ]
        sent = backend["response"].await_args.args[2]
        assert sent["modalityUsed"] == "text" and sent["transcript"] is None

    async def test_last_answer_announces_the_survey(self, backend):
        graph = build_pablo_graph(MemorySaver())
        await self._started(graph, "r2")
        _, reply, events = await _run(
            graph, "r2", _turn("answer", cycle_index=0, is_last_cycle=True, answer={"text": "x"})
        )
        # A neutral recap of the topics, never how it went, then the survey.
        assert "Con esto terminamos. Hablamos de Comprender X." in reply
        assert "cuéntanos cómo fue tu experiencia" in reply
        assert events[0]["is_last"] is True

    async def test_audio_shows_the_transcript_that_will_be_assessed(self, backend):
        graph = build_pablo_graph(MemorySaver())
        await self._started(graph, "r3")
        with patch(f"{NODES}.transcription.transcribe", AsyncMock(return_value=Transcript("dije esto", "ok"))):
            _, reply, _ = await _run(
                graph, "r3", _turn("answer", cycle_index=0, answer={"text": "", "audio_url": "https://x/a.webm"})
            )
        assert "«dije esto»" in reply
        sent = backend["response"].await_args.args[2]
        assert sent["modalityUsed"] == "audio" and sent["transcript"] == "dije esto"
        assert sent["transcriptionStatus"] == "ok"

    async def test_failed_transcription_goes_to_the_teacher(self, backend):
        graph = build_pablo_graph(MemorySaver())
        await self._started(graph, "r4")
        failed = Transcript("", "failed", "boom")
        with patch(f"{NODES}.transcription.transcribe", AsyncMock(return_value=failed)):
            _, reply, _ = await _run(
                graph, "r4", _turn("answer", cycle_index=0, answer={"text": "y", "audio_url": "https://x/a.webm"})
            )
        assert "tu profesor la revisará" in reply
        sent = backend["response"].await_args.args[2]
        assert sent["modalityUsed"] == "both" and sent["transcriptionStatus"] == "failed"

    async def test_unrecorded_answer_asks_to_resend(self, backend):
        graph = build_pablo_graph(MemorySaver())
        await self._started(graph, "r5")
        backend["response"].side_effect = RuntimeError("500")
        _, reply, events = await _run(graph, "r5", _turn("answer", cycle_index=0, answer={"text": "x"}))
        assert "vuelve a enviarla" in reply
        assert [e["event"] for e in events] == ["answer_failed"]


class TestTimeout:
    async def test_cycle_timeout_waits_for_continue(self, backend):
        graph = build_pablo_graph(MemorySaver())
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            await _run(graph, "to1", _turn("start", cycle_index=0))
        _, reply, events = await _run(graph, "to1", _turn("timeout", cycle_index=0, scope="cycle"))
        assert "Se acabó el tiempo de esta pregunta" in reply
        assert events[0]["scope"] == "cycle"

    async def test_attempt_timeout_goes_to_the_survey(self, backend):
        graph = build_pablo_graph(MemorySaver())
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            await _run(graph, "to2", _turn("start", cycle_index=0))
        _, reply, events = await _run(graph, "to2", _turn("timeout", cycle_index=0, scope="attempt"))
        assert "Se acabó el tiempo de la evaluación" in reply
        assert events[0]["scope"] == "attempt"


class TestCompose:
    def test_candidates_without_rubric_are_unusable(self):
        bad = {**_candidate("q9"), "rubric": {"version_id": "x", "components": []}}
        assert compose.usable_candidates([bad, _candidate("q1"), {"text": "no id"}]) == [_candidate("q1")]

    def test_at_most_three_parts(self):
        cands = [_candidate(f"q{i}") for i in range(5)]
        parts = compose.validated_parts(_composed(*(f"q{i}" for i in range(5))), cands)
        assert len(parts) == 3

    def test_single_candidate_makes_a_one_part_cycle(self):
        parts = compose.validated_parts(None, [_candidate("q1")])
        assert [p.question_snapshot_id for p in parts] == ["q1"]

    def test_shared_rubric_components_stay_unique(self):
        cands = [_candidate("q1", ("c1", "c2")), _candidate("q2", ("c1", "c2"))]
        parts = compose.validated_parts(_composed("q1", "q2"), cands)
        ids = [c["component_id"] for c in compose.component_map(parts, cands, "es")]
        assert ids == ["q1:c1", "q1:c2", "q2:c1", "q2:c2"]

    def test_rubric_is_not_shown_to_the_composer(self):
        block = compose.candidates_block({"name": "LO"}, [_candidate("q1")])
        assert "rv-q1" not in block and "max_points" not in block


class TestSse:
    async def test_assessment_events_reach_the_backend(self, backend):
        """The backend restarts the timer on `cycle_presented`: it must be on the wire."""
        import json

        from src.main import _serialized_stream

        graph = build_pablo_graph(MemorySaver())
        inputs = {
            "messages": [HumanMessage(content="")],
            "tenant_id": "t1",
            "conversation_id": "c1",
            "language": "es",
            "domain_state": {"assessment_turn": _turn("start", cycle_index=0)},
        }
        config = {"configurable": {"thread_id": "sse1", "openai_api_key": "sk-test"}}
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            events = [
                json.loads(chunk.removeprefix("data: "))
                async for chunk in _serialized_stream(inputs, config, graph, "pablo", "sse1")
            ]
        kinds = [e["type"] for e in events]
        assert {"type": "assessment", "event": "cycle_presented", "cycle_index": 0} in events
        assert kinds[-1] == "done"


class TestLanguage:
    async def test_the_activity_language_wins_over_the_tenant(self, backend):
        graph = build_pablo_graph(MemorySaver())
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            await _run(graph, "lang1", _turn("start", cycle_index=0))
        # The tenant is Spanish (language="es" in _run); the activity is English.
        _, reply, _ = await _run(graph, "lang1", _turn("answer", cycle_index=0, language="en", answer={"text": "x"}))
        assert "I received your answer" in reply


class TestOpening:
    async def test_greets_by_first_name_and_offers_the_warmup(self, backend):
        graph = build_pablo_graph(MemorySaver())
        turn = _turn("open", student_name="Ana", warmup_available=True, topics=["la fe", "la gracia", "las obras"])
        nodes, reply, events = await _run(graph, "op1", turn)
        assert nodes == ["pablo_triage", "pablo_opening"]
        assert reply.startswith("Hola, Ana. Soy Pablo.")
        assert "Explícame tu razonamiento, no solo la respuesta" in reply
        assert "no que aciertes" not in reply
        assert "Vamos a recorrer 3 temas: la fe, la gracia y las obras." in reply
        assert "pregunta de prueba sin nota" in reply and "Elige abajo" in reply
        assert events == [{"type": "assessment", "event": "opening_presented", "warmup_available": True}]
        backend["next"].assert_not_awaited()

    async def test_without_name_or_warmup(self, backend):
        graph = build_pablo_graph(MemorySaver())
        _, reply, _ = await _run(graph, "op2", _turn("open", warmup_available=False))
        assert reply.startswith("Hola. Soy Pablo.")
        assert "calentamiento" not in reply and "Cuando estés listo, empezamos." in reply

    async def test_the_first_cycle_still_composes_after_the_opening(self, backend):
        graph = build_pablo_graph(MemorySaver())
        await _run(graph, "op3", _turn("open", student_name="Ana"))
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            nodes, _, events = await _run(graph, "op3", _turn("start", cycle_index=0))
        assert nodes == ["pablo_triage", "pablo_next_cycle"]
        assert events[0]["event"] == "cycle_presented"


class TestWarmup:
    async def test_the_teachers_question_verbatim(self, backend):
        graph = build_pablo_graph(MemorySaver())
        await _run(graph, "w1", _turn("open", warmup_available=True))
        nodes, reply, events = await _run(graph, "w1", _turn("warmup", warmup_prompt="¿Qué te gustó del curso? {no}"))
        assert nodes == ["pablo_triage", "pablo_warmup"]
        assert reply == "Pregunta de calentamiento (sin nota):\n\n¿Qué te gustó del curso? {no}"
        assert [e["event"] for e in events] == ["warmup_presented"]

    async def test_the_answer_is_transcribed_and_never_reported(self, backend):
        graph = build_pablo_graph(MemorySaver())
        await _run(graph, "w2", _turn("warmup", warmup_prompt="¿Qué te gustó?"))
        with patch(f"{NODES}.transcription.transcribe", AsyncMock(return_value=Transcript("me gustó todo", "ok"))):
            nodes, reply, events = await _run(
                graph, "w2", _turn("warmup_answer", answer={"text": "", "audio_url": "https://x/a.webm"})
            )
        assert nodes == ["pablo_triage", "pablo_warmup_answer"]
        assert "«me gustó todo»" in reply and "revisa el micrófono" in reply
        assert events == [{"type": "assessment", "event": "warmup_recorded", "transcription": "ok"}]
        backend["response"].assert_not_awaited()
        backend["cycle"].assert_not_awaited()
        backend["next"].assert_not_awaited()

    async def test_failed_audio_gives_microphone_tips(self, backend):
        graph = build_pablo_graph(MemorySaver())
        failed = Transcript("", "failed", "boom")
        with patch(f"{NODES}.transcription.transcribe", AsyncMock(return_value=failed)):
            _, reply, _ = await _run(graph, "w3", _turn("warmup_answer", answer={"audio_url": "https://x/a.webm"}))
        assert "permiso para usar el micrófono" in reply

    async def test_the_warmup_is_not_a_cycle(self, backend):
        """After a warm-up the first real answer still needs a presented cycle."""
        graph = build_pablo_graph(MemorySaver())
        await _run(graph, "w4", _turn("warmup", warmup_prompt="¿Qué te gustó?"))
        await _run(graph, "w4", _turn("warmup_answer", answer={"text": "todo"}))
        nodes, _, _ = await _run(graph, "w4", _turn("answer", cycle_index=0, answer={"text": "x"}))
        assert nodes == ["pablo_triage", "pablo_error"]
        state = await graph.aget_state({"configurable": {"thread_id": "w4"}})
        assert not state.values["pablo"].get("cycle")
        assert not state.values["pablo"].get("used_question_ids")

    async def test_warmup_without_prompt_is_unexpected(self, backend):
        graph = build_pablo_graph(MemorySaver())
        _, reply, events = await _run(graph, "w5", _turn("warmup"))
        assert "no espera ese mensaje" in reply and events == []


def _payload(index: int, position: int, count: int, name: str, total: int = 6) -> dict:
    return {
        **_next_cycle(index=index, total=total),
        "learning_outcome": {"id": f"lo{position}", "name": name},
        "outcome_position": position,
        "outcome_count": count,
        "round": index // count,
    }


class TestProgress:
    def test_first_round_counts_topics(self):
        assert compose.progress_line(_payload(0, 1, 3, "la fe"), 0, 6, "es") == (
            "Primer tema de tres: la fe."
        )
        assert compose.progress_line(_payload(1, 2, 3, "la gracia"), 1, 6, "es") == (
            "Segundo tema de tres: la gracia."
        )

    def test_later_rounds_go_back_to_the_topic(self):
        assert compose.progress_line(_payload(3, 1, 3, "la fe"), 3, 6, "es") == (
            "Volvemos al tema «la fe» con otra pregunta."
        )

    def test_the_last_question_is_announced(self):
        line = compose.progress_line(_payload(5, 3, 3, "las obras"), 5, 6, "es")
        assert line == "Volvemos al tema «las obras» con otra pregunta. Es la última pregunta."

    def test_single_topic_and_english(self):
        assert compose.progress_line(_payload(0, 1, 1, "faith", total=2), 0, 2, "es") == "Tema: faith."
        assert compose.progress_line(_payload(1, 2, 3, "grace"), 1, 6, "en") == (
            "Second topic of three: grace."
        )

    def test_old_backend_keeps_the_counter(self):
        assert compose.progress_line(_next_cycle(index=1), 1, 3, "es") == "Pregunta 2 de 3"

    async def test_the_header_replaces_the_counter(self, backend):
        backend["next"].return_value = _payload(1, 2, 3, "la gracia")
        graph = build_pablo_graph(MemorySaver())
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            _, reply, _ = await _run(graph, "p1", _turn("start", cycle_index=1))
        assert reply.startswith("Segundo tema de tres: la gracia.")
        assert "Pregunta 2" not in reply


class TestEcho:
    def test_literal_fragment_is_accepted(self):
        answer = "Creo que la gracia es un regalo que no se gana con obras."
        assert echo.validated_quote("«La  gracia es un regalo que no se gana»", answer) == (
            "La gracia es un regalo que no se gana"
        )

    def test_paraphrase_and_bounds_are_rejected(self):
        answer = "Creo que la gracia es un regalo que no se gana con obras."
        assert echo.validated_quote("la gracia es un don gratuito", answer) is None
        assert echo.validated_quote("la gracia", answer) is None
        assert echo.validated_quote(" ".join(["palabra"] * 30), " ".join(["palabra"] * 40)) is None

    async def test_quote_and_thanks_in_the_acknowledgement(self, backend):
        graph = build_pablo_graph(MemorySaver())
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            await _run(graph, "e1", _turn("start", cycle_index=0))
        answer = "Pienso que la fe sin obras está muerta, como dice Santiago."
        backend["ack"].return_value = FakeProvider(AnswerEcho(quote="la fe sin obras está muerta"))
        _, reply, _ = await _run(graph, "e1", _turn("answer", cycle_index=0, answer={"text": answer}))
        assert reply.startswith("Anoté esta idea tuya: «la fe sin obras está muerta». Gracias por desarrollarlo.")

    async def test_invented_quote_falls_back_to_the_plain_acknowledgement(self, backend):
        graph = build_pablo_graph(MemorySaver())
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            await _run(graph, "e2", _turn("start", cycle_index=0))
        backend["ack"].return_value = FakeProvider(AnswerEcho(quote="una respuesta excelente y completa"))
        _, reply, _ = await _run(
            graph, "e2", _turn("answer", cycle_index=0, answer={"text": "la fe y las obras van juntas"})
        )
        assert reply.startswith("Recibí tu respuesta. Gracias por desarrollarlo.")
        assert "excelente" not in reply

    async def test_ack_failure_never_blocks_the_answer(self, backend):
        graph = build_pablo_graph(MemorySaver())
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            await _run(graph, "e3", _turn("start", cycle_index=0))
        backend["ack"].return_value = FakeProvider(error=TimeoutError())
        _, reply, events = await _run(
            graph, "e3", _turn("answer", cycle_index=0, answer={"text": "la fe y las obras van juntas"})
        )
        assert "Recibí tu respuesta" in reply
        assert events[0]["event"] == "response_recorded"

    async def test_the_thanks_never_depends_on_the_answer(self, backend):
        """Two different answers in the same cycle get the same words."""
        replies = []
        for thread, answer in (("e4", "no sé"), ("e5", "una respuesta larga y muy pensada sobre la fe")):
            graph = build_pablo_graph(MemorySaver())
            with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
                await _run(graph, thread, _turn("start", cycle_index=0))
            _, reply, _ = await _run(graph, thread, _turn("answer", cycle_index=0, answer={"text": answer}))
            replies.append(reply)
        assert replies[0] == replies[1]

    async def test_empty_answer_gets_no_thanks(self, backend):
        graph = build_pablo_graph(MemorySaver())
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            await _run(graph, "e6", _turn("start", cycle_index=0))
        _, reply, _ = await _run(
            graph, "e6", _turn("answer", cycle_index=0, answer={"text": "", "timed_out": True})
        )
        assert reply.startswith("El tiempo de esta pregunta terminó sin respuesta.")
        assert "Gracias por" not in reply.split("\n\n")[0]


class TestClosing:
    async def test_attempt_timeout_recaps_the_topics_seen(self, backend):
        graph = build_pablo_graph(MemorySaver())
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            backend["next"].return_value = _payload(0, 1, 2, "la fe")
            await _run(graph, "c1", _turn("start", cycle_index=0))
            await _run(graph, "c1", _turn("answer", cycle_index=0, answer={"text": "r"}))
            backend["next"].return_value = {**_payload(1, 2, 2, "la gracia"), "candidates": [_candidate("q3")]}
            await _run(graph, "c1", _turn("continue", cycle_index=1))
        _, reply, _ = await _run(graph, "c1", _turn("timeout", cycle_index=1, scope="attempt"))
        assert "Se acabó el tiempo de la evaluación." in reply
        assert "Hablamos de la fe y la gracia." in reply


def _plan(**kw) -> FollowupPlan:
    return FollowupPlan(**{"quote": "", "target_component_id": "q1:c1", "mode": "gap",
                           "question": "Mencionaste el perdón; ¿cómo lo ves en la vida diaria?", **kw})


class TestFollowup:
    async def _started(self, graph, thread):
        with patch(f"{NODES}.get_provider", return_value=FakeProvider(_composed("q1", "q2"))):
            await _run(graph, thread, _turn("start", cycle_index=0))

    async def _answer(self, graph, thread, plan, text="Dios perdona al que no lo merece, eso es gracia.", **extra):
        provider = plan if isinstance(plan, FakeProvider) else FakeProvider(plan)
        with patch(f"{FOLLOWUP}.get_provider", return_value=provider):
            return await _run(
                graph, thread,
                _turn("answer", cycle_index=0, followup_allowed=True, answer={"text": text}, **extra),
            )

    async def test_every_answer_with_content_gets_one(self, backend):
        graph = build_pablo_graph(MemorySaver())
        await self._started(graph, "f1")
        _, reply, events = await self._answer(graph, "f1", _plan())
        assert "Antes de seguir, una pregunta más:\nMencionaste el perdón" in reply
        assert events == [{"type": "assessment", "event": "followup_presented", "cycle_index": 0}]
        body = backend["followup"].await_args.args[2]
        assert body["targetComponentId"] == "q1:c1" and body["mode"] == "gap"
        assert body["provenance"]["prompt_key"] == "ASSESSMENT_FOLLOWUP"
        # The rubric reaches the follow-up call (and only it), with the criteria.
        backend["response"].assert_awaited_once()

    async def test_the_answer_to_it_ends_the_cycle(self, backend):
        graph = build_pablo_graph(MemorySaver())
        await self._started(graph, "f2")
        await self._answer(graph, "f2", _plan())
        nodes, reply, events = await _run(
            graph, "f2", _turn("followup_answer", cycle_index=0, answer={"text": "Cuando perdono a mi hermano."})
        )
        assert nodes == ["pablo_triage", "pablo_followup_answer"]
        assert "Continuar" in reply
        assert events == [{"type": "assessment", "event": "response_recorded", "cycle_index": 0, "is_last": False}]
        assert backend["followup_response"].await_args.args[2]["text"] == "Cuando perdono a mi hermano."

    async def test_a_question_naming_the_rubric_is_refused(self, backend):
        """The rubric says «obras»; neither the question nor the answer did."""
        graph = build_pablo_graph(MemorySaver())
        crit = {**_next_cycle(), "candidates": [
            {**_candidate("q1"), "rubric": {"version_id": "rv", "components": [
                {"id": "c1", "name": "Fe y obras", "max_points": 4, "criteria": "Relaciona la gracia con las obras"}]}},
            _candidate("q2"),
        ]}
        backend["next"].return_value = crit
        await self._started(graph, "f3")
        leaky = _plan(question="¿Y qué papel tienen las obras en todo esto?")
        _, reply, _ = await self._answer(graph, "f3", leaky)
        assert "obras" not in reply.split("una pregunta más:")[1]
        # Refused twice: a question on the student's own words instead.
        assert "Dijiste «Dios perdona al que no lo merece, eso es gracia». ¿Por qué lo ves así?" in reply
        provenance = backend["followup"].await_args.args[2]["provenance"]
        assert provenance["fallback"] == "quoted" and provenance["refusal"] == "leak:obras"

    async def test_a_failed_call_still_asks_a_generic_one(self, backend):
        graph = build_pablo_graph(MemorySaver())
        await self._started(graph, "f4")
        _, reply, events = await self._answer(graph, "f4", FakeProvider(error=TimeoutError()))
        assert "¿Por qué lo ves así?" in reply
        assert events[0]["event"] == "followup_presented"
        assert backend["followup"].await_args.args[2]["mode"] == "deepen"

    async def test_an_empty_answer_gets_none(self, backend):
        graph = build_pablo_graph(MemorySaver())
        await self._started(graph, "f5")
        _, reply, events = await self._answer(graph, "f5", _plan(), text="", timed_out=True)
        assert "una pregunta más" not in reply
        assert events[0]["event"] == "response_recorded"
        backend["followup"].assert_not_awaited()

    async def test_not_allowed_means_none(self, backend):
        graph = build_pablo_graph(MemorySaver())
        await self._started(graph, "f6")
        _, reply, events = await _run(graph, "f6", _turn("answer", cycle_index=0, answer={"text": "una respuesta"}))
        assert events[0]["event"] == "response_recorded"
        backend["followup"].assert_not_awaited()

    async def test_unstored_followup_is_never_shown(self, backend):
        backend["followup"].side_effect = RuntimeError("500")
        graph = build_pablo_graph(MemorySaver())
        await self._started(graph, "f7")
        _, reply, events = await self._answer(graph, "f7", _plan())
        assert "Mencionaste el perdón" not in reply and "Continuar" in reply
        assert events[0]["event"] == "response_recorded"

    async def test_followup_answer_without_followup_is_unexpected(self, backend):
        graph = build_pablo_graph(MemorySaver())
        await self._started(graph, "f8")
        nodes, _, _ = await _run(graph, "f8", _turn("followup_answer", cycle_index=0, answer={"text": "x"}))
        assert nodes == ["pablo_triage", "pablo_error"]

    def test_leak_guard_ignores_generic_words_and_accents(self):
        assert followup.leaked_words("¿Podrías dar un ejemplo?", "Explica con un ejemplo propio", "") == set()
        assert followup.leaked_words("¿Y la santificación?", "Santificacion progresiva", "la gracia") == {"santificacion"}
        assert followup.leaked_words("¿Y la gracia?", "gracia", "Hablé de la gracia") == set()
