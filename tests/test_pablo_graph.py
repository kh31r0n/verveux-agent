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

from src.agents.pablo import compose
from src.agents.pablo.transcription import Transcript
from src.graphs.pablo_graph import build_pablo_graph
from src.providers.base import UsageInfo
from src.schemas.pablo import ComposedCycle, ComposedPart

NODES = "src.agents.pablo.nodes"


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
    ):
        yield {"next": nxt, "cycle": cyc, "response": resp}


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
        assert "última pregunta" in reply
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
