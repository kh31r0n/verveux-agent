"""ismael (theology over Company Brain): routing, the once-per-contact survey,
one Brain job per conversation, and how the job turns Brain's outcome into the
message the student receives.

The graph runs for real on a MemorySaver; the LLM, Brain and the backend are
patched at the module boundary.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agents.ismael import rag_job
from src.agents.ismael.nodes import parse_survey_answer
from src.agents.ismael.texts import SURVEY, text
from src.graphs.ismael_graph import build_ismael_graph
from src.main import _stream_graph
from src.schemas.ismael import IsmaelIntent, SurveyAnswer, TriageResult

NODES = "src.agents.ismael.nodes"


class FakeProvider:
    name = "openai"

    def __init__(self, result):
        self._result = result
        self.calls = 0
        from src.providers.base import UsageInfo

        self.last_usage = UsageInfo(input_tokens=10, output_tokens=5)

    async def generate_structured(self, messages, model, schema, **_):
        self.calls += 1
        return self._result


def _inputs(message: str, **extra) -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "tenant_id": "t1",
        "conversation_id": "c1",
        "contact_id": "k1",
        "agent_type": "theology",
        "agent_code_name": "ismael",
        "language": "es",
        "user_context": extra.pop("user_context", {"name": "Ana Pérez"}),
        **extra,
    }


def _config(thread: str) -> dict:
    return {"configurable": {"thread_id": thread, "openai_api_key": "sk-test"}}


async def _events(graph, thread: str, message: str, **extra) -> tuple[list[str], list[dict]]:
    """One turn: the nodes it went through and every custom stream event."""
    nodes: list[str] = []
    events: list[dict] = []
    async for mode, chunk in graph.astream(
        _inputs(message, **extra), config=_config(thread), stream_mode=["updates", "custom"]
    ):
        if mode == "updates":
            nodes.extend(k for k in chunk if not k.startswith("__"))
        elif isinstance(chunk, dict):
            events.append(chunk)
    return nodes, events


async def _run(graph, thread: str, message: str, **extra) -> tuple[list[str], str, list[dict]]:
    """One turn: the nodes it went through, the reply text and quick-reply events."""
    nodes, events = await _events(graph, thread, message, **extra)
    reply = "".join(e["content"] for e in events if e.get("type") == "token")
    return nodes, reply, [e for e in events if e.get("type") == "quick_replies"]


async def _turn(graph, thread: str, message: str, **extra) -> tuple[list[str], str]:
    nodes, reply, _ = await _run(graph, thread, message, **extra)
    return nodes, reply


@pytest.fixture(autouse=True)
def _clean_local_jobs():
    rag_job._local_jobs.clear()
    yield
    rag_job._local_jobs.clear()


@pytest.fixture
def boundary():
    """Patch everything that leaves the process; yields the mocks."""
    triage = FakeProvider(TriageResult(intent=IsmaelIntent.THEOLOGY, question="¿Qué es la gracia?"))
    with (
        patch(f"{NODES}.get_provider", return_value=triage),
        patch(f"{NODES}.resolve_model", return_value="gpt-test"),
        patch(f"{NODES}.rag_job.spawn", new=AsyncMock()) as spawn,
        patch(f"{NODES}.rag_job.start_host_in_background") as start_host,
        patch(f"{NODES}.backend_client.save_ismael_survey", new=AsyncMock()) as save,
        patch("src.agents.ismael.rag_job.get_store_or_none", return_value=None),
    ):
        yield {"triage": triage, "spawn": spawn, "start_host": start_host, "save": save}


# ── Routing and the survey ───────────────────────────────────────────────────


class TestFirstQuestion:
    async def test_new_contact_gets_the_survey_while_brain_boots(self, boundary):
        graph = build_ismael_graph(MemorySaver())
        nodes, reply = await _turn(graph, "th1", "que es la gracia")

        assert nodes == ["ismael_triage", "ismael_start"]
        boundary["start_host"].assert_called_once()
        boundary["spawn"].assert_not_called()  # the answer waits for the survey
        assert "Ana" in reply and "1/3" in reply and "Estudiante" in reply

    async def test_full_survey_saves_stats_and_spawns_the_pending_question(self, boundary):
        graph = build_ismael_graph(MemorySaver())
        await _turn(graph, "th2", "que es la gracia")

        nodes, reply = await _turn(graph, "th2", "1")
        assert nodes == ["ismael_triage", "ismael_survey"] and "2/3" in reply
        _, reply = await _turn(graph, "th2", "Historia de la Iglesia")
        assert "3/3" in reply
        _, reply = await _turn(graph, "th2", "la 2")

        assert reply == text("survey_done", "es")
        boundary["save"].assert_awaited_once_with(
            "k1", "c1", {"level": "student", "topic": "church_history", "intendedUse": "coursework"}
        )
        job = boundary["spawn"].await_args.args[0]
        assert job.question == "¿Qué es la gracia?"
        assert (job.tenant_id, job.conversation_id, job.language) == ("t1", "c1", "es")
        # Survey turns are deterministic: only the first turn called the LLM.
        assert boundary["triage"].calls == 1

    async def test_contact_who_answered_before_is_not_asked_again(self, boundary):
        graph = build_ismael_graph(MemorySaver())
        nodes, reply = await _turn(
            graph, "th3", "que es la gracia", user_context={"name": "Ana", "ismael_survey_done": True}
        )

        assert nodes == ["ismael_triage", "ismael_start"]
        assert reply == text("consulting", "es")
        boundary["spawn"].assert_awaited_once()
        boundary["start_host"].assert_not_called()

    async def test_unreadable_answer_is_recorded_as_no_answer_and_never_re_asked(self, boundary):
        boundary["triage"]._result = TriageResult(intent=IsmaelIntent.THEOLOGY, question="q")
        graph = build_ismael_graph(MemorySaver())
        await _turn(graph, "th4", "q")
        with patch(
            f"{NODES}._classify_survey_answer",
            new=AsyncMock(return_value=(SurveyAnswer(answer="no_answer"), [])),
        ):
            _, reply = await _turn(graph, "th4", "prefiero no decirlo")
        assert "2/3" in reply

    async def test_a_new_question_during_the_survey_replaces_the_pending_one(self, boundary):
        graph = build_ismael_graph(MemorySaver())
        await _turn(graph, "th5", "que es la gracia")
        with patch(
            f"{NODES}._classify_survey_answer",
            new=AsyncMock(return_value=(SurveyAnswer(is_new_question=True), [])),
        ):
            await _turn(graph, "th5", "¿mejor explícame la Trinidad?")
        await _turn(graph, "th5", "1")
        await _turn(graph, "th5", "1")
        assert boundary["spawn"].await_args.args[0].question == "¿mejor explícame la Trinidad?"


class TestOtherBranches:
    async def test_running_job_holds_theology_questions_in_pending(self, boundary):
        rag_job._local_jobs["c1"] = ("job-1", "¿Qué es la gracia?", time.monotonic())
        graph = build_ismael_graph(MemorySaver())
        nodes, reply = await _turn(graph, "th6", "¿ya tienes la respuesta?")

        assert nodes == ["ismael_triage", "ismael_pending"]
        assert "¿Qué es la gracia?" in reply
        boundary["spawn"].assert_not_called()

    async def test_off_topic(self, boundary):
        boundary["triage"]._result = TriageResult(intent=IsmaelIntent.OTHER)
        graph = build_ismael_graph(MemorySaver())
        nodes, reply = await _turn(graph, "th7", "¿cuándo es el examen?")
        assert nodes == ["ismael_triage", "ismael_off_topic"]
        assert "Ismael" in reply and "teología" in reply

    async def test_greeting_uses_the_shared_greeting_node(self, boundary):
        boundary["triage"]._result = TriageResult(intent=IsmaelIntent.GREETING)
        graph = build_ismael_graph(MemorySaver())
        with patch("src.agents.greeting_response.get_provider", side_effect=RuntimeError("no llm")):
            nodes, reply = await _turn(graph, "th8", "hola")
        assert nodes == ["ismael_triage", "greeting_response"]
        assert "Ismael" in reply and "teología" in reply

    async def test_triage_failure_treats_the_message_as_a_question(self, boundary):
        with patch(f"{NODES}.get_provider", side_effect=RuntimeError("provider down")):
            graph = build_ismael_graph(MemorySaver())
            nodes, _ = await _turn(graph, "th9", "¿qué es la gracia?")
        assert nodes == ["ismael_triage", "ismael_start"]


class TestLanguage:
    def test_the_tenant_language_decides_not_the_moodle_users(self):
        from src.agents.ismael.texts import lang_of

        assert lang_of({"language": "es", "user_context": {"lms_lang": "en"}}) == "es"
        assert lang_of({"language": "en", "user_context": {"lms_lang": "es"}}) == "en"
        assert lang_of({"language": "es-CO"}) == "es"
        assert lang_of({"language": "pt"}) == "es"  # unsupported → Spanish
        assert lang_of({}) == "es"

    async def test_a_moodle_user_in_english_still_gets_the_tenants_spanish(self, boundary):
        graph = build_ismael_graph(MemorySaver())
        _, reply = await _turn(
            graph, "th-lang", "what is grace", language="es",
            user_context={"name": "Admin", "lms_lang": "en"},
        )
        assert "¿Cuál es tu rol?" in reply and "What is your role?" not in reply

    async def test_an_english_tenant_gets_english(self, boundary):
        graph = build_ismael_graph(MemorySaver())
        _, reply = await _turn(
            graph, "th-lang-en", "what is grace", language="en", user_context={"name": "Ana"}
        )
        assert "What is your role?" in reply


def _numbered_suffix(options: list[str]) -> str:
    # The canonical suffix the backend and the widget match the buttons against.
    return "\n" + "\n".join(f"{n}) {option}" for n, option in enumerate(options, start=1))


class TestQuickReplies:
    async def test_first_survey_question_offers_its_options_as_buttons(self, boundary):
        graph = build_ismael_graph(MemorySaver())
        _, reply, events = await _run(graph, "qr1", "que es la gracia")

        assert events == [
            {
                "type": "quick_replies",
                "options": ["Estudiante", "Docente", "Pastor o líder", "Otro"],
                "allow_other": False,
            }
        ]
        # The text still carries the numbered list, for every channel that
        # draws no buttons; it must end with exactly the canonical suffix.
        assert reply.endswith(_numbered_suffix(events[0]["options"]))

    async def test_each_step_offers_its_own_options_and_the_end_offers_none(self, boundary):
        graph = build_ismael_graph(MemorySaver())
        await _run(graph, "qr2", "que es la gracia")

        _, reply, events = await _run(graph, "qr2", "Estudiante")
        assert [e["options"] for e in events] == [SURVEY[1].labels("es")]
        assert reply.endswith(_numbered_suffix(SURVEY[1].labels("es")))

        _, reply, events = await _run(graph, "qr2", "Doctrina")
        assert [e["options"] for e in events] == [SURVEY[2].labels("es")]
        assert reply.endswith(_numbered_suffix(SURVEY[2].labels("es")))

        _, reply, events = await _run(graph, "qr2", "Curiosidad")
        assert reply == text("survey_done", "es") and events == []
        boundary["save"].assert_awaited_once_with(
            "k1", "c1", {"level": "student", "topic": "doctrine", "intendedUse": "curiosity"}
        )

    async def test_buttons_follow_the_tenant_language(self, boundary):
        graph = build_ismael_graph(MemorySaver())
        _, _, events = await _run(
            graph, "qr3", "what is grace", language="en", user_context={"name": "Ana"}
        )
        assert events[0]["options"] == ["Student", "Teacher", "Pastor or church leader", "Other"]

    async def test_replies_that_are_not_survey_questions_offer_no_buttons(self, boundary):
        graph = build_ismael_graph(MemorySaver())
        _, _, events = await _run(
            graph, "qr4", "que es la gracia", user_context={"name": "Ana", "ismael_survey_done": True}
        )
        assert events == []  # consulting

        rag_job._local_jobs["c1"] = ("job-1", "¿Qué es la gracia?", time.monotonic())
        _, _, events = await _run(graph, "qr5", "¿ya está?")
        assert events == []  # pending
        rag_job._local_jobs.clear()

        boundary["triage"]._result = TriageResult(intent=IsmaelIntent.OTHER)
        _, _, events = await _run(graph, "qr6", "¿cuándo es el examen?")
        assert events == []  # off topic

    @pytest.mark.parametrize("lang", ["es", "en"])
    @pytest.mark.parametrize("step_index", range(len(SURVEY)))
    def test_every_label_fits_the_transport_limits(self, step_index, lang):
        labels = SURVEY[step_index].labels(lang)
        # Mirrors acceptQuickReplies in the backend (src/messages/quick-replies.ts):
        # anything outside these limits would silently lose its buttons.
        assert 2 <= len(labels) <= 8
        assert len({label.casefold() for label in labels}) == len(labels)
        for label in labels:
            assert label == label.strip() and 1 <= len(label) <= 60
            assert not any(ord(c) < 32 or ord(c) == 127 for c in label)

    @pytest.mark.parametrize("lang", ["es", "en"])
    @pytest.mark.parametrize("step_index", range(len(SURVEY)))
    def test_a_clicked_label_is_read_without_the_llm(self, step_index, lang):
        step = SURVEY[step_index]
        for key, label in zip(step.keys(), step.labels(lang), strict=True):
            assert parse_survey_answer(step_index, label) == key


def _followups(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("type") == "pending_followup"]


class TestPendingFollowUp:
    WORKING = {
        "type": "pending_followup",
        "label": "Consultando la biblioteca…",
        "ttl_seconds": int(rag_job.JOB_TTL_SECONDS),
    }

    async def test_the_end_of_the_survey_promises_the_answer(self, boundary):
        graph = build_ismael_graph(MemorySaver())
        _, events = await _events(graph, "pf1", "que es la gracia")
        assert _followups(events) == []  # a survey question is not a wait
        for answer in ("Estudiante", "Doctrina"):
            await _events(graph, "pf1", answer)
        _, events = await _events(graph, "pf1", "Curiosidad")
        assert _followups(events) == [self.WORKING]

    async def test_consulting_and_pending_promise_it_too(self, boundary):
        graph = build_ismael_graph(MemorySaver())
        _, events = await _events(
            graph, "pf2", "que es la gracia", user_context={"name": "Ana", "ismael_survey_done": True}
        )
        assert _followups(events) == [self.WORKING]

        rag_job._local_jobs["c1"] = ("job-1", "¿Qué es la gracia?", time.monotonic())
        nodes, events = await _events(graph, "pf3", "¿ya está?")
        assert nodes[-1] == "ismael_pending" and _followups(events) == [self.WORKING]

    async def test_in_english(self, boundary):
        graph = build_ismael_graph(MemorySaver())
        _, events = await _events(
            graph, "pf4", "what is grace", language="en",
            user_context={"name": "Ana", "ismael_survey_done": True},
        )
        assert _followups(events)[0]["label"] == "Checking the library…"

    def test_it_fits_the_backend_limits(self):
        # Mirrors src/messages/pending-follow-up.ts: a longer ttl is capped,
        # a label over 80 characters would drop the indicator.
        assert rag_job.JOB_TTL_SECONDS <= 900
        for lang in ("es", "en"):
            assert 1 <= len(text("working", lang)) <= 80


class TestQuickRepliesOverSse:
    async def test_the_event_reaches_the_sse_stream_once_per_turn(self, boundary):
        graph = build_ismael_graph(MemorySaver())

        async def sse_turn(message: str) -> list[dict]:
            config = _config("qr-sse")
            config["configurable"]["turn_request_id"] = "req-1"
            events = []
            async for sse in _stream_graph(
                _inputs(message), config, graph=graph, agent_code_name="ismael"
            ):
                events.append(json.loads(sse.removeprefix("data: ").strip()))
            return [e for e in events if e.get("type") in ("quick_replies", "pending_followup")]

        first = await sse_turn("que es la gracia")
        assert first == [
            {"type": "quick_replies", "options": SURVEY[0].labels("es"), "allow_other": False}
        ]
        # The event is not state: the next turn carries only its own buttons.
        second = await sse_turn("Docente")
        assert [e["options"] for e in second] == [SURVEY[1].labels("es")]
        await sse_turn("Doctrina")
        last = await sse_turn("Curiosidad")
        assert last == [
            {
                "type": "pending_followup",
                "label": "Consultando la biblioteca…",
                "ttl_seconds": int(rag_job.JOB_TTL_SECONDS),
            }
        ]


class TestParseSurveyAnswer:
    @pytest.mark.parametrize(
        ("step", "reply", "expected"),
        [
            (0, "1", "student"),
            (0, "opción 3", "pastor"),
            (0, "soy docente", "teacher"),
            (1, "doctrina", "doctrine"),
            (1, "Historia de la iglesia", "church_history"),
            (2, "Curiosidad", "curiosity"),
            (2, "I'm preparing a sermon or a class", "sermon_or_class"),
        ],
    )
    def test_numbers_and_labels(self, step, reply, expected):
        assert parse_survey_answer(step, reply) == expected

    @pytest.mark.parametrize(("step", "reply"), [(0, "9"), (0, "no sé"), (1, "")])
    def test_out_of_range_or_unknown_needs_the_llm(self, step, reply):
        assert parse_survey_answer(step, reply) is None


# ── The background job ───────────────────────────────────────────────────────


ANSWERED = {
    "state": "done",
    "answer": {
        "state": "answered",
        "text": "La gracia es el favor inmerecido de Dios.",
        "citations": [
            {"chunk_id": "a", "locator": "", "claim": "", "page": 12},
            {"chunk_id": "a", "locator": "", "claim": "", "page": 12},
            {"chunk_id": "b", "locator": "", "claim": "", "section_title": "Capítulo 3"},
            {"chunk_id": "missing", "locator": "", "claim": ""},
        ],
        "evidence": [
            {"chunk_id": "a", "title": "Teología Sistemática"},
            {"chunk_id": "b", "title": "Historia de la Iglesia"},
        ],
    },
}


def _job(**over) -> rag_job.RagJob:
    base = dict(
        tenant_id="t1",
        conversation_id="c1",
        question="¿Qué es la gracia?",
        language="es",
        persona="Ismael",
        llm_config={"llm_provider": "openai", "openai_api_key": "sk-test"},
    )
    base.update(over)
    return rag_job.RagJob(**base)


class TestFormatAnswer:
    def test_references_are_deduplicated_in_citation_order(self):
        message = rag_job.format_answer(ANSWERED["answer"], "es")
        assert message == (
            "La gracia es el favor inmerecido de Dios.\n\n"
            "Referencias:\n"
            "- Teología Sistemática, p. 12\n"
            "- Historia de la Iglesia, Capítulo 3"
        )

    @pytest.mark.parametrize(
        "raw",
        [
            "Según los textos proporcionados, la gracia es un don.",
            "según las fuentes disponibles,  la gracia es un don.",
            "De acuerdo con los fragmentos, la gracia es un don.",
            "Con base en los documentos citados, la gracia es un don.",
        ],
    )
    def test_source_preamble_is_dropped(self, raw):
        assert rag_job.strip_source_preamble(raw) == "La gracia es un don."

    def test_english_source_preamble_is_dropped(self):
        raw = "According to the provided texts, grace is a gift."
        assert rag_job.strip_source_preamble(raw) == "Grace is a gift."

    @pytest.mark.parametrize(
        "raw",
        [
            "La gracia es un don, según los textos proporcionados.",
            "Según Agustín, la gracia es un don.",
            "Según los textos proporcionados,",
        ],
    )
    def test_everything_else_is_left_alone(self, raw):
        assert rag_job.strip_source_preamble(raw) == raw

    def test_format_answer_drops_the_preamble_before_the_references(self):
        text_ = "Según los textos proporcionados, la gracia es el favor inmerecido de Dios."
        answer = {**ANSWERED["answer"], "text": text_}
        expected = rag_job.format_answer(ANSWERED["answer"], "es")
        assert rag_job.format_answer(answer, "es") == expected

    def test_no_citations_means_no_reference_block(self):
        assert rag_job.format_answer({"text": "Hola"}, "es") == "Hola"
        assert rag_job.answer_references({"text": "Hola"}, "es") is None

    def test_references_as_data_match_the_block_exactly(self):
        refs = rag_job.answer_references(ANSWERED["answer"], "en")
        assert refs == {
            "heading": "References:",
            "items": [
                {"title": "Teología Sistemática", "detail": "p. 12"},
                {"title": "Historia de la Iglesia", "detail": "Capítulo 3"},
            ],
        }
        # The backend's referencesSuffix rendering of the same data.
        suffix = f"\n\n{refs['heading']}\n" + "\n".join(
            f"- {i['title']}" + (f", {i['detail']}" if i["detail"] else "") for i in refs["items"]
        )
        assert rag_job.format_answer(ANSWERED["answer"], "en").endswith(suffix)

    def test_a_detail_equal_to_the_title_is_left_out_of_both(self):
        answer = {
            "text": "Cuerpo.",
            "citations": [{"chunk_id": "a", "section_title": "Prólogo"}],
            "evidence": [{"chunk_id": "a", "title": "Prólogo"}],
        }
        assert rag_job.format_answer(answer, "es").endswith("\n- Prólogo")
        assert rag_job.answer_references(answer, "es")["items"] == [
            {"title": "Prólogo", "detail": None}
        ]


@pytest.fixture
def brain():
    with (
        patch("src.agents.ismael.rag_job.brain.ensure_started", new=AsyncMock(return_value="stopped")),
        patch("src.agents.ismael.rag_job.brain.wait_healthy", new=AsyncMock(return_value=True)) as healthy,
        patch("src.agents.ismael.rag_job.brain.ask", new=AsyncMock(return_value="q-1")) as ask,
        patch("src.agents.ismael.rag_job.brain.wait_answer", new=AsyncMock(return_value=ANSWERED)) as wait,
    ):
        yield {"healthy": healthy, "ask": ask, "wait": wait}


class TestCompose:
    async def test_answered_is_sent_verbatim_with_references(self, brain):
        message, usage, outcome, references = await rag_job._compose(_job())
        assert outcome == "answered" and usage == []
        assert message.startswith("La gracia es el favor inmerecido de Dios.")
        assert references == rag_job.answer_references(ANSWERED["answer"], "es")
        brain["ask"].assert_awaited_once_with("¿Qué es la gracia?")

    @pytest.mark.parametrize("state", ["insufficient_evidence", "off_corpus"])
    async def test_uncovered_question_gets_a_plain_general_answer(self, brain, state):
        brain["wait"].return_value = {"state": "done", "answer": {"state": state, "text": ""}}
        with patch(
            "src.agents.ismael.rag_job._general_answer",
            new=AsyncMock(return_value=("Respuesta breve.", [{"node": "x"}])),
        ):
            composed = await rag_job._compose(_job())
        assert composed == ("Respuesta breve.", [{"node": "x"}], f"general:{state}", None)

    async def test_general_answer_is_the_bare_reply_with_usage(self):
        class Provider(FakeProvider):
            async def chat(self, messages, model, **_):
                assert "120 palabras" in messages[0]["content"]
                assert "{persona}" not in messages[0]["content"]
                return "Breve."

        with patch("src.agents.ismael.rag_job.get_provider", return_value=Provider(None)):
            message, usage = await rag_job._general_answer(_job())
        assert message == "Breve."
        assert usage[0]["node"] == "ismael_general_answer"

    async def test_boot_timeout_apologises(self, brain):
        brain["healthy"].return_value = False
        message, _, outcome, _ = await rag_job._compose(_job())
        assert (message, outcome) == (text("failed", "es"), "boot_timeout")
        brain["ask"].assert_not_called()

    async def test_answer_timeout_and_failed_question_apologise(self, brain):
        brain["wait"].return_value = None
        assert (await rag_job._compose(_job()))[2] == "answer_timeout"
        brain["wait"].return_value = {"state": "failed", "error": {"kind": "x"}}
        assert (await rag_job._compose(_job()))[2] == "brain_failed"

    async def test_unexpected_error_still_produces_a_reply(self, brain):
        brain["ask"].side_effect = RuntimeError("boom")
        message, _, outcome, _ = await rag_job._compose(_job())
        assert (message, outcome) == (text("failed", "es"), "error")


class TestPostAgentMessage:
    async def test_references_ride_only_when_there_are_some(self):
        from src.agents import backend_client

        with patch.object(backend_client, "_post", new=AsyncMock(return_value={})) as post:
            await backend_client.post_agent_message("c1", job_id="j1", text="t")
            refs = {"heading": "Referencias:", "items": [{"title": "A", "detail": None}]}
            await backend_client.post_agent_message("c1", job_id="j2", text="t", references=refs)

        # The endpoint rejects unknown fields: an answer without sources must
        # not send the key at all, so it also works against an older backend.
        assert "references" not in post.await_args_list[0].kwargs["json"]
        assert post.await_args_list[1].kwargs["json"]["references"] == refs


class TestRunAndRegistry:
    async def test_run_delivers_once_and_frees_the_conversation(self, brain):
        job = _job()
        with (
            patch("src.agents.ismael.rag_job.get_store_or_none", return_value=None),
            patch(
                "src.agents.ismael.rag_job.backend_client.post_agent_message", new=AsyncMock()
            ) as post,
        ):
            await rag_job.spawn(job)
            assert await rag_job.running_job("t1", "c1") == {
                "job_id": job.job_id,
                "question": "¿Qué es la gracia?",
            }
            await asyncio.gather(*list(rag_job._tasks))

        post.assert_awaited_once()
        assert post.await_args.kwargs["job_id"] == job.job_id
        assert post.await_args.kwargs["references"]["heading"] == "Referencias:"
        with patch("src.agents.ismael.rag_job.get_store_or_none", return_value=None):
            assert await rag_job.running_job("t1", "c1") is None

    async def test_store_entry_past_its_ttl_is_not_running(self):
        class Item:
            value = {
                "job_id": "old",
                "status": "running",
                "question": "q",
                "started_at": "2020-01-01T00:00:00+00:00",
            }

        store = AsyncMock()
        store.aget.return_value = Item()
        with patch("src.agents.ismael.rag_job.get_store_or_none", return_value=store):
            assert await rag_job.running_job("t1", "c1") is None

    def test_job_config_keeps_only_what_the_job_needs(self):
        cfg = {
            "configurable": {
                "thread_id": "x",
                "llm_provider": "gemini",
                "gemini_project_id": "p",
                "prompts": {"THEOLOGY_GENERAL": {"content": "c"}, "THEOLOGY_TRIAGE": {}},
                "turn_request_id": "r",
            }
        }
        assert rag_job.job_llm_config(cfg) == {
            "llm_provider": "gemini",
            "gemini_project_id": "p",
            "prompts": {"THEOLOGY_GENERAL": {"content": "c"}},
        }


# ── Brain client: starting the host ──────────────────────────────────────────


BRAIN = "src.services.brain"


class TestEnsureStarted:
    async def test_skip_mode_never_touches_aws(self, monkeypatch):
        from src.services import brain

        monkeypatch.setattr(brain.settings, "brain_start_mode", "skip")
        with patch(f"{BRAIN}._google_id_token") as token:
            assert await brain.ensure_started() == "skipped"
        token.assert_not_called()

    async def test_unconfigured_role_is_an_error_not_an_exception(self, monkeypatch):
        from src.services import brain

        monkeypatch.setattr(brain.settings, "brain_start_mode", "aws")
        monkeypatch.setattr(brain.settings, "brain_aws_role_arn", "")
        assert await brain.ensure_started() == "error"

    @pytest.mark.parametrize(("state", "starts"), [("stopped", True), ("running", False), ("stopping", False)])
    def test_start_only_from_a_startable_state(self, monkeypatch, state, starts):
        from src.services import brain

        monkeypatch.setattr(brain.settings, "brain_aws_role_arn", "arn:aws:iam::1:role/r")
        monkeypatch.setattr(brain.settings, "brain_instance_id", "i-1")
        sts, ec2 = MagicMock(), MagicMock()
        sts.assume_role_with_web_identity.return_value = {
            "Credentials": {"AccessKeyId": "a", "SecretAccessKey": "s", "SessionToken": "t"}
        }
        ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"State": {"Name": state}}]}]
        }
        with patch("boto3.client", side_effect=lambda svc, **_: sts if svc == "sts" else ec2):
            assert brain._start_instance_sync("google-id-token") == state

        kwargs = sts.assume_role_with_web_identity.call_args.kwargs
        assert kwargs["WebIdentityToken"] == "google-id-token"
        assert kwargs["RoleArn"] == "arn:aws:iam::1:role/r"
        assert ec2.start_instances.called is starts

    async def test_failed_start_degrades_to_error(self, monkeypatch):
        from src.services import brain

        monkeypatch.setattr(brain.settings, "brain_start_mode", "aws")
        monkeypatch.setattr(brain.settings, "brain_aws_role_arn", "arn")
        monkeypatch.setattr(brain.settings, "brain_instance_id", "i-1")
        with patch(f"{BRAIN}._google_id_token", new=AsyncMock(side_effect=OSError("no metadata"))):
            assert await brain.ensure_started() == "error"


class TestWaitHealthy:
    async def test_retries_the_start_while_the_host_is_down(self, monkeypatch):
        from src.services import brain

        monkeypatch.setattr(brain.settings, "brain_poll_interval_seconds", 0)
        health = AsyncMock(side_effect=[False, False, False, True])
        with (
            patch(f"{BRAIN}.is_healthy", new=health),
            patch(f"{BRAIN}.ensure_started", new=AsyncMock()) as start,
        ):
            assert await brain.wait_healthy(time.monotonic() + 5, restart_every=0) is True
        assert start.await_count == 3

    async def test_gives_up_at_the_deadline(self):
        from src.services import brain

        with patch(f"{BRAIN}.is_healthy", new=AsyncMock(return_value=False)):
            assert await brain.wait_healthy(time.monotonic() - 1) is False
