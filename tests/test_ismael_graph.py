"""ismael (theology over Company Brain): routing, the once-per-contact survey,
one Brain job per conversation, and how the job turns Brain's outcome into the
message the student receives.

The graph runs for real on a MemorySaver; the LLM, Brain and the backend are
patched at the module boundary.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agents.ismael import rag_job
from src.agents.ismael.nodes import parse_survey_answer
from src.agents.ismael.texts import text
from src.graphs.ismael_graph import build_ismael_graph
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


async def _turn(graph, thread: str, message: str, **extra) -> tuple[list[str], str]:
    nodes: list[str] = []
    tokens: list[str] = []
    async for mode, chunk in graph.astream(
        _inputs(message, **extra), config=_config(thread), stream_mode=["updates", "custom"]
    ):
        if mode == "updates":
            nodes.extend(k for k in chunk if not k.startswith("__"))
        elif isinstance(chunk, dict) and chunk.get("type") == "token":
            tokens.append(chunk["content"])
    return nodes, "".join(tokens)


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
    async def test_running_job_short_circuits_to_pending_without_an_llm_call(self, boundary):
        rag_job._local_jobs["c1"] = ("job-1", "¿Qué es la gracia?", time.monotonic())
        graph = build_ismael_graph(MemorySaver())
        nodes, reply = await _turn(graph, "th6", "¿ya tienes la respuesta?")

        assert nodes == ["ismael_triage", "ismael_pending"]
        assert "¿Qué es la gracia?" in reply
        assert boundary["triage"].calls == 0

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
    def test_moodle_language_wins_over_the_tenant_default(self):
        from src.agents.ismael.texts import lang_of

        assert lang_of({"language": "en", "user_context": {"lms_lang": "es"}}) == "es"
        assert lang_of({"language": "en", "user_context": {}}) == "en"
        assert lang_of({"language": "pt"}) == "es"  # unsupported → Spanish

    async def test_replies_follow_lms_lang(self, boundary):
        graph = build_ismael_graph(MemorySaver())
        _, reply = await _turn(
            graph, "th-lang", "what is grace", language="es",
            user_context={"name": "Ana", "lms_lang": "en"},
        )
        assert "What is your role?" in reply


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

    def test_no_citations_means_no_reference_block(self):
        assert rag_job.format_answer({"text": "Hola"}, "es") == "Hola"


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
        message, usage, outcome = await rag_job._compose(_job())
        assert outcome == "answered" and usage == []
        assert message.startswith("La gracia es el favor inmerecido de Dios.")
        brain["ask"].assert_awaited_once_with("¿Qué es la gracia?")

    @pytest.mark.parametrize("state", ["insufficient_evidence", "off_corpus"])
    async def test_uncovered_question_gets_a_labelled_general_answer(self, brain, state):
        brain["wait"].return_value = {"state": "done", "answer": {"state": state, "text": ""}}
        with patch(
            "src.agents.ismael.rag_job._general_answer",
            new=AsyncMock(return_value=("Respuesta breve.", [{"node": "x"}])),
        ):
            message, usage, outcome = await rag_job._compose(_job())
        assert (message, usage, outcome) == ("Respuesta breve.", [{"node": "x"}], f"general:{state}")

    async def test_general_answer_carries_the_notice_and_usage(self):
        class Provider(FakeProvider):
            async def chat(self, messages, model, **_):
                assert "120 palabras" in messages[0]["content"]
                assert "{persona}" not in messages[0]["content"]
                return "Breve."

        with patch("src.agents.ismael.rag_job.get_provider", return_value=Provider(None)):
            message, usage = await rag_job._general_answer(_job())
        assert message == f"Breve.\n\n{text('general_notice', 'es')}"
        assert usage[0]["node"] == "ismael_general_answer"

    async def test_boot_timeout_apologises(self, brain):
        brain["healthy"].return_value = False
        message, _, outcome = await rag_job._compose(_job())
        assert (message, outcome) == (text("failed", "es"), "boot_timeout")
        brain["ask"].assert_not_called()

    async def test_answer_timeout_and_failed_question_apologise(self, brain):
        brain["wait"].return_value = None
        assert (await rag_job._compose(_job()))[2] == "answer_timeout"
        brain["wait"].return_value = {"state": "failed", "error": {"kind": "x"}}
        assert (await rag_job._compose(_job()))[2] == "brain_failed"

    async def test_unexpected_error_still_produces_a_reply(self, brain):
        brain["ask"].side_effect = RuntimeError("boom")
        message, _, outcome = await rag_job._compose(_job())
        assert (message, outcome) == (text("failed", "es"), "error")


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
