"""ismael on Moodle: platform support (with the guided ticket) and the
"contact my teacher" flow — help first, then send through Moodle or suggest.

The graph runs for real on a MemorySaver; the LLM and the backend are patched
at the module boundary. One fake provider answers every structured call by
schema, so a whole multi-turn flow runs through it.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.agents.ismael import rag_job
from src.agents.ismael.texts import text
from src.graphs.ismael_graph import build_ismael_graph
from src.schemas.ismael import (
    FaqAnswer,
    IsmaelIntent,
    MoodleSupportOutcome,
    MoodleSupportResult,
    TeacherDraft,
    TeacherStepAnswer,
    TeacherTopic,
    TriageResult,
)
from tests.test_ismael_graph import _run, _turn

CTX = {
    "name": "Ana Pérez",
    "email": "ana@example.edu",
    "lms_platform": "Moodle",
    "lms_idnumber": "1020304050",
    "lms_support_url": "https://ryca.unidadeducativa.org/soporte",
    "lms_support_instructions": "Adjunta una captura JPG, PNG o PDF de hasta 2 MB.",
    "lms_other_contacts": "Matrículas: uebogota.matriculas@casaroca.org",
    "lms_teacher_messaging": "true",
    "ismael_survey_done": "true",
}

JUAN = {"id": "5", "name": "Prof. Juan", "messageUrl": "https://aulas/message/index.php?id=5"}
MARTA = {"id": "6", "name": "Prof. Marta", "messageUrl": "https://aulas/message/index.php?id=6"}


def teachers_view(**overrides) -> dict:
    view = {
        "available": True,
        "messagingEnabled": True,
        "currentCourseId": 7,
        "courses": [
            {
                "id": 7,
                "fullname": "Teología I",
                "participantsUrl": "https://aulas/user/index.php?id=7",
                "teachers": [JUAN],
            },
            {
                "id": 9,
                "fullname": "Griego",
                "participantsUrl": "https://aulas/user/index.php?id=9",
                "teachers": [MARTA, JUAN],
            },
        ],
    }
    view.update(overrides)
    return view


class SchemaProvider:
    """Answers each structured call with the result registered for its schema."""

    name = "openai"

    def __init__(self):
        from src.providers.base import UsageInfo

        self.last_usage = UsageInfo(input_tokens=10, output_tokens=5)
        self.results: dict = {}
        self.calls: list = []

    async def generate_structured(self, messages, model, schema, **_):
        self.calls.append((schema.__name__, messages))
        result = self.results.get(schema)
        if isinstance(result, Exception):
            raise result
        if result is None:
            raise RuntimeError(f"no fake result for {schema.__name__}")
        return result

    def count(self, schema) -> int:
        return sum(1 for name, _ in self.calls if name == schema.__name__)


@pytest.fixture(autouse=True)
def _clean_local_jobs():
    rag_job._local_jobs.clear()
    yield
    rag_job._local_jobs.clear()


@pytest.fixture
def b():
    provider = SchemaProvider()
    provider.results[TeacherDraft] = TeacherDraft(
        message="Hola Prof. Juan, no puedo abrir la lectura de la semana 3. Gracias, Ana."
    )
    with (
        patch("src.agents.ismael.nodes.get_provider", return_value=provider),
        patch("src.agents.ismael.support.get_provider", return_value=provider),
        patch("src.agents.ismael.teacher.get_provider", return_value=provider),
        patch("src.agents.ismael.faq.get_provider", return_value=provider),
        patch("src.agents.ismael.faq.resolve_model", return_value="m"),
        patch("src.agents.ismael.nodes.resolve_model", return_value="m"),
        patch("src.agents.ismael.support.resolve_model", return_value="m"),
        patch("src.agents.ismael.teacher.resolve_model", return_value="m"),
        patch("src.agents.ismael.nodes.rag_job.spawn", new=AsyncMock()) as spawn,
        patch("src.agents.ismael.nodes.rag_job.start_host_in_background"),
        patch("src.agents.ismael.rag_job.get_store_or_none", return_value=None),
        patch(
            "src.agents.ismael.teacher.backend_client.get_moodle_teachers",
            new=AsyncMock(return_value=teachers_view()),
        ) as lookup,
        patch(
            "src.agents.ismael.teacher.backend_client.post_teacher_message",
            new=AsyncMock(
                return_value={"status": "SENT", "teacherName": "Prof. Juan", "messageUrl": JUAN["messageUrl"]}
            ),
        ) as send,
    ):
        yield {"provider": provider, "spawn": spawn, "lookup": lookup, "send": send}


def triage(b, intent, question="", topic=TeacherTopic.NONE):
    b["provider"].results[TriageResult] = TriageResult(
        intent=intent, question=question, teacher_topic=topic
    )


async def go(graph, thread, message, ctx=CTX):
    return await _run(graph, thread, message, user_context=ctx)


# ── Moodle support ───────────────────────────────────────────────────────────


class TestMoodleSupport:
    async def test_how_to_question_is_answered_without_a_ticket(self, b):
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "cómo entregar una tarea")
        b["provider"].results[MoodleSupportResult] = MoodleSupportResult(
            reply="Entra al curso, abre la tarea y pulsa «Agregar entrega».",
            outcome=MoodleSupportOutcome.ANSWERED,
        )
        graph = build_ismael_graph(MemorySaver())
        nodes, reply, _ = await go(graph, "s1", "¿cómo entrego una tarea?")

        assert nodes == ["ismael_triage", "ismael_moodle_support"]
        assert reply == "Entra al curso, abre la tarea y pulsa «Agregar entrega»."
        b["spawn"].assert_not_called()

    async def test_the_prompt_carries_the_institutions_support_data(self, b):
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "matrícula")
        b["provider"].results[MoodleSupportResult] = MoodleSupportResult(
            reply="Escribe a Matrículas: uebogota.matriculas@casaroca.org",
            outcome=MoodleSupportOutcome.REDIRECT,
        )
        graph = build_ismael_graph(MemorySaver())
        await go(graph, "s2", "¿cómo pago la matrícula?")

        system = next(m for name, m in b["provider"].calls if name == "MoodleSupportResult")[0]
        assert "uebogota.matriculas@casaroca.org" in system["content"]
        assert "https://ryca.unidadeducativa.org/soporte" in system["content"]
        assert "{persona}" not in system["content"] and "{language_rule}" not in system["content"]

    async def test_needs_ticket_appends_the_form_and_the_users_data_in_code(self, b):
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "error en el cuestionario")
        b["provider"].results[MoodleSupportResult] = MoodleSupportResult(
            reply="Esto lo tiene que revisar soporte técnico.",
            outcome=MoodleSupportOutcome.NEEDS_TICKET,
            ticket_description="No puedo abrir el cuestionario 2 de Teología I; sale un error.",
        )
        graph = build_ismael_graph(MemorySaver())
        _, reply, _ = await go(graph, "s3", "me sale error al abrir el cuestionario")

        assert "https://ryca.unidadeducativa.org/soporte" in reply
        assert "1020304050" in reply and "Ana Pérez" in reply and "ana@example.edu" in reply
        assert "No puedo abrir el cuestionario 2" in reply
        assert reply.endswith("Adjunta una captura JPG, PNG o PDF de hasta 2 MB.")

    async def test_no_support_url_never_invents_a_link(self, b):
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "error")
        b["provider"].results[MoodleSupportResult] = MoodleSupportResult(
            reply="Hay que reportarlo.", outcome=MoodleSupportOutcome.NEEDS_TICKET,
            ticket_description="Me sale un error.",
        )
        ctx = {k: v for k, v in CTX.items() if not k.startswith("lms_support")}
        graph = build_ismael_graph(MemorySaver())
        _, reply, _ = await go(graph, "s4", "error", ctx=ctx)

        assert "http" not in reply
        assert text("ticket_no_url", "es") in reply and "Me sale un error." in reply

    async def test_llm_failure_still_guides_the_ticket(self, b):
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "no puedo entrar al curso")
        b["provider"].results[MoodleSupportResult] = RuntimeError("down")
        graph = build_ismael_graph(MemorySaver())
        _, reply, _ = await go(graph, "s5", "no puedo entrar al curso")

        assert reply.startswith(text("support_failed", "es"))
        assert "https://ryca.unidadeducativa.org/soporte" in reply

    async def test_a_running_brain_job_does_not_block_support(self, b):
        rag_job._local_jobs["c1"] = ("job-1", "¿Qué es la gracia?", time.monotonic())
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "calificaciones")
        b["provider"].results[MoodleSupportResult] = MoodleSupportResult(reply="Ve a «Calificaciones».")
        graph = build_ismael_graph(MemorySaver())
        nodes, _, _ = await go(graph, "s6", "¿dónde veo mis notas?")
        assert nodes == ["ismael_triage", "ismael_moodle_support"]


# ── Contacting a teacher ─────────────────────────────────────────────────────


class TestOfferHelpFirst:
    async def test_the_first_answer_is_an_offer_to_help(self, b):
        triage(b, IsmaelIntent.CONTACT_TEACHER)
        graph = build_ismael_graph(MemorySaver())
        nodes, reply, quick = await go(graph, "t1", "quiero hablar con mi profesor")

        assert nodes == ["ismael_triage", "ismael_teacher"]
        assert reply.startswith(text("teacher_offer", "es"))
        assert quick[0]["options"] == ["Sí, ayúdame", "No, quiero contactar al profesor"]
        assert reply.endswith("1) Sí, ayúdame\n2) No, quiero contactar al profesor")
        b["lookup"].assert_not_called()  # nothing looked up until they decline

    async def test_accepting_help_with_a_theology_reason_asks_the_library(self, b):
        triage(b, IsmaelIntent.CONTACT_TEACHER, "qué es la soteriología", TeacherTopic.THEOLOGY)
        graph = build_ismael_graph(MemorySaver())
        _, reply, _ = await go(graph, "t2", "quiero preguntarle al profe qué es la soteriología")
        assert reply.startswith(text("teacher_offer_theology", "es"))

        nodes, reply, _ = await go(graph, "t2", "Sí, ayúdame")
        assert nodes == ["ismael_triage", "ismael_teacher", "ismael_start"]
        assert b["spawn"].await_args.args[0].question == "qué es la soteriología"
        assert reply == text("consulting", "es")

    async def test_accepting_help_with_a_moodle_reason_answers_it(self, b):
        triage(b, IsmaelIntent.CONTACT_TEACHER, "no encuentro la tarea 2", TeacherTopic.MOODLE_SUPPORT)
        b["provider"].results[MoodleSupportResult] = MoodleSupportResult(reply="Está en la semana 2.")
        graph = build_ismael_graph(MemorySaver())
        await go(graph, "t3", "le quiero decir al profesor que no encuentro la tarea 2")

        nodes, reply, _ = await go(graph, "t3", "1")
        assert nodes == ["ismael_triage", "ismael_teacher", "ismael_moodle_support"]
        assert reply == "Está en la semana 2."
        user_msg = next(m for name, m in b["provider"].calls if name == "MoodleSupportResult")[-1]
        assert user_msg["content"] == "no encuentro la tarea 2"

    async def test_accepting_help_without_a_reason_asks_what_they_need(self, b):
        triage(b, IsmaelIntent.CONTACT_TEACHER)
        graph = build_ismael_graph(MemorySaver())
        await go(graph, "t4", "necesito al profesor")
        _, reply, _ = await go(graph, "t4", "sí, ayúdame")
        assert reply == text("teacher_tell_me", "es")

        # The flow is closed: the next message is triaged from scratch.
        triage(b, IsmaelIntent.GREETING)
        with patch("src.agents.greeting_response.get_provider", side_effect=RuntimeError):
            nodes, _, _ = await go(graph, "t4", "gracias")
        assert nodes == ["ismael_triage", "greeting_response"]

    async def test_a_new_question_abandons_the_flow_and_is_answered(self, b):
        triage(b, IsmaelIntent.CONTACT_TEACHER)
        graph = build_ismael_graph(MemorySaver())
        await go(graph, "t5", "quiero hablar con el profesor")

        b["provider"].results[TeacherStepAnswer] = TeacherStepAnswer(is_new_question=True)
        triage(b, IsmaelIntent.THEOLOGY, "¿Qué es la Trinidad?")
        nodes, _, _ = await go(graph, "t5", "mejor explícame la Trinidad")
        assert nodes == ["ismael_triage", "ismael_teacher", "ismael_triage", "ismael_start"]
        assert b["spawn"].await_args.args[0].question == "¿Qué es la Trinidad?"


class TestSendThroughMoodle:
    async def _decline(self, b, graph, thread, reason=""):
        triage(b, IsmaelIntent.CONTACT_TEACHER, reason)
        await go(graph, thread, "quiero escribirle al profesor")
        return await go(graph, thread, "No, quiero contactar al profesor")

    async def test_full_flow_sends_only_after_explicit_confirmation(self, b):
        graph = build_ismael_graph(MemorySaver())
        _, reply, quick = await self._decline(b, graph, "m1")
        assert reply.startswith(text("teacher_choose_mode", "es"))
        assert quick[0]["options"] == ["Envíalo tú", "Le escribo yo"]

        # Current course Teología I has one teacher: no course/teacher pick.
        _, reply, _ = await go(graph, "m1", "Envíalo tú")
        assert reply == text("teacher_compose", "es", teacher="Prof. Juan")

        _, reply, quick = await go(graph, "m1", "que no puedo abrir la lectura de la semana 3")
        assert "Hola Prof. Juan, no puedo abrir la lectura" in reply
        assert quick[0]["options"] == ["Enviar", "Cambiar algo", "Cancelar"]
        b["send"].assert_not_called()

        _, reply, _ = await go(graph, "m1", "Enviar")
        kwargs = b["send"].await_args.kwargs
        assert b["send"].await_args.args == ("c1",)
        assert kwargs["teacher_id"] == "5" and kwargs["course_id"] == 7
        assert kwargs["text"].startswith("Hola Prof. Juan")
        assert kwargs["idempotency_key"].startswith("c1:") and kwargs["language"] == "es"
        assert reply == text("teacher_sent", "es", teacher="Prof. Juan", url=JUAN["messageUrl"])

    async def test_a_known_reason_skips_compose_and_goes_to_the_draft(self, b):
        graph = build_ismael_graph(MemorySaver())
        await self._decline(b, graph, "m2", reason="pedirle una prórroga para el ensayo")
        _, reply, _ = await go(graph, "m2", "1")
        assert reply.startswith("Este es el mensaje para Prof. Juan")
        draft_input = [m for name, m in b["provider"].calls if name == "TeacherDraft"][0][-1]
        assert "pedirle una prórroga para el ensayo" in draft_input["content"]

    async def test_change_then_send(self, b):
        graph = build_ismael_graph(MemorySaver())
        await self._decline(b, graph, "m3", reason="no entiendo la tarea")
        await go(graph, "m3", "Envíalo tú")
        _, reply, _ = await go(graph, "m3", "Cambiar algo")
        assert reply == text("teacher_edit", "es")

        b["provider"].results[TeacherDraft] = TeacherDraft(message="Versión corta.")
        _, reply, _ = await go(graph, "m3", "hazlo más corto")
        assert "«Versión corta.»" in reply
        draft_input = [m for name, m in b["provider"].calls if name == "TeacherDraft"][-1][-1]
        assert "Cambio pedido: hazlo más corto" in draft_input["content"]

        await go(graph, "m3", "enviar")
        assert b["send"].await_args.kwargs["text"] == "Versión corta."

    async def test_cancel_sends_nothing(self, b):
        graph = build_ismael_graph(MemorySaver())
        await self._decline(b, graph, "m4", reason="una duda")
        await go(graph, "m4", "Envíalo tú")
        _, reply, _ = await go(graph, "m4", "Cancelar")
        assert reply == text("teacher_cancelled", "es")
        b["send"].assert_not_called()

    async def test_a_refusal_explains_and_gives_the_link(self, b):
        b["send"].return_value = {"status": "FAILED", "code": "recipient_blocks", "messageUrl": JUAN["messageUrl"]}
        graph = build_ismael_graph(MemorySaver())
        await self._decline(b, graph, "m5", reason="una duda")
        await go(graph, "m5", "Envíalo tú")
        _, reply, _ = await go(graph, "m5", "Enviar")
        assert "solo recibe mensajes de sus contactos" in reply
        assert JUAN["messageUrl"] in reply

    async def test_backend_down_still_gives_the_link(self, b):
        b["send"].side_effect = RuntimeError("timeout")
        graph = build_ismael_graph(MemorySaver())
        await self._decline(b, graph, "m6", reason="una duda")
        await go(graph, "m6", "Envíalo tú")
        _, reply, _ = await go(graph, "m6", "Enviar")
        assert JUAN["messageUrl"] in reply and "No pude enviar" in reply


class TestSuggestAndPick:
    async def test_write_yourself_gives_the_conversation_link(self, b):
        triage(b, IsmaelIntent.CONTACT_TEACHER)
        graph = build_ismael_graph(MemorySaver())
        await go(graph, "p1", "quiero hablar con mi profesor")
        await go(graph, "p1", "no")
        _, reply, _ = await go(graph, "p1", "Le escribo yo")
        assert reply == text("teacher_self_link", "es", teacher="Prof. Juan", url=JUAN["messageUrl"])
        b["send"].assert_not_called()

    async def test_messaging_off_skips_the_choice_and_only_suggests(self, b):
        b["lookup"].return_value = teachers_view(messagingEnabled=False)
        triage(b, IsmaelIntent.CONTACT_TEACHER)
        graph = build_ismael_graph(MemorySaver())
        await go(graph, "p2", "quiero hablar con mi profesor")
        _, reply, _ = await go(graph, "p2", "2")
        assert reply == text("teacher_self_link", "es", teacher="Prof. Juan", url=JUAN["messageUrl"])

    async def test_no_current_course_asks_the_course_then_the_teacher(self, b):
        b["lookup"].return_value = teachers_view(currentCourseId=None)
        triage(b, IsmaelIntent.CONTACT_TEACHER)
        graph = build_ismael_graph(MemorySaver())
        await go(graph, "p3", "quiero hablar con mi profesor")
        await go(graph, "p3", "No, quiero contactar al profesor")
        _, reply, quick = await go(graph, "p3", "Le escribo yo")
        assert reply.startswith(text("teacher_pick_course", "es"))
        assert quick[0]["options"] == ["Teología I", "Griego"]

        _, reply, quick = await go(graph, "p3", "Griego")
        assert quick[0]["options"] == ["Prof. Marta", "Prof. Juan"]
        _, reply, _ = await go(graph, "p3", "Prof. Marta")
        assert reply == text("teacher_self_link", "es", teacher="Prof. Marta", url=MARTA["messageUrl"])

    async def test_a_course_without_teachers_points_to_participants(self, b):
        view = teachers_view()
        view["courses"][0]["teachers"] = []
        b["lookup"].return_value = view
        triage(b, IsmaelIntent.CONTACT_TEACHER)
        graph = build_ismael_graph(MemorySaver())
        await go(graph, "p4", "quiero hablar con mi profesor")
        await go(graph, "p4", "no")
        _, reply, _ = await go(graph, "p4", "Le escribo yo")
        assert "https://aulas/user/index.php?id=7" in reply

    async def test_lookup_failure_still_offers_a_way(self, b):
        b["lookup"].side_effect = RuntimeError("down")
        triage(b, IsmaelIntent.CONTACT_TEACHER)
        graph = build_ismael_graph(MemorySaver())
        await go(graph, "p5", "quiero hablar con mi profesor")
        _, reply, _ = await go(graph, "p5", "no")
        assert reply == text("teacher_lookup_failed", "es")

    async def test_teacher_flow_steps_never_call_triage(self, b):
        triage(b, IsmaelIntent.CONTACT_TEACHER)
        graph = build_ismael_graph(MemorySaver())
        await go(graph, "p6", "quiero hablar con mi profesor")
        await go(graph, "p6", "no")
        await go(graph, "p6", "Le escribo yo")
        assert b["provider"].count(TriageResult) == 1

    async def test_a_stale_flow_expires(self, b):
        triage(b, IsmaelIntent.CONTACT_TEACHER)
        graph = build_ismael_graph(MemorySaver())
        await go(graph, "p7", "quiero hablar con mi profesor")
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "notas")
        b["provider"].results[MoodleSupportResult] = MoodleSupportResult(reply="Ve a «Calificaciones».")
        with patch("src.agents.ismael.teacher.time.time", return_value=time.time() + 3600):
            nodes, _, _ = await go(graph, "p7", "¿dónde veo mis notas?")
        assert nodes == ["ismael_triage", "ismael_moodle_support"]

    async def test_english_tenant(self, b):
        triage(b, IsmaelIntent.CONTACT_TEACHER)
        graph = build_ismael_graph(MemorySaver())
        nodes, reply = await _turn(graph, "p8", "I want to talk to my teacher", user_context=CTX, language="en")
        assert reply.startswith(text("teacher_offer", "en"))
        assert reply.endswith("1) Yes, help me\n2) No, I want to contact my teacher")


async def test_long_course_names_are_shortened_for_buttons_and_matching(b):
    long_name = "Seminario de Teología Sistemática Avanzada: Cristología y Soteriología (2026-II)"
    view = teachers_view(currentCourseId=None)
    view["courses"][1]["fullname"] = long_name
    b["lookup"].return_value = view
    triage(b, IsmaelIntent.CONTACT_TEACHER)
    graph = build_ismael_graph(MemorySaver())
    await go(graph, "l1", "quiero hablar con mi profesor")
    await go(graph, "l1", "no")
    _, _, quick = await go(graph, "l1", "Le escribo yo")
    label = quick[0]["options"][1]
    assert len(label) <= 60 and label.endswith("…")

    _, _, quick = await go(graph, "l1", label)  # the button sends its own label back
    assert quick[0]["options"] == ["Prof. Marta", "Prof. Juan"]


# ── Institution FAQs ─────────────────────────────────────────────────────────

FAQS = [
    {
        "id": "faq-1",
        "question": "¿Cuándo son los exámenes finales?",
        "answer": "Los exámenes finales son del 1 al 5 de diciembre.",
        "category": "calendario",
        "score": 0.42,
    },
    {
        "id": "faq-2",
        "question": "¿Qué enseña la institución sobre la Trinidad?",
        "answer": "Un solo Dios en tres personas: Padre, Hijo y Espíritu Santo.",
        "category": "doctrina",
        "score": 0.31,
    },
]


async def faq_turn(graph, thread, message, faqs=FAQS):
    return await _run(graph, thread, message, user_context=CTX, faqs=faqs)


class TestFaqs:
    async def test_triage_sees_the_candidates_and_a_match_is_answered_from_the_faq(self, b):
        b["provider"].results[TriageResult] = TriageResult(
            intent=IsmaelIntent.OTHER, faq_id="faq-1"
        )
        b["provider"].results[FaqAnswer] = FaqAnswer(reply="Del 1 al 5 de diciembre.")
        graph = build_ismael_graph(MemorySaver())
        nodes, reply, _ = await faq_turn(graph, "f1", "¿cuándo son los finales?")

        assert nodes == ["ismael_triage", "ismael_faq"]
        assert reply == "Del 1 al 5 de diciembre."
        triage_system = b["provider"].calls[0][1][0]["content"]
        assert "id=faq-1" in triage_system and "id=faq-2" in triage_system
        faq_system = next(m for name, m in b["provider"].calls if name == "FaqAnswer")[0]["content"]
        assert "Los exámenes finales son del 1 al 5 de diciembre." in faq_system

    async def test_a_theology_faq_never_reaches_brain_nor_the_survey(self, b):
        b["provider"].results[TriageResult] = TriageResult(
            intent=IsmaelIntent.THEOLOGY, question="¿Qué es la Trinidad?", faq_id="faq-2"
        )
        b["provider"].results[FaqAnswer] = FaqAnswer(reply="Un solo Dios en tres personas.")
        ctx = {k: v for k, v in CTX.items() if k != "ismael_survey_done"}
        graph = build_ismael_graph(MemorySaver())
        nodes, _, _ = await _run(graph, "f2", "¿qué es la Trinidad?", user_context=ctx, faqs=FAQS)
        assert nodes == ["ismael_triage", "ismael_faq"]
        b["spawn"].assert_not_called()

    async def test_reports_the_used_faq_for_analytics(self, b):
        b["provider"].results[TriageResult] = TriageResult(intent=IsmaelIntent.OTHER, faq_id="faq-1")
        b["provider"].results[FaqAnswer] = FaqAnswer(reply="ok")
        graph = build_ismael_graph(MemorySaver())
        await faq_turn(graph, "f3", "¿cuándo son los finales?")
        state = await graph.aget_state({"configurable": {"thread_id": "f3"}})
        assert state.values["faq_used"] == [
            {"id": "faq-1", "question": "¿Cuándo son los exámenes finales?", "confidence": 0.42}
        ]

    async def test_an_invented_faq_id_is_ignored(self, b):
        b["provider"].results[TriageResult] = TriageResult(
            intent=IsmaelIntent.MOODLE_SUPPORT, question="notas", faq_id="faq-999"
        )
        b["provider"].results[MoodleSupportResult] = MoodleSupportResult(reply="Ve a «Calificaciones».")
        graph = build_ismael_graph(MemorySaver())
        nodes, _, _ = await faq_turn(graph, "f4", "¿dónde veo mis notas?")
        assert nodes == ["ismael_triage", "ismael_moodle_support"]

    async def test_rewrite_failure_sends_the_faq_answer_verbatim(self, b):
        b["provider"].results[TriageResult] = TriageResult(intent=IsmaelIntent.OTHER, faq_id="faq-1")
        b["provider"].results[FaqAnswer] = RuntimeError("down")
        graph = build_ismael_graph(MemorySaver())
        _, reply, _ = await faq_turn(graph, "f5", "¿cuándo son los finales?")
        assert reply == "Los exámenes finales son del 1 al 5 de diciembre."

    async def test_a_faq_answers_even_while_a_brain_job_runs(self, b):
        rag_job._local_jobs["c1"] = ("job-1", "¿Qué es la gracia?", time.monotonic())
        b["provider"].results[TriageResult] = TriageResult(
            intent=IsmaelIntent.THEOLOGY, faq_id="faq-2"
        )
        b["provider"].results[FaqAnswer] = FaqAnswer(reply="Un solo Dios.")
        graph = build_ismael_graph(MemorySaver())
        nodes, _, _ = await faq_turn(graph, "f6", "¿y la Trinidad?")
        assert nodes == ["ismael_triage", "ismael_faq"]

    async def test_a_greeting_stays_a_greeting_even_with_a_faq_id(self, b):
        b["provider"].results[TriageResult] = TriageResult(intent=IsmaelIntent.GREETING, faq_id="faq-1")
        graph = build_ismael_graph(MemorySaver())
        with patch("src.agents.greeting_response.get_provider", side_effect=RuntimeError):
            nodes, _, _ = await faq_turn(graph, "f7", "hola")
        assert nodes == ["ismael_triage", "greeting_response"]

    async def test_teacher_flow_offers_the_faq_first_and_answers_it_on_yes(self, b):
        b["provider"].results[TriageResult] = TriageResult(
            intent=IsmaelIntent.CONTACT_TEACHER,
            question="cuándo son los exámenes finales",
            faq_id="faq-1",
        )
        b["provider"].results[FaqAnswer] = FaqAnswer(reply="Del 1 al 5 de diciembre.")
        graph = build_ismael_graph(MemorySaver())
        _, reply, _ = await faq_turn(graph, "f8", "le quiero preguntar al profe cuándo son los finales")
        assert reply.startswith(text("teacher_offer_faq", "es"))

        # This turn's own candidates no longer contain faq-1: the flow kept it.
        nodes, reply, _ = await faq_turn(graph, "f8", "Sí, ayúdame", faqs=[])
        assert nodes == ["ismael_triage", "ismael_teacher", "ismael_faq"]
        assert reply == "Del 1 al 5 de diciembre."
        user_msg = next(m for name, m in b["provider"].calls if name == "FaqAnswer")[-1]
        assert user_msg["content"] == "cuándo son los exámenes finales"


async def test_a_hung_model_call_times_out_and_the_turn_still_replies(b, monkeypatch):
    """2026-10-06: a support call that never returned held the conversation's
    lock for minutes; now it is cut off and the fallback goes out in time."""
    import asyncio

    from src.config import settings

    monkeypatch.setattr(settings, "ismael_timeout_answer_seconds", 0.05)
    triage(b, IsmaelIntent.MOODLE_SUPPORT, "Course not found")

    async def hang(messages, model, schema, **kwargs):
        if schema is MoodleSupportResult:
            await asyncio.sleep(3600)
        return b["provider"].results[schema]

    b["provider"].generate_structured = hang
    graph = build_ismael_graph(MemorySaver())
    nodes, reply, _ = await asyncio.wait_for(go(graph, "h1", "El error es Course not found"), 5)
    assert nodes == ["ismael_triage", "ismael_moodle_support"]
    assert reply.startswith(text("support_failed", "es"))
    assert "https://ryca.unidadeducativa.org/soporte" in reply


async def test_every_ismael_call_caps_thinking(b):
    seen = []
    original = b["provider"].generate_structured

    async def spy(messages, model, schema, **kwargs):
        seen.append((schema.__name__, kwargs.get("thinking_budget")))
        return await original(messages, model, schema)

    b["provider"].generate_structured = spy
    triage(b, IsmaelIntent.MOODLE_SUPPORT, "notas")
    b["provider"].results[MoodleSupportResult] = MoodleSupportResult(reply="ok")
    graph = build_ismael_graph(MemorySaver())
    await go(graph, "h2", "¿dónde veo mis notas?")
    assert seen == [("TriageResult", 0), ("MoodleSupportResult", 0)]


class TestTriageFallback:
    """2026-10-06: a triage timeout routed "Tengo un error con moodle" to the
    library. A failed triage now routes by keyword, newest fragment first."""

    async def test_a_moodle_error_goes_to_support_when_triage_fails(self, b):
        b["provider"].results[TriageResult] = TimeoutError()
        b["provider"].results[MoodleSupportResult] = MoodleSupportResult(reply="Revisemos el error.")
        graph = build_ismael_graph(MemorySaver())
        nodes, reply, _ = await go(graph, "fb1", "Tengo un error con moodle")
        assert nodes == ["ismael_triage", "ismael_moodle_support"]
        b["spawn"].assert_not_called()

    async def test_a_teacher_request_opens_the_teacher_flow_when_triage_fails(self, b):
        b["provider"].results[TriageResult] = TimeoutError()
        graph = build_ismael_graph(MemorySaver())
        nodes, reply, _ = await go(graph, "fb2", "Me gustaría comunicarme con el Profesor Jaime Quiceno")
        assert nodes == ["ismael_triage", "ismael_teacher"]
        assert reply.startswith(text("teacher_offer", "es"))

    async def test_a_theology_question_still_reaches_the_library(self, b):
        b["provider"].results[TriageResult] = TimeoutError()
        graph = build_ismael_graph(MemorySaver())
        nodes, _, _ = await go(graph, "fb3", "¿Qué dice Pablo sobre la justificación?")
        assert nodes == ["ismael_triage", "ismael_start"]
        b["spawn"].assert_awaited_once()


def test_fallback_intent_reads_the_newest_fragment_first():
    from src.agents.ismael.common import fallback_intent

    burst = [
        "El error que me aparece es Course not found",
        "Me gustaría comunicarme con el Profesor Jaime Quiceno",
        "Tengo un error con moodle",
    ]
    assert fallback_intent(burst) == IsmaelIntent.MOODLE_SUPPORT
    assert fallback_intent(burst[:2]) == IsmaelIntent.CONTACT_TEACHER
    assert fallback_intent(["¿Quién fue el apóstol Pedro?"]) == IsmaelIntent.THEOLOGY
    assert fallback_intent([]) == IsmaelIntent.THEOLOGY


async def test_support_fallback_sends_an_admin_question_to_the_contacts_not_the_form(b):
    triage(b, IsmaelIntent.MOODLE_SUPPORT, "cómo pago la matrícula")
    b["provider"].results[MoodleSupportResult] = TimeoutError()
    graph = build_ismael_graph(MemorySaver())
    _, reply, _ = await go(graph, "sf1", "¿Cómo pago la matrícula del próximo semestre?")
    assert "uebogota.matriculas@casaroca.org" in reply
    assert "ryca.unidadeducativa.org/soporte" not in reply
