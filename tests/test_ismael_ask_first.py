"""ismael asks before it solves or escalates, sends grade disputes to the
teacher, and closes with a service rating instead of the introduction.

Each case comes from a production conversation at Casa Roca (2026-10-07 to
2026-10-12). The graph runs for real on a MemorySaver; the LLM and the backend
are patched at the module boundary (fixture ``b`` from test_ismael_moodle).
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.agents.ismael import closing
from src.agents.ismael.common import FLOW_TTL_SECONDS, is_bare_closing
from src.agents.ismael.texts import text
from src.graphs.ismael_graph import build_ismael_graph
from src.schemas.ismael import (
    IsmaelIntent,
    MoodleSupportOutcome,
    MoodleSupportResult,
    TeacherStepAnswer,
    TriageResult,
)
from tests.test_ismael_moodle import (  # noqa: F401 — `b` is a fixture
    b,
    clarify_first,
    go,
    teachers_view,
    triage,
)

RATE_ES = ["👍 Me ayudó", "👎 No me ayudó"]


@pytest.fixture
def rating(b):
    """The backend's rating endpoint and the closing module's LLM."""
    with (
        patch(
            "src.agents.ismael.closing.backend_client.save_ismael_rating",
            new=AsyncMock(return_value={"messageId": "m-closing"}),
        ) as save,
        patch("src.agents.ismael.closing.get_provider", return_value=b["provider"]),
        patch("src.agents.ismael.closing.resolve_background_model", return_value="m"),
    ):
        yield save


def support(b, outcome, reply="", ticket="", ask=""):
    b["provider"].results[MoodleSupportResult] = MoodleSupportResult(
        reply=reply, outcome=outcome, ticket_description=ticket, clarifying_question=ask
    )


def no_messaging(b):
    b["lookup"].return_value = teachers_view(messagingEnabled=False)


# ── Ask before escalating ────────────────────────────────────────────────────


class TestAskBeforeEscalating:
    async def test_josihas_a_missing_button_only_asks_then_tickets(self, b):
        """07-10: the question and the ticket went out in ONE message."""
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "no aparece el botón para publicar")
        support(
            b,
            MoodleSupportOutcome.NEEDS_TICKET,
            reply="Hay que abrir un ticket.",
            ticket="No aparece el botón.",
            ask="¿En qué curso y actividad pasa, y te sale algún mensaje?",
        )
        graph = build_ismael_graph(MemorySaver())
        nodes, reply, _ = await go(graph, "j1", "no aparece el boton")

        assert nodes == ["ismael_triage", "ismael_moodle_support"]
        assert reply == "¿En qué curso y actividad pasa, y te sale algún mensaje?"
        assert "ryca.unidadeducativa.org/soporte" not in reply

        nodes, reply, _ = await go(graph, "j1", "sí, en Teología Social, no sale nada")
        assert nodes == ["ismael_triage", "ismael_moodle_support"]
        assert reply.startswith("Hay que abrir un ticket.")
        assert "https://ryca.unidadeducativa.org/soporte" in reply

    async def test_a_redirect_is_held_too(self, b):
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "matrícula")
        support(b, MoodleSupportOutcome.REDIRECT, reply="Escribe a Matrículas.", ask="¿Qué curso?")
        graph = build_ismael_graph(MemorySaver())
        _, reply, _ = await go(graph, "j2", "no me puedo inscribir en Sanidad Interior")
        assert reply == "¿Qué curso?"

    async def test_without_a_clarifying_question_the_generic_one_is_sent(self, b):
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "error")
        support(b, MoodleSupportOutcome.NEEDS_TICKET, reply="Ticket.", ticket="Error.")
        graph = build_ismael_graph(MemorySaver())
        _, reply, _ = await go(graph, "j3", "tengo un error")
        assert reply == text("support_clarify", "es")

    async def test_the_second_turn_tells_the_model_it_already_asked(self, b):
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "error")
        graph = build_ismael_graph(MemorySaver())
        await clarify_first(b, graph, "j4", "tengo un error")
        support(b, MoodleSupportOutcome.ANSWERED, reply="Prueba a recargar.")
        await go(graph, "j4", "en Griego, dice Course not found")

        systems = [m[0]["content"] for name, m in b["provider"].calls if name == "MoodleSupportResult"]
        assert "Caso abierto" not in systems[0]
        assert "ya respondió 1 pregunta" in systems[1]

    async def test_a_clear_how_to_is_answered_directly(self, b):
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "cómo entregar una tarea")
        support(b, MoodleSupportOutcome.ANSWERED, reply="Abre la tarea y pulsa «Agregar entrega».")
        graph = build_ismael_graph(MemorySaver())
        _, reply, _ = await go(graph, "j5", "¿cómo entrego la tarea?")
        assert reply == "Abre la tarea y pulsa «Agregar entrega»."

    async def test_steps_that_did_not_work_may_escalate_next_turn(self, b):
        """Steps count as the back-and-forth: «no me funcionó» is not asked again."""
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "no veo el curso")
        support(b, MoodleSupportOutcome.ANSWERED, reply="Revisa «Mis cursos».")
        graph = build_ismael_graph(MemorySaver())
        await go(graph, "j6", "no veo el curso de Griego")

        support(b, MoodleSupportOutcome.NEEDS_TICKET, reply="Abramos un ticket.", ticket="No veo Griego.")
        _, reply, _ = await go(graph, "j6", "no me funcionó, no aparece")
        assert reply.startswith("Abramos un ticket.") and "soporte" in reply

    async def test_an_answer_to_the_question_stays_in_the_case(self, b):
        """Triage reading the reply as "other" must not drop the case."""
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "error")
        graph = build_ismael_graph(MemorySaver())
        await clarify_first(b, graph, "j7", "tengo un error")

        triage(b, IsmaelIntent.OTHER)
        support(b, MoodleSupportOutcome.ANSWERED, reply="Eso se arregla así.")
        nodes, reply, _ = await go(graph, "j7", "en la semana 3")
        assert nodes == ["ismael_triage", "ismael_moodle_support"]
        assert reply == "Eso se arregla así."

        system = next(m for name, m in reversed(b["provider"].calls) if name == "TriageResult")[0]
        assert "Hay un caso de soporte abierto" in system["content"]

    async def test_more_detail_after_a_ticket_is_not_asked_about_again(self, b):
        """Replay 13-10: Josihas' extra detail after the ticket reopened a case
        and got the generic question again."""
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "botón")
        graph = build_ismael_graph(MemorySaver())
        await clarify_first(b, graph, "j9", "no aparece el botón")
        support(b, MoodleSupportOutcome.NEEDS_TICKET, reply="Abre un ticket.", ticket="x")
        await go(graph, "j9", "en Teología Social")

        support(b, MoodleSupportOutcome.NEEDS_TICKET, reply="Añádelo al ticket.", ticket="y", ask="¿Cuál?")
        _, reply, _ = await go(graph, "j9", "es en el foro de la tarea 3")
        assert reply.startswith("Añádelo al ticket.")

    async def test_an_ask_without_a_question_gets_one(self, b):
        """Replay 13-10: «necesito que me confirmes algunos detalles» asked nothing."""
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "inscripción")
        support(
            b,
            MoodleSupportOutcome.ASK,
            reply="Necesito que me confirmes algunos detalles.",
            ask="¿Qué mensaje te aparece al inscribirte?",
        )
        graph = build_ismael_graph(MemorySaver())
        _, reply, _ = await go(graph, "j10", "no me puedo inscribir")
        assert reply == "Necesito que me confirmes algunos detalles. ¿Qué mensaje te aparece al inscribirte?"

    async def test_a_long_ask_keeps_only_its_opening_before_the_question(self, b):
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "nota")
        support(
            b,
            MoodleSupportOutcome.ASK,
            reply="Hola, Ingrid. Entiendo tu duda. Necesito saber el curso y qué te genera duda.",
            ask="¿En qué curso está el cuestionario?",
        )
        graph = build_ismael_graph(MemorySaver())
        _, reply, _ = await go(graph, "j12", "duda con la nota")
        assert reply == "Hola, Ingrid. ¿En qué curso está el cuestionario?"

    async def test_a_held_teacher_matter_gets_a_grade_question(self, b):
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "nota pendiente")
        support(b, MoodleSupportOutcome.TEACHER, reply="Lo ve tu profesor.", ticket="x")
        graph = build_ismael_graph(MemorySaver())
        _, reply, _ = await go(graph, "j11", "tengo pendiente una nota de Evangelismo")
        assert reply == text("support_clarify_teacher", "es")

    async def test_the_case_expires(self, b):
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "error")
        graph = build_ismael_graph(MemorySaver())
        await clarify_first(b, graph, "j8", "tengo un error")

        support(b, MoodleSupportOutcome.NEEDS_TICKET, reply="Ticket.", ticket="x", ask="¿Cuál?")
        later = time.time() + FLOW_TTL_SECONDS + 1
        with patch("src.agents.ismael.common.time.time", return_value=later):
            _, reply, _ = await go(graph, "j8", "otro error distinto")
        assert reply == "¿Cuál?"


# ── Grades and quiz questions go to the teacher ──────────────────────────────


class TestAcademicToTeacher:
    async def test_jairo_a_wrong_quiz_question_reaches_the_teachers_email(self, b):
        """12-10: «la pregunta 8 tiene dos respuestas verdaderas» got a TECHNICAL ticket."""
        no_messaging(b)
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "pregunta 8 de la evaluación 3 mal planteada")
        support(
            b,
            MoodleSupportOutcome.TEACHER,
            reply="Eso lo revisa tu profesor.",
            ticket="La pregunta 8 de la evaluación 3 tiene dos respuestas verdaderas.",
            ask="¿De qué curso es la evaluación y qué opciones marcaste?",
        )
        graph = build_ismael_graph(MemorySaver())
        nodes, reply, _ = await go(graph, "g1", "en la pregunta 8 hay dos respuestas verdaderas")
        assert nodes == ["ismael_triage", "ismael_moodle_support"]
        assert reply == "¿De qué curso es la evaluación y qué opciones marcaste?"
        b["lookup"].assert_not_called()

        nodes, reply, _ = await go(graph, "g1", "Teología I, marqué la b y la c")
        assert nodes == ["ismael_triage", "ismael_moodle_support", "ismael_teacher"]
        assert reply.startswith(text("teacher_from_support", "es"))
        assert "juan@colegio.edu" in reply
        assert "soporte" not in reply  # never the technical form
        assert text("teacher_offer", "es") not in reply

    async def test_ingrid_a_disputed_grade_asks_then_goes_to_the_teacher(self, b):
        no_messaging(b)
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "duda con la calificación del cuestionario 3")
        support(b, MoodleSupportOutcome.ASK, reply="¿Qué nota ves y cuál esperabas?")
        graph = build_ismael_graph(MemorySaver())
        _, reply, _ = await go(graph, "g2", "tengo duda con la calificación del 3 cuestionario")
        assert reply == "¿Qué nota ves y cuál esperabas?"

        support(b, MoodleSupportOutcome.TEACHER, reply="Lo revisa tu profesor.", ticket="Saqué 46/50.")
        nodes, reply, _ = await go(graph, "g2", "me quedaron todas bien y saqué 46 sobre 50")
        assert nodes[-1] == "ismael_teacher"
        assert "juan@colegio.edu" in reply

    async def test_with_messaging_on_the_reason_goes_straight_to_a_draft(self, b):
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "nota")
        graph = build_ismael_graph(MemorySaver())
        await clarify_first(b, graph, "g3", "mi nota está mal")
        support(b, MoodleSupportOutcome.TEACHER, reply="x", ticket="Mi nota del quiz 2 está mal.")
        _, reply, quick = await go(graph, "g3", "el quiz 2 de Teología I")
        assert reply.startswith(text("teacher_from_support", "es"))
        assert quick[0]["options"] == ["Envíalo tú", "Le escribo yo"]


# ── Closing and the service rating ───────────────────────────────────────────


async def _helped(b, graph, thread):
    triage(b, IsmaelIntent.MOODLE_SUPPORT, "cómo entregar")
    support(b, MoodleSupportOutcome.ANSWERED, reply="Pulsa «Agregar entrega».")
    await go(graph, thread, "¿cómo entrego la tarea?")


class TestClosing:
    @pytest.mark.parametrize("message", ["gracias", "Gracias...", "ok 👍", "listo, gracias Salomón"])
    def test_bare_thanks_are_recognised(self, message):
        assert is_bare_closing(message, "Salomón")

    @pytest.mark.parametrize("message", ["gracias, pero sigue sin aparecer", "no me deja entrar", "buenas"])
    def test_requests_are_not_closings(self, message):
        assert not is_bare_closing(message, "Salomón")

    async def test_sonia_thanks_after_help_asks_for_a_rating(self, b, rating):
        """12-10: «Gracias...» got «¡Hola, Sonia…! Soy Salomón…» again."""
        graph = build_ismael_graph(MemorySaver())
        await _helped(b, graph, "c1")
        calls = len(b["provider"].calls)

        nodes, reply, quick = await go(graph, "c1", "Gracias...")
        assert nodes == ["ismael_triage", "ismael_closing"]
        assert len(b["provider"].calls) == calls  # no LLM call
        assert reply.startswith(text("closing_rate", "es", name=", Ana"))
        assert reply.endswith("1) 👍 Me ayudó\n2) 👎 No me ayudó")
        assert quick[0]["options"] == RATE_ES
        assert "Soy" not in reply

    async def test_thumbs_up_is_saved(self, b, rating):
        graph = build_ismael_graph(MemorySaver())
        await _helped(b, graph, "c2")
        await go(graph, "c2", "gracias")

        nodes, reply, _ = await go(graph, "c2", "👍 Me ayudó")
        assert nodes == ["ismael_triage", "ismael_rating"]
        rating.assert_awaited_once_with("c1", "UP", reason="", message_id="")
        assert reply == text("rating_thanks", "es")

    async def test_thumbs_down_asks_why_and_saves_the_reason(self, b, rating):
        graph = build_ismael_graph(MemorySaver())
        await _helped(b, graph, "c3")
        await go(graph, "c3", "gracias")

        _, reply, _ = await go(graph, "c3", "2")
        assert reply == text("rating_ask_reason", "es")
        rating.assert_awaited_with("c1", "DOWN", reason="", message_id="")

        nodes, reply, _ = await go(graph, "c3", "no me explicó dónde está el botón")
        assert nodes == ["ismael_triage", "ismael_rating"]
        rating.assert_awaited_with(
            "c1", "DOWN", reason="no me explicó dónde está el botón", message_id="m-closing"
        )
        assert reply == text("rating_reason_thanks", "es")

    async def test_asked_once_a_day(self, b, rating):
        graph = build_ismael_graph(MemorySaver())
        await _helped(b, graph, "c4")
        await go(graph, "c4", "gracias")
        await go(graph, "c4", "1")

        await _helped(b, graph, "c4")
        nodes, reply, quick = await go(graph, "c4", "gracias")
        assert nodes == ["ismael_triage", "ismael_closing"]
        assert reply == text("closing_plain", "es", name=", Ana") and quick == []

    async def test_nothing_to_rate_without_help(self, b, rating):
        graph = build_ismael_graph(MemorySaver())
        _, reply, quick = await go(graph, "c5", "gracias")
        assert reply == text("closing_plain", "es", name=", Ana") and quick == []

    async def test_a_new_question_instead_of_a_rating_is_answered(self, b, rating):
        graph = build_ismael_graph(MemorySaver())
        await _helped(b, graph, "c6")
        await go(graph, "c6", "gracias")

        b["provider"].results[TeacherStepAnswer] = TeacherStepAnswer(is_new_question=True)
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "dónde veo mis notas")
        support(b, MoodleSupportOutcome.ANSWERED, reply="En «Calificaciones».")
        nodes, reply, _ = await go(graph, "c6", "¿y dónde veo mis notas?")
        assert nodes == ["ismael_triage", "ismael_rating", "ismael_triage", "ismael_moodle_support"]
        assert reply == "En «Calificaciones»."
        rating.assert_not_awaited()

    async def test_a_failed_save_still_replies(self, b, rating):
        rating.side_effect = RuntimeError("backend down")
        graph = build_ismael_graph(MemorySaver())
        await _helped(b, graph, "c7")
        await go(graph, "c7", "gracias")
        _, reply, _ = await go(graph, "c7", "1")
        assert reply == text("rating_thanks", "es")

    async def test_thanks_closes_the_support_case(self, b, rating):
        triage(b, IsmaelIntent.MOODLE_SUPPORT, "error")
        graph = build_ismael_graph(MemorySaver())
        await clarify_first(b, graph, "c8", "tengo un error")
        await go(graph, "c8", "ok gracias")

        # "gracias" closed the case: a new problem starts a new one, asked again.
        support(b, MoodleSupportOutcome.NEEDS_TICKET, reply="Ticket.", ticket="x", ask="¿Cuál?")
        _, reply, _ = await go(graph, "c8", "otro problema")
        assert reply == "¿Cuál?"


class TestGreetingMidConversation:
    async def test_a_first_greeting_still_introduces(self, b):
        triage(b, IsmaelIntent.GREETING)
        graph = build_ismael_graph(MemorySaver())
        with patch("src.agents.greeting_response.get_provider", side_effect=RuntimeError):
            nodes, _, _ = await go(graph, "r1", "hola")
        assert nodes == ["ismael_triage", "greeting_response"]

    async def test_a_greeting_mid_conversation_does_not(self, b):
        graph = build_ismael_graph(MemorySaver())
        await _helped(b, graph, "r2")
        triage(b, IsmaelIntent.GREETING)
        nodes, reply, quick = await go(graph, "r2", "buenas")
        assert nodes == ["ismael_triage", "ismael_closing"]
        assert reply == text("closing_again", "es", name=", Ana") and quick == []

    async def test_a_closing_intent_from_the_model_is_a_closing(self, b, rating):
        b["provider"].results[TriageResult] = TriageResult(intent=IsmaelIntent.CLOSING)
        graph = build_ismael_graph(MemorySaver())
        nodes, _, _ = await go(graph, "r3", "mil bendiciones hermano, que esté bien")
        assert nodes == ["ismael_triage", "ismael_closing"]


def test_rating_latch_expires():
    assert closing.rating_open({"rating": {"step": "pending", "touched_at": time.time()}})
    assert not closing.rating_open({"rating": {"step": "pending", "touched_at": 0}})
