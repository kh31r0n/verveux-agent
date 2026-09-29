"""format_user_context: the user block spliced into the faq_response prompt.

The lms_* keys are sent by the backend only for Moodle contacts (see
buildChannelUserContext in yorchio-backend/src/contacts/moodle-profile.ts);
every other channel must render exactly as before.
"""

from src.agents.utils import format_user_context


def test_empty_context_renders_nothing():
    assert format_user_context({}) == ""
    assert format_user_context({"user_context": {}}) == ""


def test_non_lms_context_is_unchanged():
    out = format_user_context({"user_context": {"name": "Ana", "phone": "+573001112233"}})
    assert out == (
        "\n\nContexto del usuario (la persona con quien estás hablando):\n"
        "- Nombre: Ana\n"
        "- Teléfono: +573001112233"
    )


def test_moodle_context_adds_platform_course_and_enrolments():
    out = format_user_context(
        {
            "user_context": {
                "name": "Ana Pérez",
                "email": "ana@example.edu",
                "lms_platform": "Moodle",
                "lms_current_course": "Teología I",
                "lms_courses": "Teología I (Estudiante); Griego (Estudiante)",
                "name_capture_deferred": False,
            }
        }
    )
    assert "- Nombre: Ana Pérez" in out
    assert "- Email: ana@example.edu" in out
    assert "- Escribe desde: Moodle (usuario autenticado)" in out
    assert "- Curso que está viendo ahora: Teología I" in out
    assert "- Cursos matriculados (rol): Teología I (Estudiante); Griego (Estudiante)" in out


def test_moodle_context_without_a_current_course():
    out = format_user_context({"user_context": {"lms_platform": "Moodle"}})
    assert "Escribe desde: Moodle" in out
    assert "Curso que está viendo" not in out
