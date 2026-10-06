"""Fixed wording for ismael: the survey, the holding replies and the fallbacks.

Deterministic on purpose. These replies are sent while Brain is booting or
answering, so they must not depend on an LLM call succeeding, and the survey
options are statistics whose keys must never drift.

The tenant's language (TenantSettings.language, sent as ``language``) decides:
Spanish, or English for an English-speaking tenant. The Moodle user's own
language does not.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurveyStep:
    field: str
    # option key → (es label, en label). Order is the number the user types.
    options: tuple[tuple[str, str, str], ...]
    question: dict[str, str]

    def labels(self, lang: str) -> list[str]:
        """Visible option labels, in the order of their numbers."""
        idx = 2 if lang == "en" else 1
        return [option[idx] for option in self.options]

    def render(self, lang: str) -> str:
        # The question plus the numbered list the backend matches the
        # quick-reply buttons against (``emit_quick_replies``); keep the format.
        lines = [self.question[lang]]
        for n, label in enumerate(self.labels(lang), start=1):
            lines.append(f"{n}) {label}")
        return "\n".join(lines)

    def keys(self) -> list[str]:
        return [key for key, _, _ in self.options]


SURVEY: tuple[SurveyStep, ...] = (
    SurveyStep(
        field="level",
        options=(
            ("student", "Estudiante", "Student"),
            ("teacher", "Docente", "Teacher"),
            ("pastor", "Pastor o líder", "Pastor or church leader"),
            ("other", "Otro", "Other"),
        ),
        question={"es": "¿Cuál es tu rol?", "en": "What is your role?"},
    ),
    SurveyStep(
        field="topic",
        options=(
            ("bible", "Biblia", "Bible"),
            ("doctrine", "Doctrina", "Doctrine"),
            ("church_history", "Historia de la Iglesia", "Church history"),
            ("pastoral", "Pastoral y homilética", "Pastoral care and preaching"),
            ("other", "Otro", "Other"),
        ),
        question={
            "es": "¿Qué tema te interesa más?",
            "en": "Which topic interests you most?",
        },
    ),
    SurveyStep(
        field="intendedUse",
        options=(
            ("personal_study", "Estudio personal", "Personal study"),
            ("coursework", "Tarea del curso", "Coursework"),
            ("sermon_or_class", "Preparar un sermón o una clase", "Preparing a sermon or a class"),
            ("curiosity", "Curiosidad", "Curiosity"),
        ),
        question={
            "es": "¿Para qué vas a usar la respuesta?",
            "en": "What will you use the answer for?",
        },
    ),
)

NO_ANSWER = "no_answer"

_TEXTS: dict[str, dict[str, str]] = {
    "survey_intro": {
        "es": (
            "Buena pregunta{name}. Mientras preparo las fuentes de la biblioteca "
            "(tarda alrededor de un minuto), te hago tres preguntas rápidas. Son solo "
            "para estadística y no cambian la respuesta; elige una opción o escribe tu respuesta."
        ),
        "en": (
            "Good question{name}. While I get the library's sources ready (it takes "
            "about a minute), here are three quick questions. They are only for "
            "statistics and do not change the answer; pick an option or type your answer."
        ),
    },
    "survey_next": {"es": "Gracias.", "en": "Thanks."},
    "survey_done": {
        "es": (
            "¡Gracias! Estoy consultando la biblioteca; te escribo aquí mismo en cuanto "
            "tenga la respuesta con sus referencias."
        ),
        "en": (
            "Thank you! I'm checking the library; I'll write back right here as soon "
            "as I have the answer with its references."
        ),
    },
    "consulting": {
        "es": (
            "Estoy consultando la biblioteca de teología; te respondo aquí mismo en "
            "cuanto tenga la respuesta con sus referencias (puede tardar un par de minutos)."
        ),
        "en": (
            "I'm checking the theology library; I'll answer right here as soon as I "
            "have the answer with its references (it may take a couple of minutes)."
        ),
    },
    "pending": {
        "es": (
            "Sigo consultando tu pregunta anterior: «{question}». En cuanto tenga la "
            "respuesta te la envío; después puedes hacerme otra."
        ),
        "en": (
            "I'm still working on your previous question: “{question}”. I'll send the "
            "answer as soon as I have it; then you can ask another one."
        ),
    },
    "off_topic": {
        "es": (
            "Soy {persona}: respondo preguntas de teología cristiana, te ayudo con el "
            "aula virtual y puedo ponerte en contacto con tu profesor. ¿En qué te ayudo?"
        ),
        "en": (
            "I'm {persona}: I answer questions about Christian theology, help with the "
            "virtual classroom and can put you in touch with your teacher. How can I help?"
        ),
    },
    # ── Moodle support ──
    "support_failed": {
        "es": "Lo siento, no pude revisar tu consulta del aula virtual en este momento.",
        "en": "Sorry, I couldn't look into your virtual-classroom question right now.",
    },
    "ticket_intro": {
        "es": "Para abrir el ticket entra a {url} y completa el formulario con estos datos:",
        "en": "To open the ticket, go to {url} and fill in the form with these details:",
    },
    "ticket_no_url": {
        "es": "Comunícate con el soporte técnico de tu institución y cuéntales esto:",
        "en": "Contact your institution's technical support and tell them this:",
    },
    "ticket_document": {"es": "Nro. de documento", "en": "ID number"},
    "ticket_document_missing": {
        "es": "tu número de documento",
        "en": "your ID number",
    },
    "ticket_name": {"es": "Nombres y apellidos", "en": "Full name"},
    "ticket_email": {"es": "Correo", "en": "Email"},
    "ticket_description": {"es": "Descripción", "en": "Description"},
    # ── Contacting a teacher ──
    "teacher_offer_theology": {
        "es": (
            "Antes de escribirle a tu profesor: ¿quieres que te ayude yo con eso? "
            "Puedo buscarlo en la biblioteca de teología."
        ),
        "en": (
            "Before writing to your teacher: would you like me to help with that? "
            "I can look it up in the theology library."
        ),
    },
    "teacher_offer_moodle": {
        "es": (
            "Antes de escribirle a tu profesor: ¿quieres que te ayude yo con eso? "
            "Puedo orientarte con el aula virtual."
        ),
        "en": (
            "Before writing to your teacher: would you like me to help with that? "
            "I can guide you through the virtual classroom."
        ),
    },
    "teacher_offer_faq": {
        "es": (
            "Antes de escribirle a tu profesor: la institución ya tiene una respuesta "
            "para eso. ¿Quieres que te la dé?"
        ),
        "en": (
            "Before writing to your teacher: the institution already has an answer "
            "for that. Would you like it?"
        ),
    },
    "teacher_offer": {
        "es": (
            "Claro. Antes de escribirle a tu profesor, ¿puedo ayudarte yo? Respondo "
            "dudas de teología y del aula virtual."
        ),
        "en": (
            "Sure. Before writing to your teacher, can I help you myself? I answer "
            "questions about theology and the virtual classroom."
        ),
    },
    "teacher_tell_me": {
        "es": "Con gusto. Cuéntame qué necesitas.",
        "en": "Happy to. Tell me what you need.",
    },
    "teacher_choose_mode": {
        "es": "De acuerdo. ¿Quieres que le envíe yo el mensaje o prefieres escribirle tú?",
        "en": "All right. Should I send the message for you, or would you rather write yourself?",
    },
    "teacher_pick_course": {
        "es": "¿A qué curso pertenece el profesor?",
        "en": "Which course is the teacher in?",
    },
    "teacher_pick_teacher": {
        "es": "¿A cuál de tus profesores de «{course}»?",
        "en": "Which of your teachers in “{course}”?",
    },
    "teacher_none": {
        "es": (
            "No encontré profesores publicados para «{course}». Puedes ver a los "
            "participantes del curso aquí: {url}"
        ),
        "en": "I couldn't find any listed teachers for “{course}”. You can see the course participants here: {url}",
    },
    "teacher_no_courses": {
        "es": (
            "No encontré tus cursos. Puedes escribirle a tu profesor desde la "
            "mensajería de Moodle o desde «Participantes» dentro del curso."
        ),
        "en": (
            "I couldn't find your courses. You can write to your teacher from Moodle's "
            "messaging or from “Participants” inside the course."
        ),
    },
    "teacher_lookup_failed": {
        "es": (
            "No pude consultar tus cursos ahora. Puedes escribirle a tu profesor desde "
            "la mensajería de Moodle o desde «Participantes» dentro del curso."
        ),
        "en": (
            "I couldn't look up your courses right now. You can write to your teacher "
            "from Moodle's messaging or from “Participants” inside the course."
        ),
    },
    "teacher_self_link": {
        "es": "Puedes escribirle a {teacher} directamente aquí: {url}",
        "en": "You can write to {teacher} directly here: {url}",
    },
    "teacher_compose": {
        "es": "¿Qué quieres decirle a {teacher}?",
        "en": "What would you like to tell {teacher}?",
    },
    "teacher_confirm": {
        "es": "Este es el mensaje para {teacher}:\n\n«{draft}»\n\n¿Lo envío?",
        "en": "Here is the message for {teacher}:\n\n“{draft}”\n\nShall I send it?",
    },
    "teacher_edit": {
        "es": "¿Qué quieres cambiar?",
        "en": "What would you like to change?",
    },
    "teacher_cancelled": {
        "es": "Listo, no envié nada. Si necesitas algo más, aquí estoy.",
        "en": "Okay, I didn't send anything. If you need anything else, I'm here.",
    },
    "teacher_sent": {
        "es": (
            "Listo, le envié tu mensaje a {teacher}. Te responderá por la mensajería "
            "de Moodle; también puedes ver la conversación aquí: {url}"
        ),
        "en": (
            "Done, I sent your message to {teacher}. They will reply through Moodle's "
            "messaging; you can also open the conversation here: {url}"
        ),
    },
    "teacher_send_failed": {
        "es": "No pude enviar el mensaje: {reason}. Puedes escribirle a {teacher} directamente aquí: {url}",
        "en": "I couldn't send the message: {reason}. You can write to {teacher} directly here: {url}",
    },
    "teacher_reason_recipient_blocks": {
        "es": "{teacher} solo recibe mensajes de sus contactos en Moodle",
        "en": "{teacher} only accepts messages from their Moodle contacts",
    },
    "teacher_reason_messaging_disabled": {
        "es": "la mensajería está desactivada en el aula virtual",
        "en": "messaging is turned off in the virtual classroom",
    },
    "teacher_reason_daily_limit": {
        "es": "ya enviaste varios mensajes hoy a través de mí",
        "en": "you have already sent several messages through me today",
    },
    "teacher_reason_other": {
        "es": "el aula virtual no lo aceptó en este momento",
        "en": "the virtual classroom did not accept it right now",
    },
    "failed": {
        "es": (
            "Lo siento, no pude consultar las fuentes en este momento. "
            "Inténtalo de nuevo en unos minutos."
        ),
        "en": "Sorry, I couldn't check the sources right now. Please try again in a few minutes.",
    },
    "references": {"es": "Referencias:", "en": "References:"},
    # The widget's waiting indicator while the answer is on its way.
    "working": {"es": "Consultando la biblioteca…", "en": "Checking the library…"},
    "page": {"es": "p.", "en": "p."},
}


def lang_of(state_or_lang) -> str:
    """Reply language: the tenant's (``language``), like every other agent.

    The signed Moodle language (``user_context.lms_lang``) is deliberately
    ignored: an admin browsing Moodle in English must still get the school's
    language. Anything but English — or nothing — is Spanish.
    """
    raw = state_or_lang if isinstance(state_or_lang, str) else state_or_lang.get("language")
    lang = (raw or "es").strip().lower()[:2]
    return lang if lang in ("es", "en") else "es"


def text(key: str, lang: str, **values: str) -> str:
    template = _TEXTS[key].get(lang) or _TEXTS[key]["es"]
    return template.format(**values) if values else template
