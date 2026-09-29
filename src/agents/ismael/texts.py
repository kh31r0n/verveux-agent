"""Fixed wording for ismael: the survey, the holding replies and the fallbacks.

Deterministic on purpose. These replies are sent while Brain is booting or
answering, so they must not depend on an LLM call succeeding, and the survey
options are statistics whose keys must never drift.

Spanish is the default; English covers a Moodle site running in English (the
widget and ``language`` follow the site).
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
            "Soy {persona} y solo respondo preguntas de teología cristiana. "
            "¿Qué te gustaría preguntar?"
        ),
        "en": (
            "I'm {persona} and I only answer questions about Christian theology. "
            "What would you like to ask?"
        ),
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
    "general_notice": {
        "es": "(Respuesta general: la biblioteca no tiene fuentes sobre esta pregunta.)",
        "en": "(General answer: the library has no sources on this question.)",
    },
    "page": {"es": "p.", "en": "p."},
}


def lang_of(state_or_lang) -> str:
    """Reply language: the user's signed Moodle language first (``lms_lang``),
    then the tenant's. TenantSettings.language defaults to "en" whatever the
    school speaks, and ismael only ever talks inside Moodle."""
    if isinstance(state_or_lang, str):
        raw = state_or_lang
    else:
        ctx = state_or_lang.get("user_context") or {}
        raw = (ctx.get("lms_lang") if isinstance(ctx, dict) else None) or state_or_lang.get(
            "language"
        )
    lang = (raw or "es").strip().lower()[:2]
    return lang if lang in ("es", "en") else "es"


def text(key: str, lang: str, **values: str) -> str:
    template = _TEXTS[key].get(lang) or _TEXTS[key]["es"]
    return template.format(**values) if values else template
