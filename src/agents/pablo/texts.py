"""Fixed messages pablo sends between cycles.

They are code, not prompts: what the student reads while the timer is paused
must never depend on a model, and must never comment on the answer.
"""

from __future__ import annotations

_TEXTS: dict[str, dict[str, str]] = {
    "received": {
        "es": "Recibí tu respuesta.",
        "en": "I received your answer.",
    },
    "transcript": {
        "es": "Esto es lo que entendí de tu audio, y es lo que se evaluará:\n«{transcript}»",
        "en": "This is what I understood from your audio, and it is what will be assessed:\n«{transcript}»",
    },
    "transcription_failed": {
        "es": (
            "No pude transcribir tu audio. Quedó guardado como respuesta de esta "
            "pregunta y tu profesor la revisará."
        ),
        "en": (
            "I could not transcribe your audio. It was kept as your answer to this "
            "question and your teacher will review it."
        ),
    },
    "continue": {
        "es": "El tiempo está en pausa. Pulsa «Continuar» cuando estés listo para la siguiente pregunta.",
        "en": "The timer is paused. Press “Continue” when you are ready for the next question.",
    },
    "last": {
        "es": "Esa fue la última pregunta. Antes de ver tu resultado, cuéntanos cómo fue tu experiencia.",
        "en": "That was the last question. Before you see your result, tell us how the experience was.",
    },
    "cycle_timeout": {
        "es": "Se acabó el tiempo de esta pregunta. Pulsa «Continuar» cuando estés listo.",
        "en": "Time is up for this question. Press “Continue” when you are ready.",
    },
    "attempt_timeout": {
        "es": "Se acabó el tiempo de la evaluación. Antes de ver tu resultado, cuéntanos cómo fue tu experiencia.",
        "en": "Time is up for the assessment. Before you see your result, tell us how the experience was.",
    },
    "compose_failed": {
        "es": "No pude preparar la siguiente pregunta. El tiempo sigue en pausa: pulsa «Continuar» para intentarlo de nuevo.",
        "en": "I could not prepare the next question. The timer is still paused: press “Continue” to try again.",
    },
    "answer_failed": {
        "es": "No pude registrar tu respuesta. El tiempo está en pausa: vuelve a enviarla.",
        "en": "I could not record your answer. The timer is paused: please send it again.",
    },
    "unexpected": {
        "es": "Esta evaluación ya no espera ese mensaje. Sigue las indicaciones de la actividad.",
        "en": "This assessment is not expecting that message. Follow the activity's instructions.",
    },
    "part": {"es": "Parte", "en": "Part"},
    "question": {"es": "Pregunta {n} de {total}", "en": "Question {n} of {total}"},
}


def lang_of(state) -> str:
    lang = str((state or {}).get("language") or "es").lower()[:2]
    return "en" if lang == "en" else "es"


def text(key: str, lang: str, **values) -> str:
    template = _TEXTS[key].get(lang) or _TEXTS[key]["es"]
    return template.format(**values) if values else template


def part_label(index: int, lang: str) -> str:
    """"Parte A", "Parte B", … — the label the scoring prompt refers back to."""
    return f"{text('part', lang)} {chr(ord('A') + index)}"
