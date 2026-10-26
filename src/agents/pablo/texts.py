"""Fixed messages pablo sends outside the questions themselves.

They are code, not prompts: what the student reads while the timer is paused
must never depend on a model, and must never comment on the answer. The one
model-chosen piece is the acknowledgement's quote, which is the student's own
words checked literally against the answer (``echo.py``).
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
    "cycle_timeout": {
        "es": "Se acabó el tiempo de esta pregunta. Pulsa «Continuar» cuando estés listo.",
        "en": "Time is up for this question. Press “Continue” when you are ready.",
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
    # ── Opening (turn ``open``) ──
    "hello_name": {"es": "Hola, {name}. Soy {persona}.", "en": "Hi, {name}. I'm {persona}."},
    "hello": {"es": "Hola. Soy {persona}.", "en": "Hi. I'm {persona}."},
    "opening_intro": {
        "es": (
            "Esto es una conversación: quiero entender cómo lo piensas. "
            "Explícame tu razonamiento, no solo la respuesta."
        ),
        "en": (
            "This is a conversation: I want to understand how you think about it. "
            "Walk me through your reasoning, not just the answer."
        ),
    },
    "opening_topics": {"es": "Vamos a recorrer {n} temas: {topics}.", "en": "We will cover {n} topics: {topics}."},
    "opening_topic": {"es": "Vamos a hablar de {topics}.", "en": "We will talk about {topics}."},
    "opening_clock": {
        "es": "El tiempo solo corre mientras una pregunta está abierta.",
        "en": "The clock only runs while a question is open.",
    },
    "opening_warmup": {
        "es": (
            "Si quieres, antes hacemos una pregunta de calentamiento sin nota para probar "
            "el micrófono y el formato. Puedes saltarla."
        ),
        "en": (
            "If you like, we can start with an ungraded warm-up question to try the "
            "microphone and the format. You can skip it."
        ),
    },
    "opening_ready": {"es": "Cuando estés listo, empezamos.", "en": "Whenever you are ready, we begin."},
    # ── Warm-up ──
    "warmup_question": {
        "es": "Pregunta de calentamiento (sin nota):\n\n{prompt}",
        "en": "Warm-up question (not graded):\n\n{prompt}",
    },
    "warmup_audio_ok": {
        "es": (
            "Esto es lo que entendí de tu audio:\n«{transcript}»\n"
            "Si no se parece a lo que dijiste, revisa el micrófono antes de empezar."
        ),
        "en": (
            "This is what I understood from your audio:\n«{transcript}»\n"
            "If it does not match what you said, check your microphone before we begin."
        ),
    },
    "warmup_audio_failed": {
        "es": (
            "No pude entender tu audio. Revisa que el navegador tenga permiso para usar "
            "el micrófono y que no esté silenciado. Puedes repetir el calentamiento o "
            "responder por escrito."
        ),
        "en": (
            "I could not understand your audio. Check that the browser may use the "
            "microphone and that it is not muted. You can repeat the warm-up or answer in writing."
        ),
    },
    "warmup_done": {
        "es": "Así funciona. Cuando quieras, empezamos la evaluación.",
        "en": "That is how it works. Whenever you like, we start the assessment.",
    },
    # ── Acknowledgement ──
    "echo": {"es": "Anoté esta idea tuya: «{quote}».", "en": "I noted this idea of yours: “{quote}”."},
    "empty_answer": {
        "es": "El tiempo de esta pregunta terminó sin respuesta.",
        "en": "Time for this question ran out without an answer.",
    },
    # ── Follow-up question ──
    "followup_intro": {"es": "Antes de seguir, una pregunta más:", "en": "Before we move on, one more question:"},
    "followup_generic": {
        "es": "¿Puedes desarrollar un poco más tu respuesta a la {part}, con un ejemplo?",
        "en": "Can you develop your answer to {part} a little further, with an example?",
    },
    "followup_generic_plain": {
        "es": "¿Puedes desarrollar un poco más tu respuesta, con un ejemplo?",
        "en": "Can you develop your answer a little further, with an example?",
    },
    # ── Narrative progress (the cycle header) ──
    "progress_first": {
        "es": "Empezamos con el primer tema de {count}: {name}.",
        "en": "We start with the first topic of {count}: {name}.",
    },
    "progress_only": {"es": "Hablemos de {name}.", "en": "Let's talk about {name}."},
    "progress_next": {
        "es": "Seguimos con el {ordinal} tema de {count}: {name}.",
        "en": "Next, the {ordinal} topic of {count}: {name}.",
    },
    "progress_again": {
        "es": "Volvemos al tema «{name}» con otra pregunta.",
        "en": "Back to “{name}” with another question.",
    },
    "progress_last": {"es": "Es la última pregunta.", "en": "This is the last question."},
    # ── Closing ──
    "closing": {
        "es": "Con esto terminamos. Hablamos de {topics}. Gracias por tu tiempo.",
        "en": "That brings us to the end. We talked about {topics}. Thank you for your time.",
    },
    "closing_plain": {
        "es": "Con esto terminamos. Gracias por tu tiempo.",
        "en": "That brings us to the end. Thank you for your time.",
    },
    "survey_invite": {
        "es": "Antes de ver tu resultado, cuéntanos cómo fue tu experiencia.",
        "en": "Before you see your result, tell us how the experience was.",
    },
    "attempt_time_up": {
        "es": "Se acabó el tiempo de la evaluación.",
        "en": "Time is up for the assessment.",
    },
}

# Said after every answer that has content, chosen by the cycle's position and
# never by the answer: it values the effort, so it must read the same for a
# brilliant answer and a weak one.
_EFFORT: dict[str, tuple[str, ...]] = {
    "es": (
        "Gracias por desarrollarlo.",
        "Gracias por explicarlo con tus palabras.",
        "Gracias por tomarte el tiempo de pensarlo.",
        "Gracias por compartir tu razonamiento.",
    ),
    "en": (
        "Thank you for working it through.",
        "Thank you for explaining it in your own words.",
        "Thank you for taking the time to think it over.",
        "Thank you for sharing your reasoning.",
    ),
}

_ORDINALS = {
    "es": ("primer", "segundo", "tercer", "cuarto", "quinto", "sexto", "séptimo", "octavo", "noveno", "décimo"),
    "en": ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"),
}
_CARDINALS = {
    "es": ("uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez"),
    "en": ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"),
}
_AND = {"es": "y", "en": "and"}


def lang_of(state) -> str:
    """The activity's language (sent with every turn), else the tenant's."""
    state = state or {}
    domain = state.get("domain_state") or {}
    turn = domain.get("assessment_turn") if isinstance(domain, dict) else None
    lang = str((turn or {}).get("language") or state.get("language") or "es").lower()[:2]
    return "en" if lang == "en" else "es"


def text(key: str, lang: str, **values) -> str:
    template = _TEXTS[key].get(lang) or _TEXTS[key]["es"]
    return template.format(**values) if values else template


def effort(cycle_index: int, lang: str) -> str:
    phrases = _EFFORT.get(lang) or _EFFORT["es"]
    return phrases[max(cycle_index, 0) % len(phrases)]


def ordinal(n: int, lang: str) -> str:
    words = _ORDINALS.get(lang) or _ORDINALS["es"]
    return words[n - 1] if 1 <= n <= len(words) else f"{n}."


def cardinal(n: int, lang: str) -> str:
    words = _CARDINALS.get(lang) or _CARDINALS["es"]
    return words[n - 1] if 1 <= n <= len(words) else str(n)


def join_names(names: list[str], lang: str) -> str:
    """«la fe, la gracia y las obras»."""
    names = [n for n in names if n]
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} {_AND.get(lang, 'y')} {names[-1]}"


def part_label(index: int, lang: str) -> str:
    """"Parte A", "Parte B", … — the label the scoring prompt refers back to."""
    return f"{text('part', lang)} {chr(ord('A') + index)}"
