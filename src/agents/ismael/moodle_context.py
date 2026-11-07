"""What Moodle says about the student's own classroom, for the support branch.

Before plugin 0.6.0 the assistant knew the student's courses and nothing else:
«presenté una evaluación pero no aparece la nota» (2026-11-05) could only be
answered with "ask your teacher". Now, when the conversation's contact comes
from a plugin that signs the page's activity, support reads — through the
backend, which signs a read order to the plugin — the student's own state:

- the activity the chat was opened on (``lms_current_activity_*``, in the
  identity token): name, type, dates, description;
- their state in it (``GET …/moodle/activity-status``): quiz attempts and
  whether marks show yet or only after the close, a submission and its word
  limit, whether they can post in a forum, whether the grade is hidden;
- the course's sections and activities as they see them, the current one
  marked (``GET …/moodle/course-outline``) — and the state of an activity the
  student names, when exactly one in the outline matches.

Only that student's data, as Moodle shows it to them; it is never stored. The
block goes into the support prompt fenced ``<<<MOODLE … MOODLE>>>``. Any
failure (older plugin, timeout, refusal) leaves the block out: support answers
as it did before.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from .. import backend_client
from .common import fold, user_ctx

logger = structlog.get_logger(__name__)

# One read per conversation and target within this window: a support case is
# a handful of turns, and the student's state rarely changes between them.
CACHE_SECONDS = 120.0
READ_TIMEOUT_SECONDS = 6.0
MAX_OUTLINE_CHARS = 6000
MAX_INTRO_CHARS = 800
_cache: dict[tuple[str, str], tuple[float, dict | None]] = {}

RULES = (
    "\n\nCómo usar los datos del aula: son hechos de Moodle sobre este estudiante, "
    "leídos ahora. Responde con ellos (fechas, estado del intento o de la entrega, límite "
    "de palabras, en qué sección está una actividad) antes de pensar en el profesor o en "
    "un ticket. Si el intento está finalizado y pendiente de calificación, la nota aparece "
    "cuando el profesor califique las preguntas abiertas; si la nota se muestra al cierre, "
    "dile la fecha de cierre; si la nota está oculta, el profesor aún no la publica. Nunca "
    "inventes lo que no aparece en los datos."
)


def _int(value: object) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


async def _cached(key: tuple[str, str], fetch) -> dict | None:
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and hit[0] > now:
        return hit[1]
    try:
        result = await asyncio.wait_for(fetch(), timeout=READ_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 — support answers without it
        logger.info("ismael_moodle_read_failed", kind=key[1].split(":")[0], error=str(exc) or type(exc).__name__)
        return None
    data = result.get("data") if isinstance(result, dict) and result.get("status") == "OK" else None
    if data is None:
        logger.info("ismael_moodle_read_refused", kind=key[1].split(":")[0], code=(result or {}).get("code"))
    if len(_cache) > 512:
        _cache.clear()
    _cache[key] = (now + CACHE_SECONDS, data if isinstance(data, dict) else None)
    return _cache[key][1]


async def _status(conversation_id: str, course_id: int, cmid: int) -> dict | None:
    return await _cached(
        (conversation_id, f"status:{course_id}:{cmid}"),
        lambda: backend_client.get_moodle_activity_status(conversation_id, course_id=course_id, cmid=cmid),
    )


async def _outline(conversation_id: str, course_id: int) -> dict | None:
    return await _cached(
        (conversation_id, f"outline:{course_id}"),
        lambda: backend_client.get_moodle_course_outline(conversation_id, course_id=course_id),
    )


def _activities(outline: dict) -> list[dict]:
    out = []
    for section in outline.get("sections") or []:
        if not isinstance(section, dict):
            continue
        for act in section.get("activities") or []:
            if isinstance(act, dict) and act.get("cmid"):
                out.append(act)
    return out


def _words(value: str) -> list[str]:
    return [w for w in "".join(c if c.isalnum() else " " for c in fold(value)).split() if len(w) >= 4]


def match_activity(outline: dict | None, texts: list[str]) -> dict | None:
    """The one activity of the outline the student names, or None.

    An activity matches when its whole name appears in the text, or when every
    word of four letters or more of its name does («trabajo final individual»,
    «cuestionario 2»). Two or more matches name nothing in particular.
    """
    if not outline:
        return None
    haystack = " ".join(fold(t) for t in texts if t)
    words = set(_words(haystack))
    matches = []
    for act in _activities(outline):
        name = fold(str(act.get("name") or ""))
        significant = _words(name)
        if not name:
            continue
        if name in haystack or (significant and all(w in words for w in significant)):
            matches.append(act)
    return matches[0] if len(matches) == 1 else None


def _dates(dates: object) -> str:
    parts = [
        f"{d.get('label')}: {d.get('text')}"
        for d in dates or []
        if isinstance(d, dict) and d.get("label") and d.get("text")
    ]
    return "; ".join(parts)


_GRADE = {
    "graded": "nota publicada: {text}",
    "hidden": "nota oculta (el profesor aún no la publica)",
    "none": "sin nota todavía",
}
_SUBMISSION = {
    "none": "no ha entregado",
    "new": "no ha entregado",
    "draft": "borrador guardado, NO enviado",
    "submitted": "entregada",
    "reopened": "reabierta para un nuevo envío",
}


def render_status(status: dict) -> str:
    """One activity's state for the student, in plain Spanish lines."""
    lines = [f"Estado del estudiante en «{status.get('name', '')}» ({status.get('type', '')}):"]
    if when := _dates(status.get("dates")):
        lines.append(f"- Fechas: {when}")
    grade = status.get("grade") if isinstance(status.get("grade"), dict) else {}
    if grade.get("state") in _GRADE:
        lines.append("- " + _GRADE[grade["state"]].format(text=grade.get("text") or ""))
    quiz = status.get("quiz") if isinstance(status.get("quiz"), dict) else None
    if quiz:
        allowed = quiz.get("attempts_allowed") or 0
        lines.append(
            f"- Intentos: {quiz.get('attempts_used', 0)} usados de "
            + (str(allowed) if allowed else "ilimitados")
            + (f"; tiempo límite {quiz['time_limit_minutes']} min" if quiz.get("time_limit_minutes") else "")
        )
        for attempt in quiz.get("attempts") or []:
            state = {
                "finished": "finalizado",
                "inprogress": "en curso (sin enviar)",
                "overdue": "vencido sin enviar (se acabó el tiempo)",
                "abandoned": "abandonado (nunca se envió)",
            }.get(str(attempt.get("state")), str(attempt.get("state")))
            line = f"- Intento {attempt.get('number')}: {state}"
            if attempt.get("finished"):
                line += f" el {attempt['finished']}"
            if attempt.get("needs_grading"):
                line += "; pendiente de que el profesor califique las preguntas abiertas"
            elif attempt.get("grade"):
                line += f"; calificación {attempt['grade']}"
            elif attempt.get("state") == "finished" and not attempt.get("marks_visible"):
                line += (
                    "; la nota se muestra cuando cierre el cuestionario"
                    if quiz.get("marks_after_close")
                    else "; el cuestionario no muestra la nota al estudiante"
                )
            lines.append(line)
    assign = status.get("assign") if isinstance(status.get("assign"), dict) else None
    if assign:
        sub = _SUBMISSION.get(str(assign.get("submission")), str(assign.get("submission")))
        lines.append(f"- Entrega: {sub}" + (f" el {assign['submitted']}" if assign.get("submitted") else ""))
        if assign.get("word_limit"):
            lines.append(f"- Límite de palabras del texto en línea: {assign['word_limit']}")
        files = assign.get("files") if isinstance(assign.get("files"), dict) else None
        if files:
            size = files.get("max_bytes") or 0
            lines.append(
                f"- Archivos: hasta {files.get('max_files') or 1}"
                + (f", de máximo {round(size / 1048576, 1)} MB" if size else "")
                + (f", tipos {files['types']}" if files.get("types") else "")
            )
    forum = status.get("forum") if isinstance(status.get("forum"), dict) else None
    if forum:
        lines.append(
            "- Foro: "
            + ("puede abrir temas" if forum.get("can_start_discussion") else "no puede abrir temas")
            + ", "
            + ("puede responder" if forum.get("can_reply") else "NO puede responder")
            + ", "
            + ("suscrito" if forum.get("subscribed") else "no suscrito (no impide publicar)")
            + f"; sus publicaciones en este foro: {forum.get('my_posts', 0)}"
            + (f" (la última el {forum['my_last_post']})" if forum.get("my_last_post") else "")
        )
    return "\n".join(lines)


def render_outline(outline: dict) -> str:
    course = outline.get("course") if isinstance(outline.get("course"), dict) else {}
    lines = [f"Curso «{course.get('fullname', '')}», secciones que ve el estudiante:"]
    for section in outline.get("sections") or []:
        if not isinstance(section, dict):
            continue
        acts = []
        for act in section.get("activities") or []:
            detail = [str(act.get("type") or "")]
            if when := _dates(act.get("dates")):
                detail.append(when)
            if act.get("grade") in _GRADE:
                detail.append(_GRADE[act["grade"]].format(text="").rstrip(": "))
            acts.append(f"«{act.get('name', '')}» ({'; '.join(d for d in detail if d)})")
        marker = " [sección actual]" if section.get("current") else ""
        lines.append(f"- {section.get('name', '')}{marker}: " + (", ".join(acts) if acts else "sin actividades"))
    text = "\n".join(lines)
    return text if len(text) <= MAX_OUTLINE_CHARS else text[: MAX_OUTLINE_CHARS - 1] + "…"


async def classroom_block(state, texts: list[str]) -> str:
    """The ``<<<MOODLE … MOODLE>>>`` block for the support prompt, or "".

    ``texts`` are what the student said in this case (to find an activity they
    name in the outline)."""
    ctx = user_ctx(state)
    conversation_id = state.get("conversation_id") or ""
    course_id = _int(ctx.get("lms_current_course_id"))
    cmid = _int(ctx.get("lms_current_activity_cmid"))
    if not conversation_id or not course_id:
        return ""

    reads = [_outline(conversation_id, course_id)]
    if cmid:
        reads.append(_status(conversation_id, course_id, cmid))
    results = await asyncio.gather(*reads)
    outline = results[0]
    status = results[1] if cmid else None
    named = match_activity(outline, texts)
    named_status = None
    if named and _int(named.get("cmid")) != cmid:
        named_status = await _status(conversation_id, course_id, _int(named.get("cmid")))

    parts = []
    if ctx.get("lms_current_activity"):
        page = f"Actividad abierta en la página: «{ctx['lms_current_activity']}»"
        if ctx.get("lms_current_activity_type"):
            page += f" ({ctx['lms_current_activity_type']})"
        if str(ctx.get("lms_current_activity_is_pablo") or "") == "true":
            page += ", la evaluación conversacional con Pablo (actividad de Vervux, no de Moodle)"
        if ctx.get("lms_current_activity_dates"):
            page += f". Fechas: {ctx['lms_current_activity_dates']}"
        if ctx.get("lms_current_activity_intro"):
            page += f". Descripción: {str(ctx['lms_current_activity_intro'])[:MAX_INTRO_CHARS]}"
        parts.append(page)
    for item in (status, named_status):
        if item:
            parts.append(render_status(item))
    if outline:
        parts.append(render_outline(outline))
    logger.info(
        "ismael_moodle_context",
        page=bool(cmid),
        status=bool(status),
        outline=bool(outline),
        named=bool(named_status),
    )
    if not parts:
        return ""
    body = "\n\n".join(parts).replace("MOODLE>>>", "")
    return f"\n\nDatos del aula de este estudiante:\n<<<MOODLE\n{body}\nMOODLE>>>{RULES}"
