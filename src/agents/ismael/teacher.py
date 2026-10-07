"""ismael's "contact my teacher" flow: help first, then send or suggest.

A small state machine in ``state["ismael"]["teacher"]``; while ``step`` is set
triage routes straight here without an LLM call, like the survey latch.

    (new)         offer help first ("¿puedo ayudarte yo?")
    offer_help    yes  → answer it (theology → Brain, Moodle → support) or ask what
                  no   → look up the student's courses and teachers
    choose_mode   "Envíalo tú" (ismael sends) / "Le escribo yo" (link only);
                  skipped when the site has messaging off — then the teacher's
                  email (asked of Moodle, which checks it is a teacher of the
                  student's course who shows it to participants) is the answer
    pick_course   only when there is no current course and several enrolments
    pick_teacher  only when the course lists several teachers
    compose       "¿Qué quieres decirle?" (skipped when the reason is known)
    confirm       draft + Enviar / Cambiar algo / Cancelar
    edit          what to change → new draft → confirm

The message is SENT only on an explicit "Enviar", through the backend, which
signs a command for the Moodle plugin; the teacher receives it in Moodle from
the student. When the connection or plugin cannot send, choose_mode is
skipped and the student gets the link to write themselves.

A reply that is a different question altogether abandons the flow and is
re-triaged in the same turn (``ismael_route = "retriage"``); the flow is
cleared first, so this can never loop.
"""

from __future__ import annotations

import hashlib
import time

import structlog
from langchain_core.runnables import RunnableConfig

from ...config import settings
from ...graphs.state import AgentState
from ...observability import record_node_invocation
from ...providers.registry import get_provider, resolve_background_model
from ...schemas.ismael import TeacherDraft, TeacherStepAnswer, TeacherTopic
from ...usage import make_usage_record
from .. import backend_client
from ..utils import latest_user_text
from .common import bounded_structured, ask_options, fold, ismael_dict, match_option, reply, user_ctx
from .prompts import STEP_PROMPT, TEACHER_DRAFT_PROMPT
from .texts import lang_of, text

logger = structlog.get_logger(__name__)

# A flow left half-way is forgotten after this long: the next message is a
# fresh turn, not an answer to a question asked yesterday.
FLOW_TTL_SECONDS = 30 * 60
MAX_COURSE_OPTIONS = 8

# Option labels: (es, en). The number a user types is the position.
OFFER = (("Sí, ayúdame", "Yes, help me"), ("No, quiero contactar al profesor", "No, I want to contact my teacher"))
MODE = (("Envíalo tú", "You send it"), ("Le escribo yo", "I'll write myself"))
CONFIRM = (("Enviar", "Send"), ("Cambiar algo", "Change something"), ("Cancelar", "Cancel"))

# A bare yes/no to the offer ("¿puedo ayudarte yo?") — the commonest replies,
# answered without an LLM. Compared after accent/case folding.
_YES = {"si", "sí", "claro", "dale", "ok", "vale", "bueno", "por favor", "yes", "sure", "please"}
_NO = {"no", "nop", "no gracias", "no, gracias", "no thanks", "no, thanks", "nope"}


def new_flow(reason: str, topic: TeacherTopic | str, faq: dict | None = None) -> dict:
    flow = {
        "step": "",
        "reason": (reason or "").strip(),
        "topic": str(getattr(topic, "value", topic) or TeacherTopic.NONE.value),
        "touched_at": time.time(),
    }
    if faq:
        # An institution FAQ answers the reason: that is the help offered first.
        flow["faq"] = faq
    return flow


def flow_open(ismael: dict) -> bool:
    flow = ismael.get("teacher")
    if not isinstance(flow, dict) or not flow.get("step"):
        return False
    return time.time() - float(flow.get("touched_at") or 0) < FLOW_TTL_SECONDS


# The backend keeps quick-reply buttons only for labels of at most 60
# characters; a longer course name is shortened for display AND for matching,
# so the label a button sends back is the one compared.
MAX_LABEL = 60


def _short(label: str) -> str:
    label = " ".join((label or "").split())
    return label if len(label) <= MAX_LABEL else label[: MAX_LABEL - 1].rstrip() + "…"


def _labels(pairs, lang: str) -> list[str]:
    return [pair[1] if lang == "en" else pair[0] for pair in pairs]


def _language_rule(lang: str) -> str:
    return "Always respond in English." if lang == "en" else "Responde siempre en español."


# ── Turn plumbing ────────────────────────────────────────────────────────────


class _Turn:
    """One teacher-flow turn: the flow dict, the reply language and usage."""

    def __init__(self, state: AgentState, config: RunnableConfig):
        self.state = state
        self.config = config
        self.ismael = ismael_dict(state)
        self.flow = dict(self.ismael.get("teacher") or {})
        self.lang = lang_of(state)
        self.text = latest_user_text(state)
        self.usage: list = []

    def ask(self, step: str, question: str, labels: list[str]) -> dict:
        self.flow["step"] = step
        self.flow["touched_at"] = time.time()
        self.ismael["teacher"] = self.flow
        return ask_options(
            question, labels, ismael=self.ismael, turn_usage=self.usage, ismael_route="done"
        )

    def prompt(self, step: str, message: str) -> dict:
        """An open question (no options)."""
        self.flow["step"] = step
        self.flow["touched_at"] = time.time()
        self.ismael["teacher"] = self.flow
        return reply(message, ismael=self.ismael, turn_usage=self.usage, ismael_route="done")

    def finish(self, message: str) -> dict:
        self.ismael.pop("teacher", None)
        return reply(message, ismael=self.ismael, turn_usage=self.usage, ismael_route="done")

    def hand_off(self, route: str) -> dict:
        """Leave the flow and let another node answer, in this same turn."""
        self.ismael.pop("teacher", None)
        return {"ismael": self.ismael, "turn_usage": self.usage, "ismael_route": route}

    async def choose(self, question: str, labels: list[str]) -> tuple[int | None, bool]:
        """(0-based option, is_new_question). Deterministic first, LLM otherwise."""
        index = match_option(self.text, [[label] for label in labels])
        if index is not None:
            return index, False
        if not self.text.strip():
            return None, False
        options = "\n".join(f"{n}) {label}" for n, label in enumerate(labels, start=1))
        system = STEP_PROMPT.format(question=question, options=options)
        try:
            provider = get_provider(self.config)
            model = resolve_background_model(self.config)
            result = await bounded_structured(
                provider,
                [{"role": "system", "content": system}, {"role": "user", "content": self.text}],
                model,
                TeacherStepAnswer,
                timeout=settings.ismael_timeout_step_seconds,
                thinking_budget=settings.ismael_thinking_classify,
            )
            self.usage.append(
                make_usage_record(node="ismael_teacher_step", provider=provider, model=model)
            )
        except Exception as exc:  # noqa: BLE001 — unclear is re-asked, never guessed
            logger.info("ismael_teacher_step_failed", error=str(exc))
            return None, False
        if 1 <= result.choice <= len(labels):
            return result.choice - 1, False
        return None, result.is_new_question


async def _choose_pair(turn: _Turn, question: str, pairs) -> tuple[int | None, bool]:
    # Exact/number/substring over both languages, then the LLM on this language.
    index = match_option(turn.text, [list(pair) for pair in pairs])
    if index is not None:
        return index, False
    return await turn.choose(question, _labels(pairs, turn.lang))


# ── Steps ────────────────────────────────────────────────────────────────────


def _offer(turn: _Turn) -> dict:
    topic = turn.flow.get("topic")
    key = {
        TeacherTopic.THEOLOGY.value: "teacher_offer_theology",
        TeacherTopic.MOODLE_SUPPORT.value: "teacher_offer_moodle",
    }.get(topic if turn.flow.get("reason") else "", "teacher_offer")
    if turn.flow.get("faq"):
        key = "teacher_offer_faq"
    return turn.ask("offer_help", text(key, turn.lang), _labels(OFFER, turn.lang))


async def _on_offer(turn: _Turn) -> dict:
    question = text("teacher_offer", turn.lang)
    bare = fold(turn.text).strip(" .!¡¿?")
    if bare in _YES or bare in _NO:
        index, new_question = (0 if bare in _YES else 1), False
    else:
        index, new_question = await _choose_pair(turn, question, OFFER)
    if new_question:
        return turn.hand_off("retriage")
    if index is None:
        return _offer(turn)
    if index == 0:
        reason = turn.flow.get("reason") or ""
        topic = turn.flow.get("topic")
        if turn.flow.get("faq"):
            turn.ismael["faq"] = turn.flow["faq"]
            turn.ismael["faq_question"] = reason
            return turn.hand_off("faq")
        if reason and topic == TeacherTopic.THEOLOGY.value:
            turn.ismael["question"] = reason
            return turn.hand_off("start")
        if reason and topic == TeacherTopic.MOODLE_SUPPORT.value:
            turn.ismael["support_question"] = reason
            return turn.hand_off("moodle_support")
        return turn.finish(text("teacher_tell_me", turn.lang))
    return await _begin_contact(turn)


async def _begin_contact(turn: _Turn) -> dict:
    conversation_id = turn.state.get("conversation_id") or ""
    try:
        data = await backend_client.get_moodle_teachers(conversation_id)
    except Exception as exc:  # noqa: BLE001 — the student still gets a way forward
        logger.warning("ismael_teacher_lookup_failed", error=str(exc))
        return turn.finish(text("teacher_lookup_failed", turn.lang))
    courses = [c for c in (data.get("courses") or []) if isinstance(c, dict)]
    if not data.get("available") or not courses:
        return turn.finish(text("teacher_no_courses", turn.lang))
    turn.flow["data"] = {
        "messaging_enabled": bool(data.get("messagingEnabled")),
        "current_course_id": data.get("currentCourseId"),
        "courses": courses,
    }
    if turn.flow["data"]["messaging_enabled"]:
        return turn.ask(
            "choose_mode", text("teacher_choose_mode", turn.lang), _labels(MODE, turn.lang)
        )
    turn.flow["mode"] = "self"
    return await _resolve_course(turn)


async def _on_mode(turn: _Turn) -> dict:
    question = text("teacher_choose_mode", turn.lang)
    index, new_question = await _choose_pair(turn, question, MODE)
    if new_question:
        return turn.hand_off("retriage")
    if index is None:
        return turn.ask("choose_mode", question, _labels(MODE, turn.lang))
    turn.flow["mode"] = "send" if index == 0 else "self"
    return await _resolve_course(turn)


def _course_options(turn: _Turn) -> list[dict]:
    return (turn.flow.get("data") or {}).get("courses", [])[:MAX_COURSE_OPTIONS]


async def _resolve_course(turn: _Turn) -> dict:
    data = turn.flow.get("data") or {}
    courses = data.get("courses") or []
    current = data.get("current_course_id")
    course = next((c for c in courses if c.get("id") == current), None) if current else None
    if course is None and len(courses) == 1:
        course = courses[0]
    if course is None:
        return turn.ask(
            "pick_course",
            text("teacher_pick_course", turn.lang),
            [_short(c.get("fullname") or str(c.get("id"))) for c in _course_options(turn)],
        )
    return await _resolve_teacher(turn, course)


async def _on_course(turn: _Turn) -> dict:
    options = _course_options(turn)
    labels = [_short(c.get("fullname") or str(c.get("id"))) for c in options]
    question = text("teacher_pick_course", turn.lang)
    index, new_question = await turn.choose(question, labels)
    if new_question:
        return turn.hand_off("retriage")
    if index is None:
        return turn.ask("pick_course", question, labels)
    return await _resolve_teacher(turn, options[index])


async def _resolve_teacher(turn: _Turn, course: dict) -> dict:
    turn.flow["course"] = {
        "id": course.get("id"),
        "fullname": course.get("fullname") or "",
        "teachers": course.get("teachers") or [],
    }
    teachers = turn.flow["course"]["teachers"]
    if not teachers:
        return turn.finish(
            text(
                "teacher_none",
                turn.lang,
                course=turn.flow["course"]["fullname"],
                url=course.get("participantsUrl") or "",
            )
        )
    if len(teachers) == 1:
        return await _after_teacher(turn, teachers[0])
    return turn.ask(
        "pick_teacher",
        text("teacher_pick_teacher", turn.lang, course=turn.flow["course"]["fullname"]),
        [_short(t.get("name") or "") for t in teachers],
    )


async def _on_teacher(turn: _Turn) -> dict:
    course = turn.flow.get("course") or {}
    teachers = course.get("teachers") or []
    labels = [_short(t.get("name") or "") for t in teachers]
    question = text("teacher_pick_teacher", turn.lang, course=course.get("fullname") or "")
    index, new_question = await turn.choose(question, labels)
    if new_question:
        return turn.hand_off("retriage")
    if index is None:
        return turn.ask("pick_teacher", question, labels)
    return await _after_teacher(turn, teachers[index])


async def _after_teacher(turn: _Turn, teacher: dict) -> dict:
    turn.flow["teacher"] = {
        "id": str(teacher.get("id") or ""),
        "name": teacher.get("name") or "",
        "messageUrl": teacher.get("messageUrl") or "",
    }
    name = turn.flow["teacher"]["name"]
    if turn.flow.get("mode") != "send":
        if (turn.flow.get("data") or {}).get("messaging_enabled"):
            # They chose to write themselves, and Moodle messaging works.
            return turn.finish(
                text("teacher_self_link", turn.lang, teacher=name, url=turn.flow["teacher"]["messageUrl"])
            )
        # Messaging is off on this site: the teacher's email is the way.
        return await _give_email(turn)
    reason = turn.flow.get("reason") or ""
    if reason:
        # They already said what it is about: go straight to a draft.
        turn.flow["content"] = reason
        return _confirm(turn, await _draft(turn, reason))
    return turn.prompt("compose", text("teacher_compose", turn.lang, teacher=name))


def _institution_fallback(turn: _Turn) -> str:
    """Where to turn when the teacher's email cannot be given."""
    ctx = user_ctx(turn.state)
    if ctx.get("lms_other_contacts"):
        return text("teacher_fallback_contacts", turn.lang, contacts=ctx["lms_other_contacts"])
    if ctx.get("lms_support_url"):
        return text("teacher_fallback_support", turn.lang, url=ctx["lms_support_url"])
    return text("teacher_fallback_generic", turn.lang)


async def _give_email(turn: _Turn) -> dict:
    """The teacher's email, from Moodle (which re-checks that it is a teacher
    of the student's course who shows their email to participants)."""
    teacher = turn.flow.get("teacher") or {}
    course = turn.flow.get("course") or {}
    name = teacher.get("name") or ""
    try:
        result = await backend_client.get_teacher_email(
            turn.state.get("conversation_id") or "",
            teacher_id=str(teacher.get("id") or ""),
            course_id=int(course.get("id") or 0),
        )
    except Exception as exc:  # noqa: BLE001 — the institution is still a way forward
        logger.warning("ismael_teacher_email_failed", error=str(exc) or type(exc).__name__)
        result = {"status": "FAILED", "code": "unreachable"}
    logger.info("ismael_teacher_email", status=result.get("status"), code=result.get("code"))
    email = result.get("email")
    if result.get("status") == "OK" and email:
        return turn.finish(text("teacher_email_given", turn.lang, teacher=name, email=email))
    key = "teacher_email_hidden" if result.get("code") == "email_hidden" else "teacher_email_failed"
    return turn.finish(f"{text(key, turn.lang, teacher=name)} {_institution_fallback(turn)}")


async def _draft(turn: _Turn, content: str, change: str = "") -> str:
    """The message in the student's voice; on failure, their own words."""
    ctx = user_ctx(turn.state)
    teacher = turn.flow.get("teacher") or {}
    course = turn.flow.get("course") or {}
    lines = [
        f"Estudiante: {ctx.get('name') or '(sin nombre)'}",
        f"Profesor: {teacher.get('name') or ''}",
        f"Curso: {course.get('fullname') or '(desconocido)'}",
        f"Lo que el estudiante quiere decir: {content}",
    ]
    previous = turn.flow.get("draft") or ""
    if change and previous:
        lines += [f"Borrador anterior: {previous}", f"Cambio pedido: {change}"]
    system = TEACHER_DRAFT_PROMPT.format(language_rule=_language_rule(turn.lang))
    try:
        provider = get_provider(turn.config)
        model = resolve_background_model(turn.config)
        result = await bounded_structured(
            provider,
            [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}],
            model,
            TeacherDraft,
            timeout=settings.ismael_timeout_answer_seconds,
            thinking_budget=settings.ismael_thinking_answer,
        )
        turn.usage.append(make_usage_record(node="ismael_teacher_draft", provider=provider, model=model))
        message = result.message.strip()
        if message:
            return message
    except Exception as exc:  # noqa: BLE001 — the student's own words still work
        logger.info("ismael_teacher_draft_failed", error=str(exc))
    return previous if change and previous else content.strip()


def _confirm(turn: _Turn, draft: str) -> dict:
    turn.flow["draft"] = draft
    name = (turn.flow.get("teacher") or {}).get("name") or ""
    return turn.ask(
        "confirm",
        text("teacher_confirm", turn.lang, teacher=name, draft=draft),
        _labels(CONFIRM, turn.lang),
    )


async def _on_compose(turn: _Turn) -> dict:
    content = turn.text.strip()
    if not content:
        name = (turn.flow.get("teacher") or {}).get("name") or ""
        return turn.prompt("compose", text("teacher_compose", turn.lang, teacher=name))
    turn.flow["content"] = content
    return _confirm(turn, await _draft(turn, content))


async def _on_confirm(turn: _Turn) -> dict:
    question = text(
        "teacher_confirm",
        turn.lang,
        teacher=(turn.flow.get("teacher") or {}).get("name") or "",
        draft=turn.flow.get("draft") or "",
    )
    index, new_question = await _choose_pair(turn, question, CONFIRM)
    if new_question:
        return turn.hand_off("retriage")
    if index == 0:
        return await _send(turn)
    if index == 1:
        return turn.prompt("edit", text("teacher_edit", turn.lang))
    if index == 2:
        return turn.finish(text("teacher_cancelled", turn.lang))
    # Free text that is not an option: read it as the change to make.
    return await _on_edit(turn)


async def _on_edit(turn: _Turn) -> dict:
    change = turn.text.strip()
    if not change:
        return turn.prompt("edit", text("teacher_edit", turn.lang))
    draft = await _draft(turn, turn.flow.get("content") or "", change=change)
    return _confirm(turn, draft)


async def _send(turn: _Turn) -> dict:
    teacher = turn.flow.get("teacher") or {}
    course = turn.flow.get("course") or {}
    draft = turn.flow.get("draft") or ""
    conversation_id = turn.state.get("conversation_id") or ""
    digest = hashlib.sha256(f"{teacher.get('id')}|{course.get('id')}|{draft}".encode()).hexdigest()
    key = f"{conversation_id}:{digest[:40]}"
    name = teacher.get("name") or ""
    url = teacher.get("messageUrl") or ""
    try:
        result = await backend_client.post_teacher_message(
            conversation_id,
            teacher_id=str(teacher.get("id") or ""),
            course_id=int(course.get("id") or 0),
            text=draft,
            idempotency_key=key,
            language=turn.lang,
        )
    except Exception as exc:  # noqa: BLE001 — the link always works
        logger.warning("ismael_teacher_send_failed", error=str(exc))
        result = {"status": "FAILED", "code": "unreachable"}
    status = result.get("status")
    url = result.get("messageUrl") or url
    logger.info("ismael_teacher_message", status=status, code=result.get("code"))
    if status == "SENT":
        return turn.finish(text("teacher_sent", turn.lang, teacher=name, url=url))
    reason_key = {
        "recipient_blocks": "teacher_reason_recipient_blocks",
        "messaging_disabled": "teacher_reason_messaging_disabled",
        "daily_limit": "teacher_reason_daily_limit",
    }.get(str(result.get("code") or ""), "teacher_reason_other")
    reason = text(reason_key, turn.lang, teacher=name)
    return turn.finish(text("teacher_send_failed", turn.lang, reason=reason, teacher=name, url=url))


_HANDLERS = {
    "offer_help": _on_offer,
    "choose_mode": _on_mode,
    "pick_course": _on_course,
    "pick_teacher": _on_teacher,
    "compose": _on_compose,
    "confirm": _on_confirm,
    "edit": _on_edit,
}


async def ismael_teacher_node(state: AgentState, config: RunnableConfig) -> dict:
    record_node_invocation("ismael_teacher")
    turn = _Turn(state, config)
    handler = _HANDLERS.get(str(turn.flow.get("step") or ""))
    if handler is None:
        return _offer(turn)
    return await handler(turn)
