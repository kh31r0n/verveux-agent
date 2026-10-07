"""ismael — Christian theology Q&A for the Moodle bubble, answered by Brain.

Every node here replies within the turn; the answer itself arrives later from
the background job (`rag_job`). State lives in one dict, ``state["ismael"]``:

    question          the current turn's question, made standalone by triage
    pending_question  the question the survey is holding / the job is answering
    survey_step       0 not started · 1-3 waiting for answer n · 4 done
    survey_answers    {level, topic, intendedUse} → option key or "no_answer"
    teacher           the "contact my teacher" flow (see teacher.py); its
                      ``step`` latches triage like ``survey_step`` does

The survey is asked once per contact: the backend reports
``user_context.ismael_survey_done`` once the answers are stored.
"""

from __future__ import annotations

import structlog
from langchain_core.runnables import RunnableConfig

from ...config import settings
from ...graphs.state import AgentState
from ...observability import record_node_invocation
from ...providers.registry import get_provider, resolve_background_model
from ...schemas.ismael import IsmaelIntent, SurveyAnswer, TriageResult
from ...usage import make_usage_record
from .. import backend_client
from ..utils import (
    emit_pending_followup,
    emit_quick_replies,
    format_user_context,
    latest_user_messages,
    latest_user_text,
    resolve_persona,
    resolve_prompt,
)
from . import rag_job, teacher
from .faq import faq_candidates, faq_candidates_block, pick_faq
from .common import (
    DEFAULT_PERSONA,
    bounded_structured,
    fallback_intent,
    history_messages,
    match_option,
)
from .common import ismael_dict as _ismael
from .common import reply as _reply
from .prompts import SURVEY_PROMPT, TRIAGE_PROMPT
from .texts import NO_ANSWER, SURVEY, SurveyStep, lang_of, text

logger = structlog.get_logger(__name__)

SURVEY_DONE = len(SURVEY) + 1

# Intents that are answered even while a Brain job is running: they never
# touch the library, so the student is not made to wait for it.
_NOT_BLOCKED_BY_JOB = {IsmaelIntent.MOODLE_SUPPORT, IsmaelIntent.CONTACT_TEACHER}
# Intents a matching FAQ answers directly. A greeting stays a greeting; a
# teacher request keeps its flow and offers the FAQ as the "help first".
_FAQ_INTENTS = {IsmaelIntent.THEOLOGY, IsmaelIntent.MOODLE_SUPPORT, IsmaelIntent.OTHER}


def _reply_while_consulting(message: str, lang: str, **updates) -> dict:
    """A reply that promises the answer later: the widget keeps a waiting
    indicator up until it arrives, for as long as a job may legitimately run."""
    result = _reply(message, **updates)
    emit_pending_followup(text("working", lang), ttl_seconds=int(rag_job.JOB_TTL_SECONDS))
    return result


def _ask_survey_step(message: str, step: SurveyStep, lang: str, **updates) -> dict:
    """A survey question: the numbered text for every channel, buttons where drawn.

    ``message`` must end with ``step.render(lang)`` — the backend only keeps the
    buttons when the text ends with exactly those numbered options.
    """
    result = _reply(message, **updates)
    emit_quick_replies(step.labels(lang))
    return result


def _survey_done(state: AgentState, ismael: dict) -> bool:
    ctx = state.get("user_context") or {}
    from_backend = isinstance(ctx, dict) and bool(ctx.get("ismael_survey_done"))
    return from_backend or ismael.get("survey_step", 0) >= SURVEY_DONE


def _first_name(state: AgentState) -> str:
    ctx = state.get("user_context") or {}
    name = (ctx.get("name") or "").strip() if isinstance(ctx, dict) else ""
    return name.split()[0] if name else ""


async def _spawn_answer(state: AgentState, config: RunnableConfig, question: str) -> None:
    await rag_job.spawn(
        rag_job.RagJob(
            tenant_id=state.get("tenant_id") or "",
            conversation_id=state.get("conversation_id") or "",
            question=question,
            language=lang_of(state),
            persona=resolve_persona(state, DEFAULT_PERSONA),
            llm_config=rag_job.job_llm_config(config),
        )
    )


# ── Triage ───────────────────────────────────────────────────────────────────


async def ismael_triage_node(state: AgentState, config: RunnableConfig) -> dict:
    """Decide the turn's branch. Deterministic whenever state already decides it."""
    record_node_invocation("ismael_triage")
    ismael = _ismael(state)

    if 1 <= ismael.get("survey_step", 0) < SURVEY_DONE:
        return {"ismael_route": "survey"}

    if teacher.flow_open(ismael):
        return {"ismael_route": "teacher"}
    ismael.pop("teacher", None)  # an expired flow is forgotten, not resumed

    running = await rag_job.running_job(
        state.get("tenant_id") or "", state.get("conversation_id") or ""
    )

    user_text = latest_user_text(state)
    candidates = faq_candidates(state)
    system = resolve_prompt(config, "THEOLOGY_TRIAGE", TRIAGE_PROMPT, state)
    usage: list = []
    try:
        provider = get_provider(config)
        model = resolve_background_model(config)
        result = await bounded_structured(
            provider,
            [
                {
                    "role": "system",
                    "content": system + format_user_context(state) + faq_candidates_block(candidates),
                },
                *history_messages(state),
                {"role": "user", "content": user_text},
            ],
            model,
            TriageResult,
            timeout=settings.ismael_timeout_triage_seconds,
            thinking_budget=settings.ismael_thinking_classify,
        )
        usage.append(make_usage_record(node="ismael_triage", provider=provider, model=model))
    except Exception as exc:  # noqa: BLE001 — a keyword guess beats no reply
        fragments = latest_user_messages(state)
        intent = fallback_intent(fragments)
        logger.warning(
            "ismael_triage_failed",
            error=str(exc) or type(exc).__name__,
            fallback_intent=intent.value,
        )
        result = TriageResult(intent=intent, question=user_text)

    faq = pick_faq(candidates, result.faq_id)
    answered_by_faq = faq is not None and result.intent in _FAQ_INTENTS

    if running and result.intent not in _NOT_BLOCKED_BY_JOB and not answered_by_faq:
        ismael["pending_question"] = running.get("question") or ismael.get("pending_question", "")
        return {"ismael_route": "pending", "ismael": ismael, "turn_usage": usage}

    route = {
        IsmaelIntent.THEOLOGY: "start",
        IsmaelIntent.MOODLE_SUPPORT: "moodle_support",
        IsmaelIntent.CONTACT_TEACHER: "teacher",
        IsmaelIntent.GREETING: "greeting",
        IsmaelIntent.OTHER: "off_topic",
    }[result.intent]
    ismael["question"] = (result.question or user_text).strip()
    if answered_by_faq:
        # The institution's curated answer wins — a theology question it
        # covers never reaches Brain, and asks no survey.
        ismael["faq"] = faq
        route = "faq"
    if result.intent == IsmaelIntent.CONTACT_TEACHER:
        ismael["teacher"] = teacher.new_flow(result.question, result.teacher_topic, faq=faq)
    return {
        "ismael_route": route,
        "intent": result.intent.value,
        "ismael": ismael,
        "turn_usage": usage,
    }


# ── A new question ───────────────────────────────────────────────────────────


async def ismael_start_node(state: AgentState, config: RunnableConfig) -> dict:
    """Start Brain, then either open the survey or hand the question to the job."""
    record_node_invocation("ismael_start")
    ismael = _ismael(state)
    lang = lang_of(state)
    question = ismael.get("question") or latest_user_text(state)
    ismael["pending_question"] = question

    if _survey_done(state, ismael):
        await _spawn_answer(state, config, question)
        return _reply_while_consulting(text("consulting", lang), lang, ismael=ismael)

    # Boot Brain now; the three survey turns are what pay for its ~1 min start.
    rag_job.start_host_in_background()
    ismael["survey_step"] = 1
    ismael["survey_answers"] = {}
    name = _first_name(state)
    message = (
        text("survey_intro", lang, name=f", {name}" if name else "")
        + f"\n\n1/{len(SURVEY)} — "
        + SURVEY[0].render(lang)
    )
    return _ask_survey_step(message, SURVEY[0], lang, ismael=ismael)


# ── Survey ───────────────────────────────────────────────────────────────────


def parse_survey_answer(step_index: int, reply: str) -> str | None:
    """Option key from a number or an option label; None when it takes an LLM."""
    step = SURVEY[step_index]
    index = match_option(reply, [[es, en] for _, es, en in step.options])
    return step.options[index][0] if index is not None else None


async def _classify_survey_answer(
    config: RunnableConfig, step_index: int, reply: str
) -> tuple[SurveyAnswer, list]:
    step = SURVEY[step_index]
    options = "\n".join(f"{key} → {es} / {en}" for key, es, en in step.options)
    system = SURVEY_PROMPT.format(question=step.question["es"], options=options)
    try:
        provider = get_provider(config)
        model = resolve_background_model(config)
        result = await bounded_structured(
            provider,
            [{"role": "system", "content": system}, {"role": "user", "content": reply}],
            model,
            SurveyAnswer,
            timeout=settings.ismael_timeout_step_seconds,
            thinking_budget=settings.ismael_thinking_classify,
        )
        usage = [make_usage_record(node="ismael_survey", provider=provider, model=model)]
    except Exception as exc:  # noqa: BLE001 — an unreadable answer is just "no_answer"
        logger.info("ismael_survey_classify_failed", error=str(exc))
        return SurveyAnswer(), []
    if result.answer not in step.keys():
        result.answer = NO_ANSWER
    return result, usage


async def ismael_survey_node(state: AgentState, config: RunnableConfig) -> dict:
    """Record one statistics answer (never asking twice) and move on."""
    record_node_invocation("ismael_survey")
    ismael = _ismael(state)
    lang = lang_of(state)
    step_number = ismael.get("survey_step", 1)
    step_index = step_number - 1
    reply = latest_user_text(state)

    usage: list = []
    answer = parse_survey_answer(step_index, reply)
    if answer is None:
        result, usage = await _classify_survey_answer(config, step_index, reply)
        answer = result.answer
        if result.is_new_question and reply.strip():
            # They asked something else instead of answering: answer that one.
            ismael["pending_question"] = reply.strip()

    answers = dict(ismael.get("survey_answers") or {})
    answers[SURVEY[step_index].field] = answer
    ismael["survey_answers"] = answers

    if step_number < len(SURVEY):
        ismael["survey_step"] = step_number + 1
        message = (
            text("survey_next", lang)
            + f"\n\n{step_number + 1}/{len(SURVEY)} — "
            + SURVEY[step_number].render(lang)
        )
        return _ask_survey_step(
            message, SURVEY[step_number], lang, ismael=ismael, turn_usage=usage
        )

    ismael["survey_step"] = SURVEY_DONE
    contact_id = state.get("contact_id") or ""
    if contact_id:
        try:
            await backend_client.save_ismael_survey(
                contact_id, state.get("conversation_id") or "", answers
            )
        except Exception as exc:  # noqa: BLE001 — statistics must not block the answer
            logger.warning("ismael_survey_save_failed", error=str(exc))
    await _spawn_answer(state, config, ismael.get("pending_question") or reply)
    return _reply_while_consulting(
        text("survey_done", lang), lang, ismael=ismael, turn_usage=usage
    )


# ── Holding and off-topic replies ────────────────────────────────────────────


async def ismael_pending_node(state: AgentState, config: RunnableConfig) -> dict:
    record_node_invocation("ismael_pending")
    ismael = _ismael(state)
    question = ismael.get("pending_question") or ""
    lang = lang_of(state)
    return _reply_while_consulting(text("pending", lang, question=question[:160]), lang)


async def ismael_off_topic_node(state: AgentState, config: RunnableConfig) -> dict:
    record_node_invocation("ismael_off_topic")
    persona = resolve_persona(state, DEFAULT_PERSONA)
    return _reply(text("off_topic", lang_of(state), persona=persona))
