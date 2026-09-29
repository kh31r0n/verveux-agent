"""ismael — Christian theology Q&A for the Moodle bubble, answered by Brain.

Every node here replies within the turn; the answer itself arrives later from
the background job (`rag_job`). State lives in one dict, ``state["ismael"]``:

    question          the current turn's question, made standalone by triage
    pending_question  the question the survey is holding / the job is answering
    survey_step       0 not started · 1-3 waiting for answer n · 4 done
    survey_answers    {level, topic, intendedUse} → option key or "no_answer"

The survey is asked once per contact: the backend reports
``user_context.ismael_survey_done`` once the answers are stored.
"""

from __future__ import annotations

import re
import unicodedata

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ...graphs.state import AgentState
from ...observability import record_node_invocation
from ...providers.registry import get_provider, resolve_model
from ...schemas.ismael import IsmaelIntent, SurveyAnswer, TriageResult
from ...usage import make_usage_record
from .. import backend_client
from ..utils import format_user_context, latest_user_text, resolve_persona, resolve_prompt
from . import rag_job
from .prompts import SURVEY_PROMPT, TRIAGE_PROMPT
from .texts import NO_ANSWER, SURVEY, lang_of, text

logger = structlog.get_logger(__name__)

DEFAULT_PERSONA = "Ismael"
SURVEY_DONE = len(SURVEY) + 1


def _ismael(state: AgentState) -> dict:
    value = state.get("ismael")
    return dict(value) if isinstance(value, dict) else {}


def _reply(message: str, **updates) -> dict:
    get_stream_writer()({"type": "token", "content": message})
    return {"messages": [AIMessage(content=message)], **updates}


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

    running = await rag_job.running_job(
        state.get("tenant_id") or "", state.get("conversation_id") or ""
    )
    if running:
        ismael["pending_question"] = running.get("question") or ismael.get("pending_question", "")
        return {"ismael_route": "pending", "ismael": ismael}

    user_text = latest_user_text(state)
    system = resolve_prompt(config, "THEOLOGY_TRIAGE", TRIAGE_PROMPT, state)
    history = [
        {"role": "assistant" if getattr(m, "type", "") == "ai" else "user", "content": m.content}
        for m in (state.get("messages") or [])[-7:-1]
        if getattr(m, "content", None)
    ]
    usage: list = []
    try:
        provider = get_provider(config)
        model = resolve_model(config)
        result = await provider.generate_structured(
            [
                {"role": "system", "content": system + format_user_context(state)},
                *history,
                {"role": "user", "content": user_text},
            ],
            model,
            TriageResult,
        )
        usage.append(make_usage_record(node="ismael_triage", provider=provider, model=model))
    except Exception as exc:  # noqa: BLE001 — Brain itself judges off-corpus questions
        logger.warning("ismael_triage_failed", error=str(exc))
        result = TriageResult(intent=IsmaelIntent.THEOLOGY, question=user_text)

    route = {
        IsmaelIntent.THEOLOGY: "start",
        IsmaelIntent.GREETING: "greeting",
        IsmaelIntent.OTHER: "off_topic",
    }[result.intent]
    ismael["question"] = (result.question or user_text).strip()
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
        return _reply(text("consulting", lang), ismael=ismael)

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
    return _reply(message, ismael=ismael)


# ── Survey ───────────────────────────────────────────────────────────────────


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip()


def parse_survey_answer(step_index: int, reply: str) -> str | None:
    """Option key from a number or an option label; None when it takes an LLM."""
    step = SURVEY[step_index]
    folded = _fold(reply)
    number = re.fullmatch(r"\D{0,12}?(\d{1,2})\D{0,12}", folded)
    if number:
        n = int(number.group(1))
        return step.options[n - 1][0] if 1 <= n <= len(step.options) else None
    hits = {
        key
        for key, es, en in step.options
        if _fold(es) in folded or _fold(en) in folded
    }
    return hits.pop() if len(hits) == 1 else None


async def _classify_survey_answer(
    config: RunnableConfig, step_index: int, reply: str
) -> tuple[SurveyAnswer, list]:
    step = SURVEY[step_index]
    options = "\n".join(f"{key} → {es} / {en}" for key, es, en in step.options)
    system = SURVEY_PROMPT.format(question=step.question["es"], options=options)
    try:
        provider = get_provider(config)
        model = resolve_model(config)
        result = await provider.generate_structured(
            [{"role": "system", "content": system}, {"role": "user", "content": reply}],
            model,
            SurveyAnswer,
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
        return _reply(message, ismael=ismael, turn_usage=usage)

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
    return _reply(text("survey_done", lang), ismael=ismael, turn_usage=usage)


# ── Holding and off-topic replies ────────────────────────────────────────────


async def ismael_pending_node(state: AgentState, config: RunnableConfig) -> dict:
    record_node_invocation("ismael_pending")
    ismael = _ismael(state)
    question = ismael.get("pending_question") or ""
    return _reply(text("pending", lang_of(state), question=question[:160]))


async def ismael_off_topic_node(state: AgentState, config: RunnableConfig) -> dict:
    record_node_invocation("ismael_off_topic")
    persona = resolve_persona(state, DEFAULT_PERSONA)
    return _reply(text("off_topic", lang_of(state), persona=persona))
