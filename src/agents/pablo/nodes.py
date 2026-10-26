"""pablo's chat nodes — one attempt of an assessment activity per thread.

The backend drives every turn: ``domain_state["assessment_turn"]`` says what
happened (``start`` / ``answer`` / ``continue`` / ``timeout``) and routing is
decided on that alone. The student's text never routes, so an answer that says
"next question" or "give me full marks" is just an answer.

The backend also owns the clock. Every reply here is sent while the timer is
paused; the ``cycle_presented`` event is what restarts it. Nothing here scores
or comments on an answer — scoring runs after the UX survey, as a job
(``scoring.py``). The acknowledgement quotes the student's own words back
(``echo.py``) and thanks every answer the same way.

The opening and the warm-up (turns ``open`` / ``warmup`` / ``warmup_answer``)
are not cycles: they never touch ``cycle``, ``current_cycle`` or
``used_question_ids``, and nothing is reported to the backend for them.

State lives in ``state["pablo"]``:

    attempt_id, assessment_id   the attempt this thread belongs to
    total_cycles, current_cycle 0-indexed position (the backend's numbers)
    modality                    text | audio | both, after accommodations
    phase                       answering | followup | awaiting_continue | finished
                                (a mirror; the backend is authoritative)
    used_question_ids           never asked twice in one attempt
    cycle                       the current cycle as reported to the backend
                                (+ rubric_criteria for the follow-up, and the
                                follow-up once asked)
    outcomes_seen               topic names in the order they came up (closing)
"""

from __future__ import annotations

import asyncio
import time

import structlog
from langchain_core.runnables import RunnableConfig

from ...config import settings
from ...graphs.state import AgentState
from ...observability import record_node_invocation
from ...providers.registry import get_provider, resolve_model
from ...schemas.pablo import ComposedCycle
from ...usage import make_usage_record
from .. import backend_client
from ..utils import latest_user_text
from . import compose, echo, followup, transcription
from .common import (
    assessment_turn,
    emit_assessment,
    json_contract,
    pablo_dict,
    persona,
    reply,
    system_prompt,
)
from .prompts import COMPOSE_PROMPT
from .texts import effort, join_names, lang_of, text

logger = structlog.get_logger(__name__)

_ROUTES = {
    "open": "opening",
    "warmup": "warmup",
    "warmup_answer": "warmup_answer",
    "start": "next_cycle",
    "continue": "next_cycle",
    "answer": "answer",
    "followup_answer": "followup_answer",
    "timeout": "timeout",
}


# ── Triage ───────────────────────────────────────────────────────────────────


async def pablo_triage_node(state: AgentState) -> dict:
    record_node_invocation("pablo_triage")
    turn = assessment_turn(state)
    pablo = pablo_dict(state)
    attempt_id = str(turn.get("attempt_id") or "")
    route = _ROUTES.get(str(turn.get("kind") or ""), "unexpected")

    if not attempt_id or (pablo.get("attempt_id") and pablo["attempt_id"] != attempt_id):
        # One thread is one attempt: a turn for another attempt is a backend bug.
        route = "unexpected"
    elif route == "answer" and not pablo.get("cycle"):
        route = "unexpected"
    elif route == "followup_answer" and not (pablo.get("cycle") or {}).get("followup"):
        route = "unexpected"
    elif route != "unexpected":
        pablo.setdefault("attempt_id", attempt_id)
        pablo["assessment_id"] = str(turn.get("assessment_id") or pablo.get("assessment_id") or "")
        pablo["total_cycles"] = int(turn.get("total_cycles") or pablo.get("total_cycles") or 0)
        pablo["modality"] = str(turn.get("modality") or pablo.get("modality") or "text")
        pablo.setdefault("used_question_ids", [])

    logger.info("pablo_triage", kind=turn.get("kind"), route=route, attempt_id=attempt_id)
    return {"pablo": pablo, "pablo_route": route}


# ── Next cycle ───────────────────────────────────────────────────────────────


def _present(pablo: dict, cycle: dict, usage: list) -> dict:
    emit_assessment("cycle_presented", cycle_index=cycle["cycle_index"])
    return reply(cycle["composite_prompt"], pablo=pablo, turn_usage=usage)


def _compose_failed(state: AgentState, pablo: dict, reason: str) -> dict:
    logger.warning("pablo_compose_failed", attempt_id=pablo.get("attempt_id"), reason=reason)
    emit_assessment("compose_failed", reason=reason)
    return reply(text("compose_failed", lang_of(state)), pablo=pablo)


async def _compose(state: AgentState, config: RunnableConfig, outcome: dict, candidates: list[dict]):
    """(ComposedCycle | None, usage, provenance). None = use the candidates verbatim."""
    lang = lang_of(state)
    system, provenance = system_prompt(
        config, "ASSESSMENT_COMPOSE", COMPOSE_PROMPT, lang, persona(state)
    )
    try:
        provider = get_provider(config)
        model = resolve_model(config)
        started = time.monotonic()
        result = await asyncio.wait_for(
            provider.generate_structured(
                [
                    {"role": "system", "content": system + json_contract(ComposedCycle)},
                    {"role": "user", "content": compose.candidates_block(outcome, candidates)},
                ],
                model,
                ComposedCycle,
                thinking_budget=settings.pablo_thinking_compose,
            ),
            timeout=settings.pablo_timeout_compose_seconds,
        )
        usage = [
            make_usage_record(
                node="pablo_next_cycle",
                provider=provider,
                model=model,
                latency_ms=int((time.monotonic() - started) * 1000),
                provenance=provenance,
            )
        ]
        return result, usage, {**provenance, "model_id": model}
    except Exception as exc:  # noqa: BLE001 — the questions verbatim still assess the outcome
        logger.info("pablo_compose_llm_failed", error=str(exc)[:300])
        return None, [], {**provenance, "model_id": "", "fallback": "verbatim"}


async def pablo_next_cycle_node(state: AgentState, config: RunnableConfig) -> dict:
    record_node_invocation("pablo_next_cycle")
    turn = assessment_turn(state)
    pablo = pablo_dict(state)
    lang = lang_of(state)
    attempt_id = pablo["attempt_id"]

    # A retried start/continue for the cycle already shown: show it again
    # rather than composing (and reporting) a second one.
    existing = pablo.get("cycle") or {}
    wanted = turn.get("cycle_index")
    if (
        existing
        and pablo.get("phase") == "answering"
        and (wanted is None or int(wanted) == existing.get("cycle_index"))
    ):
        return _present(pablo, existing, [])

    try:
        payload = await backend_client.get_assessment_next_cycle(attempt_id)
    except Exception as exc:  # noqa: BLE001
        return _compose_failed(state, pablo, f"next_cycle: {exc}"[:200])

    used = set(pablo.get("used_question_ids") or [])
    candidates = [
        c for c in compose.usable_candidates(payload.get("candidates")) if c["snapshot_id"] not in used
    ]
    if not candidates:
        return _compose_failed(state, pablo, "no_candidates")

    cycle_index = int(payload.get("cycle_index", pablo.get("current_cycle", 0)))
    total = int(payload.get("total_cycles") or pablo.get("total_cycles") or cycle_index + 1)
    outcome = payload.get("learning_outcome") or {}

    result, usage, provenance = await _compose(state, config, outcome, candidates)
    parts = compose.validated_parts(result, candidates)
    cycle = {
        "cycle_index": cycle_index,
        "learning_outcome_id": str(outcome.get("id") or ""),
        "composite_prompt": compose.render(
            result.intro if result else "",
            parts,
            cycle_index,
            total,
            lang,
            header=compose.progress_line(payload, cycle_index, total, lang),
        ),
        "question_snapshot_ids": [p.question_snapshot_id for p in parts],
        "component_map": compose.component_map(parts, candidates, lang),
        "rubric_criteria": compose.component_criteria(parts, candidates),
        "prompt_provenance": provenance,
    }
    try:
        await backend_client.report_assessment_cycle(
            attempt_id,
            cycle_index,
            {
                "learningOutcomeId": cycle["learning_outcome_id"],
                "compositePrompt": cycle["composite_prompt"],
                "questionSnapshotIds": cycle["question_snapshot_ids"],
                "components": cycle["component_map"],
                "provenance": provenance,
            },
        )
    except Exception as exc:  # noqa: BLE001 — never show a question the backend did not store
        return _compose_failed(state, pablo, f"report_cycle: {exc}"[:200])

    seen = list(pablo.get("outcomes_seen") or [])
    name = str(outcome.get("name") or "").strip()
    if name and name not in seen:
        seen.append(name)
    pablo.update(
        current_cycle=cycle_index,
        total_cycles=total,
        phase="answering",
        used_question_ids=sorted(used | set(cycle["question_snapshot_ids"])),
        cycle=cycle,
        outcomes_seen=seen,
    )
    logger.info(
        "pablo_cycle_presented",
        attempt_id=attempt_id,
        cycle_index=cycle_index,
        parts=len(parts),
        components=len(cycle["component_map"]),
        composed=result is not None,
    )
    return _present(pablo, cycle, usage)


# ── Answer ───────────────────────────────────────────────────────────────────


def _is_last(pablo: dict, turn: dict, cycle_index: int) -> bool:
    if "is_last_cycle" in turn:
        return bool(turn["is_last_cycle"])
    return cycle_index + 1 >= int(pablo.get("total_cycles") or 0)


def _closing(pablo: dict, lang: str) -> str:
    """The neutral recap before the survey: which topics, never how it went."""
    topics = join_names(list(pablo.get("outcomes_seen") or []), lang)
    closing = text("closing", lang, topics=topics) if topics else text("closing_plain", lang)
    return f"{closing}\n\n{text('survey_invite', lang)}"


async def _transcribe(answer: dict, lang: str):
    audio_url = str(answer.get("audio_url") or "")
    return (await transcription.transcribe(audio_url, lang)) if audio_url else None


def _acknowledgement(typed: str, transcript, quote: str | None, cycle_index: int, lang: str) -> list[str]:
    """The fixed lines after an answer: the student's own words (quote or
    transcript) and the same thanks for every answer with content."""
    if not typed and not transcript:
        return [text("empty_answer", lang)]
    heard = bool(transcript and transcript.status == "ok" and transcript.text)
    first = [text("echo", lang, quote=quote) if quote else text("received", lang)]
    if typed or heard:
        first.append(effort(cycle_index, lang))
    lines = [" ".join(first)]
    if heard:
        lines.append(text("transcript", lang, transcript=transcript.text))
    elif transcript:
        lines.append(text("transcription_failed", lang))
    return lines


def _response_body(typed: str, answer: dict, transcript) -> dict:
    audio_url = str(answer.get("audio_url") or "")
    return {
        "text": typed or None,
        "transcript": transcript.text if transcript and transcript.text else None,
        "modalityUsed": "both" if typed and audio_url else ("audio" if audio_url else "text"),
        "transcriptionStatus": transcript.status if transcript else None,
        "transcriptionError": transcript.error if transcript and transcript.error else None,
        "timedOut": bool(answer.get("timed_out")),
    }


def _finish(
    pablo: dict, turn: dict, cycle_index: int, typed: str, transcript, quote, usage: list, lang: str, part: str
) -> dict:
    """The acknowledgement, then Continue or the closing: the cycle is done."""
    last = _is_last(pablo, turn, cycle_index)
    lines = _acknowledgement(typed, transcript, quote, cycle_index, lang)
    lines.append(_closing(pablo, lang) if last else text("continue", lang))
    pablo["phase"] = "finished" if last else "awaiting_continue"
    emit_assessment("response_recorded", cycle_index=cycle_index, is_last=last)
    logger.info(
        "pablo_answer_recorded",
        attempt_id=pablo["attempt_id"],
        cycle_index=cycle_index,
        part=part,
        transcription=transcript.status if transcript else None,
        quoted=quote is not None,
        is_last=last,
    )
    return reply("\n\n".join(lines), pablo=pablo, turn_usage=usage)


async def _ask_followup(
    state: AgentState, config: RunnableConfig, pablo: dict, cycle_index: int, typed: str, transcript, content: str
):
    """(reply | None, quote, usage). The follow-up is reported before it is
    shown; when the backend did not store it, the cycle ends as it would have."""
    lang = lang_of(state)
    cycle = pablo.get("cycle") or {}
    plan, usage = await followup.plan_for(config, cycle, content, lang, persona(state))
    # The quote is shown only for typed words; audio already shows its transcript.
    quote = echo.validated_quote(plan["quote"] or "", typed) if typed else None
    try:
        await backend_client.report_assessment_followup(
            pablo["attempt_id"],
            cycle_index,
            {
                "prompt": plan["question"],
                "targetComponentId": plan["target_component_id"],
                "mode": plan["mode"],
                "provenance": plan["provenance"],
            },
        )
    except Exception as exc:  # noqa: BLE001 — never show a question the backend did not store
        logger.warning("pablo_followup_report_failed", error=str(exc)[:300])
        return None, quote, usage

    cycle["followup"] = {
        "prompt": plan["question"],
        "target_component_id": plan["target_component_id"],
        "mode": plan["mode"],
    }
    pablo.update(cycle=cycle, phase="followup")
    lines = _acknowledgement(typed, transcript, quote, cycle_index, lang)
    lines.append(f"{text('followup_intro', lang)}\n{plan['question']}")
    emit_assessment("followup_presented", cycle_index=cycle_index)
    logger.info(
        "pablo_followup_presented",
        attempt_id=pablo["attempt_id"],
        cycle_index=cycle_index,
        mode=plan["mode"],
        fallback=plan["provenance"].get("fallback"),
    )
    return reply("\n\n".join(lines), pablo=pablo, turn_usage=usage), quote, usage


async def pablo_receive_answer_node(state: AgentState, config: RunnableConfig) -> dict:
    record_node_invocation("pablo_receive_answer")
    turn = assessment_turn(state)
    pablo = pablo_dict(state)
    lang = lang_of(state)
    answer = turn.get("answer") if isinstance(turn.get("answer"), dict) else {}
    cycle_index = int(turn.get("cycle_index", pablo.get("current_cycle", 0)))
    # The backend says whether a follow-up may be asked (the activity allows
    # it and the attempt has time left); whether it is asked then depends
    # only on the answer having content — never on how good it is.
    followup_allowed = bool(turn.get("followup_allowed"))

    typed = str(answer.get("text") if "text" in answer else latest_user_text(state)).strip()
    # The quote only needs the typed text, so it runs while the audio is
    # transcribed and the response reported (the follow-up call quotes too).
    quote_task = (
        None if followup_allowed else asyncio.create_task(echo.quote_of(config, typed, lang, persona(state)))
    )
    transcript = await _transcribe(answer, lang)
    try:
        await backend_client.report_assessment_response(
            pablo["attempt_id"], cycle_index, _response_body(typed, answer, transcript)
        )
    except Exception as exc:  # noqa: BLE001
        if quote_task:
            quote_task.cancel()
        logger.warning("pablo_answer_report_failed", error=str(exc)[:300])
        emit_assessment("answer_failed", cycle_index=cycle_index)
        return reply(text("answer_failed", lang), pablo=pablo)

    heard = transcript.text if transcript and transcript.status == "ok" and transcript.text else ""
    content = "\n".join(p for p in (typed, heard) if p)
    quote, usage = None, []
    if followup_allowed and content:
        presented, quote, usage = await _ask_followup(state, config, pablo, cycle_index, typed, transcript, content)
        if presented:
            return presented
    elif quote_task:
        quote, usage = await quote_task
    return _finish(pablo, turn, cycle_index, typed, transcript, quote, usage, lang, "answer")


async def pablo_followup_answer_node(state: AgentState, config: RunnableConfig) -> dict:
    """The answer to the follow-up question: recorded beside the first one
    (both are scored together), then the cycle ends as any other."""
    record_node_invocation("pablo_followup_answer")
    turn = assessment_turn(state)
    pablo = pablo_dict(state)
    lang = lang_of(state)
    answer = turn.get("answer") if isinstance(turn.get("answer"), dict) else {}
    cycle_index = int(turn.get("cycle_index", pablo.get("current_cycle", 0)))
    typed = str(answer.get("text") if "text" in answer else latest_user_text(state)).strip()

    quote_task = asyncio.create_task(echo.quote_of(config, typed, lang, persona(state)))
    transcript = await _transcribe(answer, lang)
    try:
        await backend_client.report_assessment_followup_response(
            pablo["attempt_id"], cycle_index, _response_body(typed, answer, transcript)
        )
    except Exception as exc:  # noqa: BLE001
        quote_task.cancel()
        logger.warning("pablo_followup_answer_report_failed", error=str(exc)[:300])
        emit_assessment("answer_failed", cycle_index=cycle_index)
        return reply(text("answer_failed", lang), pablo=pablo)
    quote, usage = await quote_task
    return _finish(pablo, turn, cycle_index, typed, transcript, quote, usage, lang, "followup")


# ── Opening and warm-up (not cycles) ─────────────────────────────────────────


async def pablo_opening_node(state: AgentState) -> dict:
    """The greeting. Fixed text: the student's first name (attested by Moodle),
    what the conversation is about, and the warm-up offer."""
    record_node_invocation("pablo_opening")
    turn = assessment_turn(state)
    pablo = pablo_dict(state)
    lang = lang_of(state)
    name = " ".join(str(turn.get("student_name") or "").split())[:60]
    who = persona(state)
    topics = [str(t).strip() for t in turn.get("topics") or [] if str(t).strip()]
    warmup = bool(turn.get("warmup_available"))

    first = text("hello_name", lang, name=name, persona=who) if name else text("hello", lang, persona=who)
    lines = [f"{first} {text('opening_intro', lang)}"]
    plan = []
    if len(topics) > 1:
        plan.append(text("opening_topics", lang, n=len(topics), topics=join_names(topics, lang)))
    elif topics:
        plan.append(text("opening_topic", lang, topics=topics[0]))
    plan.append(text("opening_clock", lang))
    lines.append(" ".join(plan))
    lines.append(text("opening_warmup" if warmup else "opening_ready", lang))

    pablo["phase"] = "opening"
    emit_assessment("opening_presented", warmup_available=warmup)
    logger.info("pablo_opening", attempt_id=pablo.get("attempt_id"), named=bool(name), warmup=warmup)
    return reply("\n\n".join(lines), pablo=pablo)


async def pablo_warmup_node(state: AgentState) -> dict:
    """The teacher's warm-up question, verbatim. No model, no candidates."""
    record_node_invocation("pablo_warmup")
    turn = assessment_turn(state)
    pablo = pablo_dict(state)
    lang = lang_of(state)
    prompt = str(turn.get("warmup_prompt") or "").strip()
    if not prompt:
        return await pablo_error_node(state)
    pablo["phase"] = "warmup"
    emit_assessment("warmup_presented")
    return reply(text("warmup_question", lang, prompt=prompt), pablo=pablo)


async def pablo_warmup_answer_node(state: AgentState, config: RunnableConfig) -> dict:
    """The warm-up's answer: transcribed (it is the microphone test) and
    acknowledged exactly like a real one, so the student meets the format.
    Nothing is reported — the answer exists only in the conversation."""
    record_node_invocation("pablo_warmup_answer")
    turn = assessment_turn(state)
    pablo = pablo_dict(state)
    lang = lang_of(state)
    answer = turn.get("answer") if isinstance(turn.get("answer"), dict) else {}
    typed = str(answer.get("text") if "text" in answer else latest_user_text(state)).strip()

    quote_task = asyncio.create_task(echo.quote_of(config, typed, lang, persona(state)))
    transcript = await _transcribe(answer, lang)
    quote, usage = await quote_task

    lines = []
    if quote:
        lines.append(text("echo", lang, quote=quote))
    if transcript and transcript.status == "ok" and transcript.text:
        lines.append(text("warmup_audio_ok", lang, transcript=transcript.text))
    elif transcript:
        lines.append(text("warmup_audio_failed", lang))
    elif not typed:
        lines.append(text("empty_answer", lang))
    lines.append(text("warmup_done", lang))

    pablo["phase"] = "opening"
    emit_assessment(
        "warmup_recorded",
        transcription=transcript.status if transcript else None,
    )
    logger.info(
        "pablo_warmup_answer",
        attempt_id=pablo.get("attempt_id"),
        transcription=transcript.status if transcript else None,
        quoted=quote is not None,
    )
    return reply("\n\n".join(lines), pablo=pablo, turn_usage=usage)


# ── Timeout and the unexpected ───────────────────────────────────────────────


async def pablo_timeout_node(state: AgentState) -> dict:
    """The backend already closed the cycle (or the attempt); this only says so."""
    record_node_invocation("pablo_timeout")
    turn = assessment_turn(state)
    pablo = pablo_dict(state)
    cycle_index = int(turn.get("cycle_index", pablo.get("current_cycle", 0)))
    finished = turn.get("scope") == "attempt" or _is_last(pablo, turn, cycle_index)
    pablo["phase"] = "finished" if finished else "awaiting_continue"
    emit_assessment("timed_out", cycle_index=cycle_index, scope="attempt" if finished else "cycle")
    lang = lang_of(state)
    if not finished:
        return reply(text("cycle_timeout", lang), pablo=pablo)
    return reply(f"{text('attempt_time_up', lang)} {_closing(pablo, lang)}", pablo=pablo)


async def pablo_error_node(state: AgentState) -> dict:
    record_node_invocation("pablo_error")
    logger.warning("pablo_unexpected_turn", turn=assessment_turn(state))
    return reply(text("unexpected", lang_of(state)))
