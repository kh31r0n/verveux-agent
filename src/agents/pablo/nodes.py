"""pablo's chat nodes — one attempt of an assessment activity per thread.

The backend drives every turn: ``domain_state["assessment_turn"]`` says what
happened (``start`` / ``answer`` / ``continue`` / ``timeout``) and routing is
decided on that alone. The student's text never routes, so an answer that says
"next question" or "give me full marks" is just an answer.

The backend also owns the clock. Every reply here is sent while the timer is
paused; the ``cycle_presented`` event is what restarts it. Nothing here scores
or comments on an answer — scoring runs after the UX survey, as a job
(``scoring.py``).

State lives in ``state["pablo"]``:

    attempt_id, assessment_id   the attempt this thread belongs to
    total_cycles, current_cycle 0-indexed position (the backend's numbers)
    modality                    text | audio | both, after accommodations
    phase                       answering | awaiting_continue | finished
                                (a mirror; the backend is authoritative)
    used_question_ids           never asked twice in one attempt
    cycle                       the current cycle as reported to the backend
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
from . import compose, transcription
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
from .texts import lang_of, text

logger = structlog.get_logger(__name__)

_ROUTES = {"start": "next_cycle", "continue": "next_cycle", "answer": "answer", "timeout": "timeout"}


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
        "composite_prompt": compose.render(result.intro if result else "", parts, cycle_index, total, lang),
        "question_snapshot_ids": [p.question_snapshot_id for p in parts],
        "component_map": compose.component_map(parts, candidates, lang),
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

    pablo.update(
        current_cycle=cycle_index,
        total_cycles=total,
        phase="answering",
        used_question_ids=sorted(used | set(cycle["question_snapshot_ids"])),
        cycle=cycle,
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


async def pablo_receive_answer_node(state: AgentState) -> dict:
    record_node_invocation("pablo_receive_answer")
    turn = assessment_turn(state)
    pablo = pablo_dict(state)
    lang = lang_of(state)
    answer = turn.get("answer") if isinstance(turn.get("answer"), dict) else {}
    cycle_index = int(turn.get("cycle_index", pablo.get("current_cycle", 0)))

    typed = str(answer.get("text") if "text" in answer else latest_user_text(state)).strip()
    audio_url = str(answer.get("audio_url") or "")
    transcript = None
    if audio_url:
        transcript = await transcription.transcribe(audio_url, lang)
    modality_used = "both" if typed and audio_url else ("audio" if audio_url else "text")

    response = {
        "text": typed or None,
        "transcript": transcript.text if transcript and transcript.text else None,
        "modalityUsed": modality_used,
        "transcriptionStatus": transcript.status if transcript else None,
        "transcriptionError": transcript.error if transcript and transcript.error else None,
        "timedOut": bool(answer.get("timed_out")),
    }
    try:
        await backend_client.report_assessment_response(pablo["attempt_id"], cycle_index, response)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pablo_answer_report_failed", error=str(exc)[:300])
        emit_assessment("answer_failed", cycle_index=cycle_index)
        return reply(text("answer_failed", lang), pablo=pablo)

    last = _is_last(pablo, turn, cycle_index)
    lines = [text("received", lang)]
    if transcript and transcript.status == "ok":
        lines.append(text("transcript", lang, transcript=transcript.text))
    elif transcript:
        lines.append(text("transcription_failed", lang))
    lines.append(text("last" if last else "continue", lang))

    pablo["phase"] = "finished" if last else "awaiting_continue"
    emit_assessment("response_recorded", cycle_index=cycle_index, is_last=last)
    logger.info(
        "pablo_answer_recorded",
        attempt_id=pablo["attempt_id"],
        cycle_index=cycle_index,
        modality=modality_used,
        transcription=transcript.status if transcript else None,
        is_last=last,
    )
    return reply("\n\n".join(lines), pablo=pablo)


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
    return reply(text("attempt_timeout" if finished else "cycle_timeout", lang_of(state)), pablo=pablo)


async def pablo_error_node(state: AgentState) -> dict:
    record_node_invocation("pablo_error")
    logger.warning("pablo_unexpected_turn", turn=assessment_turn(state))
    return reply(text("unexpected", lang_of(state)))
