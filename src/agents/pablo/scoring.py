"""Scoring job: one structured call per answered cycle, then the feedback.

Started by the backend after the UX survey (``POST /assessment/score``), so the
survey never sees a result and the scorer never sees the survey (it is not in
the request). The model returns points per rubric component; code clamps them
to the rubric and the backend turns the attempt into a grade on the activity's
scale — no arithmetic is left to the model.

A cycle that was never answered scores 0 without a model call. A cycle whose
audio could not be transcribed (and has no text) scores 0 with
``needsReview``, which opens a review case. When the model fails a cycle after
the retries, the attempt is reported ``scoring_failed`` — never a silent zero.
"""

from __future__ import annotations

import asyncio
import json
import time

import structlog

from ...config import settings
from ...providers.registry import get_provider, resolve_model
from ...schemas.pablo import CycleScore, ScoreCycle, ScoreRequest
from ...usage import make_usage_record
from .. import backend_client
from . import feedback, jobs
from .common import fence, system_prompt
from .prompts import SCORE_PROMPT

logger = structlog.get_logger(__name__)

_NO_ANSWER = {"es": "Sin respuesta.", "en": "No answer."}
_NO_TRANSCRIPT = {
    "es": "No se pudo transcribir el audio; la revisa el profesor.",
    "en": "The audio could not be transcribed; the teacher reviews it.",
}


class ScoringFailed(Exception):
    pass


def _lang(req: ScoreRequest) -> str:
    return "en" if (req.language or "").lower().startswith("en") else "es"


def answer_text(cycle: ScoreCycle) -> str:
    response = cycle.response
    if response is None:
        return ""
    return "\n".join(p.strip() for p in (response.text, response.transcript) if p and p.strip())


def clamp(points: float, max_points: float) -> tuple[float, bool]:
    """Points inside ``[0, max_points]``; True when the model's value was outside."""
    bounded = min(max(float(points), 0.0), float(max_points))
    return bounded, bounded != float(points)


def examples_for(cycle: ScoreCycle, examples: list) -> list[dict]:
    """The teacher's calibration examples for this cycle's rubric components,
    at most ``pablo_max_calibration_examples`` each (the backend sends the most
    recent first)."""
    keys = {(c.rubric_version_id, c.rubric_component_id) for c in cycle.components}
    per_key: dict[tuple[str, str], int] = {}
    picked: list[dict] = []
    for example in examples:
        key = (example.rubric_version_id, example.rubric_component_id)
        if key not in keys or per_key.get(key, 0) >= settings.pablo_max_calibration_examples:
            continue
        per_key[key] = per_key.get(key, 0) + 1
        component_ids = [
            c.component_id
            for c in cycle.components
            if (c.rubric_version_id, c.rubric_component_id) == key
        ]
        picked.append(
            {
                "component_ids": component_ids,
                "answer": example.answer[:1500],
                "teacher_points": example.points,
                "max_points": example.max_points,
                "teacher_rationale": example.rationale[:500],
            }
        )
    return picked


def cycle_block(cycle: ScoreCycle, examples: list[dict]) -> str:
    components = [
        {
            "component_id": c.component_id,
            "part": c.part,
            "name": c.name,
            "max_points": c.max_points,
            "criteria": c.criteria,
        }
        for c in cycle.components
    ]
    questions = [{"id": q.question_snapshot_id, "text": q.text[:2000]} for q in cycle.questions]
    sections = [
        "Pregunta compuesta que vio el estudiante:\n" + cycle.composite_prompt,
        "Preguntas originales del banco (JSON):\n" + json.dumps(questions, ensure_ascii=False),
        "Componentes a calificar (JSON):\n" + json.dumps(components, ensure_ascii=False),
    ]
    if examples:
        sections.append(
            "Ejemplos de calibración del profesor (las respuestas van cercadas):\n"
            + "\n".join(
                json.dumps({k: v for k, v in e.items() if k != "answer"}, ensure_ascii=False)
                + "\n"
                + fence("EJEMPLO", e["answer"])
                for e in examples
            )
        )
    response = cycle.response
    if response and response.text and response.text.strip():
        sections.append("Respuesta escrita del estudiante:\n" + fence("RESPUESTA", response.text))
    if response and response.transcript and response.transcript.strip():
        sections.append(
            "Transcripción del audio del estudiante:\n" + fence("RESPUESTA", response.transcript)
        )
    return "\n\n".join(sections)


def _zero(cycle: ScoreCycle, rationale: str, needs_review: bool) -> list[dict]:
    return [
        {
            "cycleIndex": cycle.cycle_index,
            "componentId": c.component_id,
            "rubricComponentId": c.rubric_component_id,
            "questionSnapshotId": c.question_snapshot_id,
            "rubricVersionId": c.rubric_version_id,
            "points": 0.0,
            "maxPoints": c.max_points,
            "criterion": "",
            "rationale": rationale,
            "needsReview": needs_review,
            "clamped": False,
        }
        for c in cycle.components
    ]


def components_from(cycle: ScoreCycle, result: CycleScore) -> list[dict]:
    """The model's scores mapped onto the cycle's components, clamped. Raises
    when a component is missing — a partial score is not a score."""
    by_id = {s.component_id.strip(): s for s in result.components}
    missing = [c.component_id for c in cycle.components if c.component_id not in by_id]
    if missing:
        raise ScoringFailed(f"missing components: {missing}")
    rows = []
    for c in cycle.components:
        score = by_id[c.component_id]
        points, clamped = clamp(score.points, c.max_points)
        if clamped:
            logger.warning(
                "pablo_points_clamped", component_id=c.component_id, points=score.points, max=c.max_points
            )
        rows.append(
            {
                "cycleIndex": cycle.cycle_index,
                "componentId": c.component_id,
                "rubricComponentId": c.rubric_component_id,
                "questionSnapshotId": c.question_snapshot_id,
                "rubricVersionId": c.rubric_version_id,
                "points": points,
                "maxPoints": c.max_points,
                "criterion": score.criterion[:300],
                "rationale": score.rationale[:1500],
                "needsReview": False,
                "clamped": clamped,
            }
        )
    return rows


async def _score_cycle(config: dict, system: str, provenance: dict, cycle: ScoreCycle, examples: list):
    provider = get_provider(config)
    model = resolve_model(config)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": cycle_block(cycle, examples)},
    ]
    last_error: Exception | None = None
    usage: list = []
    for _ in range(1 + max(0, settings.pablo_score_retries)):
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(
                provider.generate_structured(
                    messages, model, CycleScore, thinking_budget=settings.pablo_thinking_score
                ),
                timeout=settings.pablo_timeout_score_seconds,
            )
            usage.append(
                make_usage_record(
                    node="pablo_score",
                    provider=provider,
                    model=model,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    provenance=provenance,
                )
            )
            return components_from(cycle, result), usage, model
        except Exception as exc:  # noqa: BLE001 — retried, then the attempt fails loudly
            last_error = exc
            logger.warning("pablo_score_cycle_failed", cycle_index=cycle.cycle_index, error=str(exc)[:300])
    raise ScoringFailed(str(last_error)[:300])


async def score_attempt(req: ScoreRequest, config: dict) -> tuple[list[dict], list, dict]:
    """(component rows, usage, provenance) for every cycle of the attempt."""
    lang = _lang(req)
    system, provenance = system_prompt(
        config, "ASSESSMENT_SCORE", SCORE_PROMPT, lang, req.agent_persona_name
    )
    rows: list[dict] = []
    usage: list = []
    model = ""
    for cycle in sorted(req.cycles, key=lambda c: c.cycle_index):
        if not cycle.components:
            continue
        if not answer_text(cycle):
            failed = bool(cycle.response and cycle.response.transcription_status == "failed")
            rows += _zero(cycle, (_NO_TRANSCRIPT if failed else _NO_ANSWER)[lang], failed)
            continue
        cycle_rows, cycle_usage, model = await _score_cycle(
            config, system, provenance, cycle, examples_for(cycle, req.calibration_examples)
        )
        rows += cycle_rows
        usage += cycle_usage
    return rows, usage, {**provenance, "model_id": model or resolve_model(config)}


async def run(req: ScoreRequest) -> None:
    config = await jobs.llm_config(req.tenant_id, req.prompts)
    base = {"scoringVersion": req.scoring_version, "mode": req.mode, "replayId": req.replay_id or None}
    try:
        rows, usage, provenance = await score_attempt(req, config)
    except ScoringFailed as exc:
        logger.error("pablo_scoring_failed", attempt_id=req.attempt_id, error=str(exc))
        await backend_client.report_assessment_score(
            req.attempt_id, {**base, "status": "scoring_failed", "error": str(exc)[:300]}
        )
        return

    await backend_client.report_assessment_score(
        req.attempt_id,
        {
            **base,
            "status": "scored",
            "components": rows,
            "provenance": provenance,
            "calibrationExamplesUsed": sum(
                len(examples_for(c, req.calibration_examples)) for c in req.cycles
            ),
            "turnUsage": usage,
        },
    )
    logger.info(
        "pablo_scored",
        attempt_id=req.attempt_id,
        mode=req.mode,
        components=len(rows),
        scoring_version=req.scoring_version,
    )
    if req.mode != "publish":
        return
    await feedback.run(req, config, rows)
