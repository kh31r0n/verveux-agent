"""Scoring job: one structured call per answered cycle, then the feedback.

Started by the backend after the UX survey (``POST /assessment/score``), so the
survey never sees a result and the scorer never sees the survey (it is not in
the request). The model returns points per rubric component; code clamps them
to the rubric and the backend turns the attempt into a grade on the activity's
scale — no arithmetic is left to the model.

A question with the teacher's "Information for graders" (``grader_info``) is
judged against it first — a verdict ``correct | partial | incorrect`` with a
reason for the teacher — and that verdict is the ground truth the rubric is
applied on. A question without one gets ``no_reference`` in code. The
reference only ever reaches the model fenced (``<<<REFERENCIA``).

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
from .common import answer_text, fence, json_contract, reference_rule, system_prompt
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
    sections = [
        "Pregunta compuesta que vio el estudiante:\n" + cycle.composite_prompt,
        "Preguntas originales del banco:\n" + "\n\n".join(_question_lines(cycle)),
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
    _response_sections(sections, cycle.response, "")
    if cycle.followup:
        sections.append("Repregunta que hizo el evaluador:\n" + cycle.followup.prompt)
        if cycle.followup.response:
            _response_sections(sections, cycle.followup.response, " a la repregunta")
        else:
            sections.append("El estudiante no respondió la repregunta.")
    return "\n\n".join(sections)


def question_parts(cycle: ScoreCycle) -> dict[str, str]:
    """{question_snapshot_id: "Parte A"} from the cycle's components."""
    parts: dict[str, str] = {}
    for c in cycle.components:
        parts.setdefault(c.question_snapshot_id, c.part)
    return parts


def _question_lines(cycle: ScoreCycle) -> list[str]:
    parts = question_parts(cycle)
    lines = []
    for q in cycle.questions:
        head = json.dumps(
            {"id": q.question_snapshot_id, "part": parts.get(q.question_snapshot_id, ""), "text": q.text[:2000]},
            ensure_ascii=False,
        )
        if q.grader_info.strip():
            lines.append(
                f"{head}\nReferencia del profesor para {q.question_snapshot_id}:\n"
                + fence("REFERENCIA", q.grader_info[:3000])
            )
        else:
            lines.append(f"{head}\nSin referencia (calificar solo con la rúbrica).")
    return lines


def _verdict(cycle: ScoreCycle, question_id: str, verdict: str, reason: str) -> dict:
    return {
        "cycleIndex": cycle.cycle_index,
        "questionSnapshotId": question_id,
        "verdict": verdict,
        "reason": reason[:1000],
    }


def verdicts_from(cycle: ScoreCycle, result: CycleScore) -> list[dict]:
    """One verdict per question of the cycle. ``no_reference`` is decided in
    code; a question with a reference the model left unjudged raises (retried,
    like a missing component). Ids the cycle did not ask are dropped."""
    by_id = {}
    for v in result.verdicts:
        by_id.setdefault(v.question_snapshot_id.strip(), v)
    out, missing = [], []
    for q in cycle.questions:
        if not q.grader_info.strip():
            out.append(_verdict(cycle, q.question_snapshot_id, "no_reference", ""))
        elif (v := by_id.get(q.question_snapshot_id)) is not None:
            out.append(_verdict(cycle, q.question_snapshot_id, v.verdict, v.reason))
        else:
            missing.append(q.question_snapshot_id)
    if missing:
        raise ScoringFailed(f"missing verdicts: {missing}")
    return out


def unanswered_verdicts(cycle: ScoreCycle, lang: str) -> list[dict]:
    """A cycle left blank: incorrect against every reference, no model call."""
    return [
        _verdict(cycle, q.question_snapshot_id, "incorrect", _NO_ANSWER[lang])
        if q.grader_info.strip()
        else _verdict(cycle, q.question_snapshot_id, "no_reference", "")
        for q in cycle.questions
    ]


def _response_sections(sections: list[str], response, suffix: str) -> None:
    if response and response.text and response.text.strip():
        sections.append(f"Respuesta escrita del estudiante{suffix}:\n" + fence("RESPUESTA", response.text))
    if response and response.transcript and response.transcript.strip():
        sections.append(
            f"Transcripción del audio del estudiante{suffix}:\n" + fence("RESPUESTA", response.transcript)
        )


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


async def _score_cycle(
    config: dict, system: str, provenance: dict, cycle: ScoreCycle, examples: list, lang: str = "es"
):
    provider = get_provider(config)
    model = resolve_model(config)
    messages = [
        {"role": "system", "content": system + reference_rule("score", lang) + json_contract(CycleScore)},
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
            return components_from(cycle, result), verdicts_from(cycle, result), usage, model
        except Exception as exc:  # noqa: BLE001 — retried, then the attempt fails loudly
            last_error = exc
            logger.warning("pablo_score_cycle_failed", cycle_index=cycle.cycle_index, error=str(exc)[:300])
    raise ScoringFailed(str(last_error)[:300])


async def score_attempt(req: ScoreRequest, config: dict) -> tuple[list[dict], list[dict], list, dict]:
    """(component rows, verdicts, usage, provenance) for every cycle of the attempt."""
    lang = _lang(req)
    system, provenance = system_prompt(
        config, "ASSESSMENT_SCORE", SCORE_PROMPT, lang, req.agent_persona_name
    )
    rows: list[dict] = []
    verdicts: list[dict] = []
    usage: list = []
    model = ""
    for cycle in sorted(req.cycles, key=lambda c: c.cycle_index):
        if not cycle.components:
            continue
        if not answer_text(cycle):
            failed = bool(cycle.response and cycle.response.transcription_status == "failed")
            rows += _zero(cycle, (_NO_TRANSCRIPT if failed else _NO_ANSWER)[lang], failed)
            # An audio that could not be transcribed is the teacher's to judge.
            if not failed:
                verdicts += unanswered_verdicts(cycle, lang)
            continue
        cycle_rows, cycle_verdicts, cycle_usage, model = await _score_cycle(
            config, system, provenance, cycle, examples_for(cycle, req.calibration_examples), lang
        )
        rows += cycle_rows
        verdicts += cycle_verdicts
        usage += cycle_usage
    return rows, verdicts, usage, {**provenance, "model_id": model or resolve_model(config)}


async def run(req: ScoreRequest) -> None:
    config = await jobs.llm_config(req.tenant_id, req.prompts)
    base = {"scoringVersion": req.scoring_version, "mode": req.mode, "replayId": req.replay_id or None}
    try:
        rows, verdicts, usage, provenance = await score_attempt(req, config)
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
            # A replay only checks points (a prompt draft); nothing else is kept.
            **({"verdicts": verdicts} if req.mode == "publish" else {}),
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
    await feedback.run(req, config, rows, verdicts)
