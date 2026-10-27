"""Academic feedback, written right after scoring (same job).

Per-outcome percentages are computed here from the clamped points, so the
model narrates numbers it was given and cannot invent its own. The backend
stores the result as ``AttemptFeedback`` and posts ``message`` into the
attempt's conversation. A failure is reported as such: the grade is already
published, so the student sees "feedback unavailable" rather than nothing.

Per question, the student reads the verdict (decided by scoring, attached
here in code — never by this model) and a formative comment. The teacher's
reference reaches this model so it can explain what was missing, never to be
repeated: if the text copies 6+ consecutive words of a reference
(``copied_span``) it is asked once more with a warning, and then written
without the references at all (``pablo_feedback_reference_leak``).
"""

from __future__ import annotations

import asyncio
import json
import time

import structlog

from ...config import settings
from ...providers.registry import get_provider, resolve_model
from ...observability import pablo_feedback_reference_leaks_total
from ...schemas.pablo import AttemptFeedbackResult, ScoreRequest
from ...usage import make_usage_record
from .. import backend_client
from .common import answer_text, copied_span, fence, json_contract, reference_rule, system_prompt
from .prompts import FEEDBACK_PROMPT

logger = structlog.get_logger(__name__)

_HEADINGS = {
    "es": {
        "strengths": "Fortalezas",
        "improve": "Para mejorar",
        "questions": "Por pregunta",
        "question": "Pregunta",
        "outcomes": "Por resultado de aprendizaje",
        "study": "Recomendaciones de estudio",
    },
    "en": {
        "strengths": "Strengths",
        "improve": "Areas for improvement",
        "questions": "By question",
        "question": "Question",
        "outcomes": "By learning outcome",
        "study": "Study recommendations",
    },
}


_VERDICT_LABELS = {
    "es": {"correct": "Correcta", "partial": "Parcialmente correcta", "incorrect": "Incorrecta"},
    "en": {"correct": "Correct", "partial": "Partially correct", "incorrect": "Incorrect"},
}
_NO_ANSWER = {"es": "No respondiste esta pregunta.", "en": "You did not answer this question."}
_NO_COPY = (
    "\n\nAVISO: tu borrador anterior copiaba frases de la referencia del profesor. "
    "Reescríbelo sin citar ni parafrasear ninguna frase de REFERENCIA."
)


def outcome_percentages(req: ScoreRequest, rows: list[dict]) -> dict[str, dict]:
    """{outcome_id: {name, score_pct}} from the scored rows. An unreached cycle
    counts: its outcome has 0 points over its components' maximum."""
    cycle_outcome = {c.cycle_index: c for c in req.cycles}
    totals: dict[str, list[float]] = {}
    names: dict[str, str] = {}
    for row in rows:
        cycle = cycle_outcome.get(row["cycleIndex"])
        if cycle is None or not cycle.learning_outcome_id:
            continue
        got = totals.setdefault(cycle.learning_outcome_id, [0.0, 0.0])
        got[0] += row["points"]
        got[1] += row["maxPoints"]
        names[cycle.learning_outcome_id] = cycle.learning_outcome_name
    return {
        oid: {"name": names[oid], "score_pct": round(100 * p / m, 1) if m else 0.0}
        for oid, (p, m) in totals.items()
    }


def _parts(cycle) -> dict[str, str]:
    parts: dict[str, str] = {}
    for c in cycle.components:
        parts.setdefault(c.question_snapshot_id, c.part)
    return parts


def feedback_block(
    req: ScoreRequest,
    rows: list[dict],
    outcomes: dict[str, dict],
    verdicts: list[dict] | None = None,
    include_references: bool = True,
) -> str:
    verdict_of = {(v["cycleIndex"], v["questionSnapshotId"]): v for v in verdicts or []}
    by_cycle: dict[int, list[dict]] = {}
    for row in rows:
        by_cycle.setdefault(row["cycleIndex"], []).append(row)
    sections = ["Resultados de aprendizaje (JSON):\n" + json.dumps(
        [{"learning_outcome_id": k, **v} for k, v in outcomes.items()], ensure_ascii=False
    )]
    reached = 0
    for cycle in sorted(req.cycles, key=lambda c: c.cycle_index):
        answer = answer_text(cycle)
        reached += 1 if answer.strip() else 0
        scored = [
            {k: r[k] for k in ("componentId", "points", "maxPoints", "criterion", "rationale")}
            for r in by_cycle.get(cycle.cycle_index, [])
        ]
        sections.append(
            f"Pregunta {cycle.cycle_index + 1} — {cycle.learning_outcome_name}:\n"
            + cycle.composite_prompt
            + "\n"
            + fence("RESPUESTA", answer or "(sin respuesta)")
            + "\nCalificación por componente (JSON):\n"
            + json.dumps(scored, ensure_ascii=False)
            + _questions_block(cycle, verdict_of, include_references)
        )
    total = req.total_cycles or len(req.cycles)
    if reached < total:
        sections.append(f"Preguntas sin responder: {total - reached} de {total}.")
    return "\n\n".join(sections)


def _questions_block(cycle, verdict_of: dict, include_references: bool) -> str:
    parts = _parts(cycle)
    lines = []
    for q in cycle.questions:
        v = verdict_of.get((cycle.cycle_index, q.question_snapshot_id)) or {}
        lines.append(
            json.dumps(
                {
                    "cycle_index": cycle.cycle_index,
                    "question_snapshot_id": q.question_snapshot_id,
                    "part": parts.get(q.question_snapshot_id, ""),
                    "verdict": v.get("verdict", ""),
                    "reason": v.get("reason", ""),
                },
                ensure_ascii=False,
            )
        )
        if include_references and q.grader_info.strip():
            lines.append(fence("REFERENCIA", q.grader_info[:2000]))
    return "\nPreguntas de este ciclo (JSON):\n" + "\n".join(lines) if lines else ""


def per_question(req: ScoreRequest, result: AttemptFeedbackResult, verdicts: list[dict], lang: str) -> list[dict]:
    """The model's comments on the attempt's real questions, with the part
    and the verdict attached in code. A blank cycle gets the no-answer line."""
    comments = {}
    for item in result.per_question_feedback:
        if item.comment.strip():
            comments.setdefault((item.cycle_index, item.question_snapshot_id.strip()), item.comment.strip())
    verdict_of = {(v["cycleIndex"], v["questionSnapshotId"]): v["verdict"] for v in verdicts}
    out = []
    for cycle in sorted(req.cycles, key=lambda c: c.cycle_index):
        if not cycle.components:
            continue
        parts = _parts(cycle)
        blank = not answer_text(cycle).strip()
        for q in cycle.questions:
            key = (cycle.cycle_index, q.question_snapshot_id)
            comment = _NO_ANSWER[lang] if blank else comments.get(key)
            if not comment:
                continue
            out.append(
                {
                    "cycleIndex": cycle.cycle_index,
                    "questionSnapshotId": q.question_snapshot_id,
                    "part": parts.get(q.question_snapshot_id, ""),
                    "verdict": verdict_of.get(key),
                    "comment": comment[:1000],
                    "_parts": len(cycle.questions),
                }
            )
    return out


def _question_line(item: dict, lang: str) -> str:
    n = item["cycleIndex"] + 1
    head = f"{_HEADINGS[lang]['question']} {n}"
    if item["_parts"] > 1 and item["part"]:
        head += f" ({item['part']})"
    label = _VERDICT_LABELS[lang].get(item["verdict"] or "")
    return f"{head} — {label}: {item['comment']}" if label else f"{head}: {item['comment']}"


def render(
    result: AttemptFeedbackResult, outcomes: dict[str, dict], lang: str, questions: list[dict] | None = None
) -> str:
    h = _HEADINGS[lang]
    lines = [result.overall_summary.strip()]

    def bullets(title: str, items: list[str]) -> None:
        items = [i.strip() for i in items if i and i.strip()]
        if items:
            lines.append(f"{title}:\n" + "\n".join(f"- {i}" for i in items))

    bullets(h["strengths"], result.strengths)
    bullets(h["improve"], result.areas_for_improvement)
    bullets(h["questions"], [_question_line(q, lang) for q in questions or []])
    bullets(
        h["outcomes"],
        [
            f"{outcomes[o.learning_outcome_id]['name']} ({outcomes[o.learning_outcome_id]['score_pct']:g}%): {o.comment}"
            for o in result.per_outcome_feedback
            if o.learning_outcome_id in outcomes
        ],
    )
    bullets(h["study"], result.study_recommendations[:3])
    return "\n\n".join(lines)


def _full_text(result: AttemptFeedbackResult, message: str) -> str:
    return "\n".join(
        [
            message,
            result.overall_summary,
            *result.strengths,
            *result.areas_for_improvement,
            *(q.comment for q in result.per_question_feedback),
            *(o.comment for o in result.per_outcome_feedback),
            *result.study_recommendations,
        ]
    )


def leaks_reference(req: ScoreRequest, text: str) -> bool:
    """True when the text copies 6+ consecutive words of any reference."""
    return any(
        copied_span(text, q.grader_info) for c in req.cycles for q in c.questions if q.grader_info.strip()
    )


async def run(req: ScoreRequest, config: dict, rows: list[dict], verdicts: list[dict] | None = None) -> None:
    lang = "en" if (req.language or "").lower().startswith("en") else "es"
    verdicts = verdicts or []
    outcomes = outcome_percentages(req, rows)
    system, provenance = system_prompt(
        config, "ASSESSMENT_FEEDBACK", FEEDBACK_PROMPT, lang, req.agent_persona_name
    )
    system += reference_rule("feedback", lang) + json_contract(AttemptFeedbackResult)
    base = {"scoringVersion": req.scoring_version}
    usage: list = []
    model = ""

    async def write(user: str) -> tuple[AttemptFeedbackResult, list[dict], str]:
        provider = get_provider(config)
        started = time.monotonic()
        result = await asyncio.wait_for(
            provider.generate_structured(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                model,
                AttemptFeedbackResult,
            ),
            timeout=settings.pablo_timeout_feedback_seconds,
        )
        usage.append(
            make_usage_record(
                node="pablo_feedback",
                provider=provider,
                model=model,
                latency_ms=int((time.monotonic() - started) * 1000),
                provenance=provenance,
            )
        )
        questions = per_question(req, result, verdicts, lang)
        return result, questions, render(result, outcomes, lang, questions)

    try:
        model = resolve_model(config)
        user = feedback_block(req, rows, outcomes, verdicts)
        result, questions, message = await write(user)
        if leaks_reference(req, _full_text(result, message)):
            result, questions, message = await write(user + _NO_COPY)
            if leaks_reference(req, _full_text(result, message)):
                logger.warning("pablo_feedback_reference_leak", attempt_id=req.attempt_id)
                pablo_feedback_reference_leaks_total.inc()
                result, questions, message = await write(
                    feedback_block(req, rows, outcomes, verdicts, include_references=False)
                )
    except Exception as exc:  # noqa: BLE001 — the grade stands without it
        logger.error("pablo_feedback_failed", attempt_id=req.attempt_id, error=str(exc)[:300])
        await backend_client.report_assessment_feedback(
            req.attempt_id, {**base, "status": "failed", "error": str(exc)[:300]}
        )
        return

    per_outcome = [
        {
            "learningOutcomeId": o.learning_outcome_id,
            "outcomeName": outcomes[o.learning_outcome_id]["name"],
            "scorePct": outcomes[o.learning_outcome_id]["score_pct"],
            "comment": o.comment,
        }
        for o in result.per_outcome_feedback
        if o.learning_outcome_id in outcomes
    ]
    await backend_client.report_assessment_feedback(
        req.attempt_id,
        {
            **base,
            "status": "ready",
            "feedback": {
                "overallSummary": result.overall_summary,
                "strengths": result.strengths,
                "areasForImprovement": result.areas_for_improvement,
                "perQuestionFeedback": [
                    {k: v for k, v in q.items() if not k.startswith("_")} for q in questions
                ],
                "perOutcomeFeedback": per_outcome,
                "studyRecommendations": result.study_recommendations[:3],
            },
            "message": message,
            "provenance": {**provenance, "model_id": model},
            "turnUsage": usage,
        },
    )
    logger.info("pablo_feedback_ready", attempt_id=req.attempt_id)
