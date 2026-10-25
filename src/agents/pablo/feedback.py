"""Academic feedback, written right after scoring (same job).

Per-outcome percentages are computed here from the clamped points, so the
model narrates numbers it was given and cannot invent its own. The backend
stores the result as ``AttemptFeedback`` and posts ``message`` into the
attempt's conversation. A failure is reported as such: the grade is already
published, so the student sees "feedback unavailable" rather than nothing.
"""

from __future__ import annotations

import asyncio
import json
import time

import structlog

from ...config import settings
from ...providers.registry import get_provider, resolve_model
from ...schemas.pablo import AttemptFeedbackResult, ScoreRequest
from ...usage import make_usage_record
from .. import backend_client
from .common import fence, json_contract, system_prompt
from .prompts import FEEDBACK_PROMPT

logger = structlog.get_logger(__name__)

_HEADINGS = {
    "es": {
        "strengths": "Fortalezas",
        "improve": "Para mejorar",
        "outcomes": "Por resultado de aprendizaje",
        "study": "Recomendaciones de estudio",
    },
    "en": {
        "strengths": "Strengths",
        "improve": "Areas for improvement",
        "outcomes": "By learning outcome",
        "study": "Study recommendations",
    },
}


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


def feedback_block(req: ScoreRequest, rows: list[dict], outcomes: dict[str, dict]) -> str:
    by_cycle: dict[int, list[dict]] = {}
    for row in rows:
        by_cycle.setdefault(row["cycleIndex"], []).append(row)
    sections = ["Resultados de aprendizaje (JSON):\n" + json.dumps(
        [{"learning_outcome_id": k, **v} for k, v in outcomes.items()], ensure_ascii=False
    )]
    reached = 0
    for cycle in sorted(req.cycles, key=lambda c: c.cycle_index):
        answer = "\n".join(
            p for p in ((cycle.response.text, cycle.response.transcript) if cycle.response else ()) if p
        )
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
        )
    total = req.total_cycles or len(req.cycles)
    if reached < total:
        sections.append(f"Preguntas sin responder: {total - reached} de {total}.")
    return "\n\n".join(sections)


def render(result: AttemptFeedbackResult, outcomes: dict[str, dict], lang: str) -> str:
    h = _HEADINGS[lang]
    lines = [result.overall_summary.strip()]

    def bullets(title: str, items: list[str]) -> None:
        items = [i.strip() for i in items if i and i.strip()]
        if items:
            lines.append(f"{title}:\n" + "\n".join(f"- {i}" for i in items))

    bullets(h["strengths"], result.strengths)
    bullets(h["improve"], result.areas_for_improvement)
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


async def run(req: ScoreRequest, config: dict, rows: list[dict]) -> None:
    lang = "en" if (req.language or "").lower().startswith("en") else "es"
    outcomes = outcome_percentages(req, rows)
    system, provenance = system_prompt(
        config, "ASSESSMENT_FEEDBACK", FEEDBACK_PROMPT, lang, req.agent_persona_name
    )
    base = {"scoringVersion": req.scoring_version}
    try:
        provider = get_provider(config)
        model = resolve_model(config)
        started = time.monotonic()
        result = await asyncio.wait_for(
            provider.generate_structured(
                [
                    {"role": "system", "content": system + json_contract(AttemptFeedbackResult)},
                    {"role": "user", "content": feedback_block(req, rows, outcomes)},
                ],
                model,
                AttemptFeedbackResult,
            ),
            timeout=settings.pablo_timeout_feedback_seconds,
        )
        usage = [
            make_usage_record(
                node="pablo_feedback",
                provider=provider,
                model=model,
                latency_ms=int((time.monotonic() - started) * 1000),
                provenance=provenance,
            )
        ]
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
                "perOutcomeFeedback": per_outcome,
                "studyRecommendations": result.study_recommendations[:3],
            },
            "message": render(result, outcomes, lang),
            "provenance": {**provenance, "model_id": model},
            "turnUsage": usage,
        },
    )
    logger.info("pablo_feedback_ready", attempt_id=req.attempt_id)
