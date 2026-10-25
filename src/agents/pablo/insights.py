"""Class report narrative and improvement proposals (``POST /assessment/insights``).

Teacher-facing, never a chat. The backend computes every number and sends
them as facts; the model writes the narrative and proposes improvements. A
proposal is only ever a proposal: the backend stores it as ``proposed`` and a
human (the teacher in Moodle, or an admin for prompt drafts) accepts, edits or
rejects it.

Guards in code, not in the prompt:
- a rubric clarification is kept only for a component the teacher adjusted at
  least ``MIN_SAME_DIRECTION`` times in the same direction (human labels only);
- ids must be ones the request carried;
- prompt guidance sharing a passage with a teacher rationale is dropped, since
  a rationale may quote a student and guidance reaches every assessment.
"""

from __future__ import annotations

import asyncio
import json
import time

import structlog

from ...config import settings
from ...providers.registry import get_provider, resolve_model
from ...schemas.pablo import ClassReportNarrative, ImprovementProposals, InsightsRequest
from ...usage import make_usage_record
from .. import backend_client
from . import jobs
from .common import system_prompt
from .prompts import CLASS_REPORT_PROMPT, IMPROVE_PROMPT

logger = structlog.get_logger(__name__)

MIN_SAME_DIRECTION = 3
MAX_GUIDANCE = 5
_QUOTE_WINDOW = 30


def adjusted_components(req: InsightsRequest) -> dict[tuple[str, str], int]:
    """{(rubric_version_id, rubric_component_id): decisions} for components the
    teacher moved ≥ MIN_SAME_DIRECTION times in one direction."""
    ups: dict[tuple[str, str], int] = {}
    downs: dict[tuple[str, str], int] = {}
    for d in req.decisions:
        key = (d.rubric_version_id, d.rubric_component_id)
        if d.teacher_points > d.pablo_points:
            ups[key] = ups.get(key, 0) + 1
        elif d.teacher_points < d.pablo_points:
            downs[key] = downs.get(key, 0) + 1
    eligible: dict[tuple[str, str], int] = {}
    for key in set(ups) | set(downs):
        count = max(ups.get(key, 0), downs.get(key, 0))
        if count >= MIN_SAME_DIRECTION:
            eligible[key] = count
    return eligible


def quotes_rationale(guidance: str, rationales: list[str]) -> bool:
    folded = guidance.lower()
    for rationale in rationales:
        text = rationale.lower()
        for start in range(0, max(1, len(text) - _QUOTE_WINDOW + 1), 10):
            window = text[start : start + _QUOTE_WINDOW]
            if len(window) == _QUOTE_WINDOW and window in folded:
                return True
    return False


def filter_proposals(req: InsightsRequest, result: ImprovementProposals) -> dict:
    eligible = adjusted_components(req)
    known_questions = {
        str(q.get("question_snapshot_id")) for q in req.question_analysis if isinstance(q, dict)
    }
    rationales = [d.teacher_rationale for d in req.decisions if d.teacher_rationale]
    clarifications = [
        {
            "rubricVersionId": c.rubric_version_id,
            "rubricComponentId": c.component_id,
            "proposedCriteria": c.proposed_criteria.strip(),
            "reason": c.reason.strip(),
            "evidenceDecisions": eligible[(c.rubric_version_id, c.component_id)],
        }
        for c in result.rubric_clarifications
        if (c.rubric_version_id, c.component_id) in eligible and c.proposed_criteria.strip()
    ]
    flags = [
        {"questionSnapshotId": f.question_snapshot_id, "reason": f.reason.strip()}
        for f in result.question_flags
        if f.question_snapshot_id in known_questions
    ]
    guidance = [
        g.strip()[:300]
        for g in result.prompt_guidance
        if g.strip() and not quotes_rationale(g, rationales)
    ][:MAX_GUIDANCE]
    return {"rubricClarifications": clarifications, "questionFlags": flags, "promptGuidance": guidance}


async def _structured(config, system, user, schema, node, provenance):
    provider = get_provider(config)
    model = resolve_model(config)
    started = time.monotonic()
    result = await asyncio.wait_for(
        provider.generate_structured(
            [{"role": "system", "content": system}, {"role": "user", "content": user}], model, schema
        ),
        timeout=settings.pablo_timeout_feedback_seconds,
    )
    usage = make_usage_record(
        node=node,
        provider=provider,
        model=model,
        latency_ms=int((time.monotonic() - started) * 1000),
        provenance=provenance,
    )
    return result, usage, model


def report_block(req: InsightsRequest) -> str:
    sections = ["Cifras del grupo (JSON, ya calculadas):\n" + json.dumps(req.stats, ensure_ascii=False)]
    if req.question_analysis:
        sections.append("Análisis de preguntas (JSON):\n" + json.dumps(req.question_analysis, ensure_ascii=False))
    for outcome, rationales in req.rationales_by_outcome.items():
        sections.append(
            f"Justificaciones de calificación — {outcome}:\n"
            + "\n".join(f"- {r[:400]}" for r in rationales[:40])
        )
    if req.ux_comments:
        sections.append("Comentarios sobre la experiencia:\n" + "\n".join(f"- {c[:300]}" for c in req.ux_comments[:60]))
    return "\n\n".join(sections)


def improve_block(req: InsightsRequest) -> str:
    eligible = adjusted_components(req)
    decisions = [d.model_dump() for d in req.decisions]
    rubrics = [
        r.model_dump()
        for r in req.rubrics
        if (r.rubric_version_id, r.rubric_component_id) in eligible
    ]
    return "\n\n".join(
        [
            "Decisiones del profesor (JSON):\n" + json.dumps(decisions, ensure_ascii=False),
            "Componentes con ajustes repetidos y su rúbrica (JSON):\n" + json.dumps(rubrics, ensure_ascii=False),
            "Análisis de preguntas (JSON):\n" + json.dumps(req.question_analysis, ensure_ascii=False),
        ]
    )


async def run(req: InsightsRequest) -> None:
    lang = "en" if (req.language or "").lower().startswith("en") else "es"
    config = await jobs.llm_config(req.tenant_id, req.prompts)
    usage: list = []
    payload: dict = {"reportId": req.report_id}

    system, provenance = system_prompt(config, "ASSESSMENT_CLASS_REPORT", CLASS_REPORT_PROMPT, lang)
    try:
        narrative, record, model = await _structured(
            config, system, report_block(req), ClassReportNarrative, "pablo_class_report", provenance
        )
        usage.append(record)
        ux_themes = narrative.ux_themes if req.ux_comments else []
        payload["narrative"] = {
            "summary": narrative.summary,
            "misconceptions": narrative.misconceptions,
            "teachingRecommendations": narrative.teaching_recommendations,
            "uxThemes": ux_themes,
        }
        payload["provenance"] = {**provenance, "model_id": model}
    except Exception as exc:  # noqa: BLE001 — the numbers still make a report
        logger.error("pablo_class_report_failed", report_id=req.report_id, error=str(exc)[:300])
        payload["narrativeError"] = str(exc)[:300]

    if req.include_proposals and (adjusted_components(req) or req.question_analysis):
        system, improve_provenance = system_prompt(config, "ASSESSMENT_IMPROVE", IMPROVE_PROMPT, lang)
        try:
            proposals, record, _ = await _structured(
                config, system, improve_block(req), ImprovementProposals, "pablo_improve", improve_provenance
            )
            usage.append(record)
            payload["proposals"] = filter_proposals(req, proposals)
        except Exception as exc:  # noqa: BLE001
            logger.error("pablo_improve_failed", report_id=req.report_id, error=str(exc)[:300])
            payload["proposalsError"] = str(exc)[:300]

    payload["status"] = "failed" if "narrativeError" in payload else "ready"
    payload["turnUsage"] = usage
    await backend_client.report_assessment_insights(req.assessment_id, payload)
    logger.info(
        "pablo_insights_reported",
        report_id=req.report_id,
        status=payload["status"],
        proposals={k: len(v) for k, v in (payload.get("proposals") or {}).items()},
    )
