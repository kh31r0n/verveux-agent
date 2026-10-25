"""Cycle composition: 2–3 authorised questions → one composite prompt.

The backend decides which questions are eligible (the next learning outcome,
none already used); the model only chooses among them and rewrites them as
open prompts. Everything the scoring later depends on — which question each
part is, which rubric components it carries — is derived here in code from the
candidates, never from the model's text.

Candidate shape (from ``POST /internal/assessment/attempts/:id/next-cycle``)::

    {snapshot_id, qtype, text, difficulty, topic,
     rubric: {version_id, components: [{id, name, max_points, criteria}]}}
"""

from __future__ import annotations

import json

from ...schemas.pablo import ComposedCycle, ComposedPart
from .texts import part_label, text

MAX_PARTS = 3
MIN_PARTS = 2


def usable_candidates(candidates: list) -> list[dict]:
    """Candidates with an id, a text and a rubric with at least one component."""
    return [
        c
        for c in candidates or []
        if isinstance(c, dict)
        and c.get("snapshot_id")
        and str(c.get("text") or "").strip()
        and ((c.get("rubric") or {}).get("components") or [])
    ]


def candidates_block(outcome: dict, candidates: list[dict]) -> str:
    """The candidates as data for the composition prompt (no rubric: the
    model must not tailor the question to the scoring)."""
    rows = [
        {
            "question_snapshot_id": c["snapshot_id"],
            "type": c.get("qtype", ""),
            "difficulty": c.get("difficulty", ""),
            "topic": c.get("topic", ""),
            "text": str(c.get("text", ""))[:2000],
        }
        for c in candidates
    ]
    return (
        f"Resultado de aprendizaje: {outcome.get('name', '')}\n"
        "Preguntas candidatas (JSON):\n" + json.dumps(rows, ensure_ascii=False)
    )


def validated_parts(result: ComposedCycle | None, candidates: list[dict]) -> list[ComposedPart]:
    """The model's parts that name a real, distinct candidate, at most 3.

    Falls back to the first candidates verbatim when the model picked fewer
    than it could have — a cycle never silently shrinks to one part because a
    model returned an unknown id.
    """
    by_id = {c["snapshot_id"]: c for c in candidates}
    parts: list[ComposedPart] = []
    seen: set[str] = set()
    for part in (result.parts if result else []):
        sid = part.question_snapshot_id.strip()
        if sid in by_id and sid not in seen and part.prompt.strip():
            parts.append(ComposedPart(question_snapshot_id=sid, prompt=part.prompt.strip()))
            seen.add(sid)
        if len(parts) == MAX_PARTS:
            break
    wanted = min(MIN_PARTS, len(candidates))
    for c in candidates:
        if len(parts) >= wanted:
            break
        if c["snapshot_id"] not in seen:
            parts.append(
                ComposedPart(question_snapshot_id=c["snapshot_id"], prompt=str(c["text"]).strip())
            )
            seen.add(c["snapshot_id"])
    return parts


def render(intro: str, parts: list[ComposedPart], cycle_index: int, total: int, lang: str) -> str:
    header = text("question", lang, n=cycle_index + 1, total=total)
    lines = [header]
    if intro.strip():
        lines.append(intro.strip())
    for index, part in enumerate(parts):
        lines.append(f"{part_label(index, lang)}: {part.prompt}")
    return "\n\n".join(lines)


def component_map(parts: list[ComposedPart], candidates: list[dict], lang: str) -> list[dict]:
    """One entry per rubric component of every chosen question.

    ``component_id`` is ``{snapshot}:{rubric component}``: a rubric linked to a
    learning outcome is shared by several questions, so the rubric component's
    own id is not unique within a cycle.
    """
    by_id = {c["snapshot_id"]: c for c in candidates}
    entries: list[dict] = []
    for index, part in enumerate(parts):
        rubric = by_id[part.question_snapshot_id].get("rubric") or {}
        for component in rubric.get("components") or []:
            entries.append(
                {
                    "component_id": f"{part.question_snapshot_id}:{component.get('id')}",
                    "rubric_component_id": str(component.get("id")),
                    "question_snapshot_id": part.question_snapshot_id,
                    "rubric_version_id": str(rubric.get("version_id") or ""),
                    "part": part_label(index, lang),
                    "name": str(component.get("name") or ""),
                    "max_points": float(component.get("max_points") or 0),
                }
            )
    return entries
