"""The follow-up question: one per cycle, asked to every student.

Asking only the students whose answer misses something would tell them so,
and give only them a second chance; asking everyone keeps both equal. The
model says which rubric component the answer covers least (never points —
scoring stays the job's) and words one open question. Code decides what
reaches the student:

- the target must be one of the cycle's components;
- the question is short and is a question;
- **leak guard**: a content word that appears in the rubric (component names
  or criteria) but neither in the question the student saw nor in their
  answer would hand them what is missing, so the question is dropped;
- whenever the model fails or is refused, a generic question in code goes
  out instead — every student still gets one.

This is the only chat-time call that sees the rubric; it never sees the
teacher's calibration examples.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import unicodedata

import structlog
from langchain_core.runnables import RunnableConfig

from ...config import settings
from ...providers.registry import get_provider, resolve_model
from ...schemas.pablo import FollowupPlan
from ...usage import make_usage_record
from . import echo
from .common import fence, json_contract, system_prompt
from .prompts import FOLLOWUP_PROMPT
from .texts import text

logger = structlog.get_logger(__name__)

MAX_CHARS = 300
_WORD = re.compile(r"[a-zñ]{5,}")
# Words any follow-up may use whatever the rubric says: asking for an
# example or a reason names no content.
_GENERIC = frozenset(
    """
    puedes podrias podria desarrollar desarrollarlo desarrollala explicar explica explicalo explicame
    explicarlo ejemplo ejemplos justificar justifica justificalo razon razones porque consecuencia
    consecuencias practica practicas respuesta respuestas pregunta preguntas parte partes mencionaste
    dijiste dices comentaste afirmaste profundizar profundiza detalle detalles concreto concreta
    situacion caso casos relacion relaciona relacionas cuenta contar cuentame sobre entonces ademas
    could would explain example examples reason reasons because consequence practice answer question
    questions mentioned said detail details concrete situation further could develop expand
    """.split()
)


def fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def words(value: str) -> set[str]:
    return set(_WORD.findall(fold(value)))


def leaked_words(question: str, rubric: str, allowed: str) -> set[str]:
    """Content words of the question taken from the rubric and found nowhere
    in what the student saw or said."""
    return (words(question) & words(rubric)) - words(allowed) - _GENERIC


def components_block(cycle: dict) -> list[dict]:
    criteria = cycle.get("rubric_criteria") or {}
    return [
        {
            "component_id": c["component_id"],
            "part": c.get("part", ""),
            "name": c.get("name", ""),
            "criteria": str(criteria.get(c["component_id"], ""))[:1000],
        }
        for c in cycle.get("component_map") or []
    ]


def generic_question(cycle: dict, target: str | None, lang: str) -> str:
    part = next(
        (c.get("part") for c in cycle.get("component_map") or [] if c["component_id"] == target),
        None,
    ) or next((c.get("part") for c in cycle.get("component_map") or []), "")
    return text("followup_generic", lang, part=part) if part else text("followup_generic_plain", lang)


def validated(plan: FollowupPlan | None, cycle: dict, answer: str) -> tuple[str | None, str | None, str]:
    """(question | None, target | None, mode). None question = refused."""
    if plan is None:
        return None, None, "deepen"
    ids = {c["component_id"] for c in cycle.get("component_map") or []}
    target = (plan.target_component_id or "").strip() or None
    if target not in ids:
        target = None
    mode = "gap" if target else "deepen"
    question = " ".join(str(plan.question or "").split())
    if not question or len(question) > MAX_CHARS or not question.rstrip().endswith("?"):
        return None, target, mode
    rubric = " ".join(f"{c['name']} {c['criteria']}" for c in components_block(cycle))
    leaked = leaked_words(question, rubric, f"{cycle.get('composite_prompt', '')} {answer}")
    if leaked:
        logger.info("pablo_followup_leak_refused", words=sorted(leaked)[:10])
        return None, target, mode
    return question, target, mode


async def plan_for(
    config: RunnableConfig, cycle: dict, answer: str, lang: str, persona: str
) -> tuple[dict, list]:
    """({question, target_component_id, mode, quote, provenance}, usage). Never
    raises and always returns a question."""
    system, provenance = system_prompt(config, "ASSESSMENT_FOLLOWUP", FOLLOWUP_PROMPT, lang, persona)
    user = (
        "Pregunta que vio el estudiante:\n"
        + str(cycle.get("composite_prompt", ""))
        + "\n\nComponentes de la rúbrica (JSON, el estudiante no los ve):\n"
        + json.dumps(components_block(cycle), ensure_ascii=False)
        + "\n\nRespuesta del estudiante:\n"
        + fence("RESPUESTA", answer[:6000])
    )
    plan: FollowupPlan | None = None
    usage: list = []
    model = ""
    try:
        provider = get_provider(config)
        model = resolve_model(config)
        started = time.monotonic()
        plan = await asyncio.wait_for(
            provider.generate_structured(
                [
                    {"role": "system", "content": system + json_contract(FollowupPlan)},
                    {"role": "user", "content": user},
                ],
                model,
                FollowupPlan,
                thinking_budget=settings.pablo_thinking_followup,
            ),
            timeout=settings.pablo_timeout_followup_seconds,
        )
        usage = [
            make_usage_record(
                node="pablo_followup",
                provider=provider,
                model=model,
                latency_ms=int((time.monotonic() - started) * 1000),
                provenance=provenance,
            )
        ]
        if not isinstance(plan, FollowupPlan):
            plan = None
    except Exception as exc:  # noqa: BLE001 — the generic question still goes out
        logger.info("pablo_followup_failed", error=str(exc)[:300])
        plan = None

    question, target, mode = validated(plan, cycle, answer)
    fallback = question is None
    if fallback:
        question = generic_question(cycle, target, lang)
    quote = echo.validated_quote(str(plan.quote or ""), answer) if plan else None
    return (
        {
            "question": question,
            "target_component_id": target,
            "mode": mode,
            "quote": quote,
            "provenance": {**provenance, "model_id": model, **({"fallback": "generic"} if fallback else {})},
        },
        usage,
    )
