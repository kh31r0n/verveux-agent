"""The follow-up question: one per cycle, asked to every student.

Asking only the students whose answer misses something would tell them so,
and give only them a second chance; asking everyone keeps both equal. The
model says which rubric component the answer covers least (never points —
scoring stays the job's) and words one open question. Code decides what
reaches the student:

- the target must be one of the cycle's components;
- the question is short and is a question;
- **leak guard**: a content word that appears in the rubric (component names
  or criteria) or in the teacher's reference (``grader_info``) but neither in
  the question the student saw nor in their answer would hand them what is
  missing, so the question is dropped;
- a refused question is asked for again ONCE, telling the model why (too
  long, not a question, which words it may not use); a failed call is
  retried the same way;
- when both fail, a question in code built on the student's own words goes
  out («Dijiste «…». ¿Por qué lo ves así?»), and the generic one only when
  there is nothing to quote — every student still gets one.

What to ask is varied by the answer (``MOVES_RULE``, appended in code like
the reference rule): the reason behind a claim, a case that tests it, a link
between two of its ideas, a term to pin down — an example is one move among
them, never the default.

This is the only chat-time call that sees the rubric and the reference (to
aim the question at what the answer gets wrong or leaves out); it never sees
the teacher's calibration examples.
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
from .common import fence, json_contract, plain, reference_rule, system_prompt
from .prompts import FOLLOWUP_PROMPT
from .texts import text

logger = structlog.get_logger(__name__)

MAX_CHARS = 400
# The moves a follow-up may make, appended in code so a tenant prompt that
# still says "ask for an example" does not make every question the same.
MOVES_RULE = (
    "\n\nCómo formular `question` (obligatorio):\n"
    "- Parte de algo concreto que el estudiante dijo (una afirmación, un término, una relación) y "
    "construye la pregunta sobre eso, no sobre la pregunta original en general.\n"
    "- Elige el movimiento que mejor sirva a ESA respuesta: pedir el porqué de una afirmación suya; "
    "plantear una consecuencia o un caso que la ponga a prueba; pedir que conecte dos ideas que mencionó "
    "o una idea suya con la pregunta; pedir que precise un término que usó. Pedir un ejemplo es solo una "
    "opción más, úsala únicamente si es lo que más aporta.\n"
    "- Una sola pregunta, que termine en «?». Nada después del signo."
)
_REFUSALS = {
    "empty": "estaba vacía",
    "too_long": "era demasiado larga (máximo dos frases cortas)",
    "no_question": "no era una pregunta (debe ser una sola pregunta que termine en «?»)",
    "failed": "no llegó o no se pudo leer",
}
_QUOTE_WORDS = (3, 14)
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
    # One question in the cycle: no "Part A" to point at.
    if len({c.get("question_snapshot_id") for c in cycle.get("component_map") or []}) <= 1:
        return text("followup_generic_plain", lang)
    part = next(
        (c.get("part") for c in cycle.get("component_map") or [] if c["component_id"] == target),
        None,
    ) or next((c.get("part") for c in cycle.get("component_map") or []), "")
    return text("followup_generic", lang, part=part) if part else text("followup_generic_plain", lang)


def validated(
    plan: FollowupPlan | None, cycle: dict, answer: str
) -> tuple[str | None, str | None, str, str]:
    """(question | None, target | None, mode, refusal). None question =
    refused, and ``refusal`` says why (``leak:w1,w2`` names the words)."""
    if plan is None:
        return None, None, "deepen", "failed"
    ids = {c["component_id"] for c in cycle.get("component_map") or []}
    target = (plan.target_component_id or "").strip() or None
    if target not in ids:
        target = None
    mode = "gap" if target else "deepen"
    question = " ".join(plain(plan.question).split())
    # «¿…? Piénsalo bien.» keeps the question and drops the tail.
    if "?" in question and not question.endswith("?"):
        question = question[: question.rindex("?") + 1]
    if not question:
        return None, target, mode, "empty"
    if not question.endswith("?"):
        return None, target, mode, "no_question"
    if len(question) > MAX_CHARS:
        return None, target, mode, "too_long"
    rubric = " ".join(f"{c['name']} {c['criteria']}" for c in components_block(cycle))
    references = " ".join((cycle.get("grader_info") or {}).values())
    rubric = f"{rubric} {references}"
    leaked = leaked_words(question, rubric, f"{cycle.get('composite_prompt', '')} {answer}")
    if leaked:
        return None, target, mode, "leak:" + ",".join(sorted(leaked)[:10])
    return question, target, mode, ""


def refusal_note(refusal: str) -> str:
    """What the retry is told about the refused question."""
    if refusal.startswith("leak:"):
        words = refusal.removeprefix("leak:").replace(",", ", ")
        reason = (
            f"usaba palabras que el estudiante no dijo ni vio en la pregunta ({words}); "
            "no las uses ni sus sinónimos cercanos, apóyate en sus propias palabras"
        )
    else:
        reason = _REFUSALS.get(refusal, refusal)
    return f"\n\nTu repregunta anterior no sirvió: {reason}. Escribe otra."


def answer_fragment(answer: str) -> str | None:
    """A literal piece of the answer to build the fallback on: its first
    clause, 3–14 words."""
    first = re.split(r"(?<=[.;:!?])\s+|\n+", " ".join((answer or "").split()))[0]
    words = first.strip(" .,;:!?¿¡«»\"'").split()
    if len(words) < _QUOTE_WORDS[0]:
        return None
    return " ".join(words[: _QUOTE_WORDS[1]])


def quoted_question(quote: str | None, cycle: dict, lang: str) -> str | None:
    """A follow-up in code on the student's own words, the template chosen by
    the cycle's position (never by the answer)."""
    if not quote:
        return None
    index = int(cycle.get("cycle_index") or 0) % 3
    return text(f"followup_quoted_{index}", lang, quote=quote)


def _references_block(cycle: dict) -> str:
    references = cycle.get("grader_info") or {}
    if not references:
        return ""
    parts = {c["question_snapshot_id"]: c.get("part", "") for c in cycle.get("component_map") or []}
    return "\n\nReferencias del profesor (el estudiante no las ve):\n" + "\n".join(
        f"{parts.get(qid, '') or qid}:\n" + fence("REFERENCIA", ref) for qid, ref in references.items() if ref
    )


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
        + _references_block(cycle)
        + "\n\nRespuesta del estudiante:\n"
        + fence("RESPUESTA", answer[:6000])
    )
    system = (
        system
        + MOVES_RULE
        + (reference_rule("followup", lang) if cycle.get("grader_info") else "")
        + json_contract(FollowupPlan)
    )
    usage: list = []
    model = ""
    quote: str | None = None
    question = target = None
    mode, refusal = "deepen", "failed"
    note = ""
    for attempt in range(2):
        plan: FollowupPlan | None = None
        try:
            provider = get_provider(config)
            model = resolve_model(config)
            started = time.monotonic()
            plan = await asyncio.wait_for(
                provider.generate_structured(
                    [{"role": "system", "content": system}, {"role": "user", "content": user + note}],
                    model,
                    FollowupPlan,
                    thinking_budget=settings.pablo_thinking_followup,
                ),
                timeout=settings.pablo_timeout_followup_seconds,
            )
            usage.append(
                make_usage_record(
                    node="pablo_followup",
                    provider=provider,
                    model=model,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    provenance=provenance,
                )
            )
            if not isinstance(plan, FollowupPlan):
                plan = None
        except Exception as exc:  # noqa: BLE001 — retried, then a question in code goes out
            logger.info("pablo_followup_failed", attempt=attempt, error=str(exc)[:300])
            plan = None
        if plan is not None:
            quote = quote or echo.validated_quote(str(plan.quote or ""), answer)
        question, target, mode, refusal = validated(plan, cycle, answer)
        if question is not None:
            break
        logger.info("pablo_followup_refused", attempt=attempt, reason=refusal[:200])
        note = refusal_note(refusal)

    fallback = None
    if question is None:
        # The student's own words: the model's quote, else the answer's first clause.
        question = quoted_question(quote or answer_fragment(answer), cycle, lang)
        fallback = "quoted" if question else "generic"
        question = question or generic_question(cycle, target, lang)
    return (
        {
            "question": question,
            "target_component_id": target,
            "mode": mode,
            "quote": quote,
            "provenance": {
                **provenance,
                "model_id": model,
                **({"fallback": fallback, "refusal": refusal[:200]} if fallback else {}),
            },
        },
        usage,
    )
