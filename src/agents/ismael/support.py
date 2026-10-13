"""ismael's Moodle support branch: how-to answers and the guided support ticket.

The answer comes from the chat model's general knowledge of Moodle 4.x (no
FAQs, no Brain). The institution-specific parts — the technical-support form,
what it asks for, and who handles enrolment or payments — come from the MOODLE
connection's ``moodleSupport`` config, which the backend forwards in
``user_context`` (``lms_support_url``, ``lms_support_instructions``,
``lms_other_contacts``).

When the problem needs the institution (``needs_ticket``) the ticket block is
assembled HERE, in code: the form link and the user's own data must be exact,
so the model only writes the description.

Ask before escalating. A support case lives in ``state["ismael"]["support"]``
(``question``, ``awaiting`` — the last reply asked something or gave steps to
try —, ``clarified`` — how many of those the student answered —,
``touched_at``).
Every escalation (a ticket, a redirect to the institution's contacts, the
teacher) is held back in code until ``clarified >= 1``: the model's
``clarifying_question`` is sent instead. Prompts alone did not hold it — on
2026-10-07 a student got the question and the ticket in one message.

A grade or a quiz question the student disputes is the teacher's call:
outcome ``teacher`` hands the turn to the teacher flow, which skips its
"can I help you first?" offer (``teacher.from_support``).
"""

from __future__ import annotations

import re
import time

import structlog
from langchain_core.runnables import RunnableConfig

from ...config import settings
from ...graphs.state import AgentState
from ...observability import record_node_invocation
from ...providers.registry import get_provider, resolve_background_model
from ...schemas.ismael import ESCALATIONS, MoodleSupportOutcome, MoodleSupportResult
from ...usage import make_usage_record
from ..utils import format_user_context, latest_user_text, resolve_persona, resolve_prompt
from . import teacher
from .common import (
    DEFAULT_PERSONA,
    bounded_structured,
    fold,
    history_messages,
    ismael_dict,
    latch_fresh,
    mark_helped,
    reply,
    user_ctx,
)
from .prompts import MOODLE_SUPPORT_PROMPT
from .texts import lang_of, text

logger = structlog.get_logger(__name__)


def _language_rule(lang: str) -> str:
    return "Always respond in English." if lang == "en" else "Responde siempre en español."


def support_data_block(ctx: dict) -> str:
    """The institution's support data, appended to the system prompt as facts."""
    lines = []
    if ctx.get("lms_support_url"):
        lines.append(f"- Formulario de soporte técnico: {ctx['lms_support_url']}")
    if ctx.get("lms_support_instructions"):
        lines.append(f"- Lo que pide el formulario: {ctx['lms_support_instructions']}")
    if ctx.get("lms_other_contacts"):
        lines.append(f"- Contactos para otros temas: {ctx['lms_other_contacts']}")
    if not lines:
        lines.append("- La institución no configuró datos de soporte.")
    return "\n\nDatos de soporte de la institución:\n" + "\n".join(lines)


def ticket_block(ctx: dict, description: str, lang: str) -> str:
    """The guided ticket: link + the user's data ready to paste."""
    url = ctx.get("lms_support_url") or ""
    description = description.strip()
    if not url:
        return text("ticket_no_url", lang) + (f"\n{description}" if description else "")

    document = ctx.get("lms_idnumber") or text("ticket_document_missing", lang)
    lines = [
        text("ticket_intro", lang, url=url),
        f"- {text('ticket_document', lang)}: {document}",
    ]
    if ctx.get("name"):
        lines.append(f"- {text('ticket_name', lang)}: {ctx['name']}")
    if ctx.get("email"):
        lines.append(f"- {text('ticket_email', lang)}: {ctx['email']}")
    if description:
        lines.append(f"- {text('ticket_description', lang)}: {description}")
    block = "\n".join(lines)
    if ctx.get("lms_support_instructions"):
        block += f"\n\n{ctx['lms_support_instructions']}"
    return block


# Administrative topics the technical-support form explicitly does not take
# (enrolment, payments, refunds, certificates). Folded, prefix match.
_ADMIN_WORDS = ("matricul", "pago", "pagar", "reintegro", "reembolso", "certificado", "factura", "cuota")


def fallback_support_result(ctx: dict, question: str, lang: str) -> MoodleSupportResult:
    """What to say when the model did not answer in time.

    An administrative question goes to the configured contacts — the support
    form is for technical problems only. Anything else gets the guided ticket.
    """
    folded = fold(question)
    if ctx.get("lms_other_contacts") and any(re.search(r"\b" + w, folded) for w in _ADMIN_WORDS):
        return MoodleSupportResult(
            reply=text("support_redirect_fallback", lang, contacts=ctx["lms_other_contacts"]),
            outcome=MoodleSupportOutcome.REDIRECT,
        )
    return MoodleSupportResult(
        reply=text("support_failed", lang),
        outcome=MoodleSupportOutcome.NEEDS_TICKET,
        ticket_description=question,
    )


def open_case(ismael: dict) -> dict:
    """The support case in progress, or {} when there is none (or it expired)."""
    case = ismael.get("support")
    return dict(case) if latch_fresh(case) else {}


def awaiting_answer(ismael: dict) -> bool:
    """True when ismael's last support reply asked something or gave steps to
    try, so the student's next message is most likely about the same case."""
    return bool(open_case(ismael).get("awaiting"))


def case_block(case: dict) -> str:
    """Tells the model where the case stands, so it does not ask forever."""
    if not case.get("clarified"):
        return ""
    return (
        f"\n\nCaso abierto: «{case.get('question', '')}». El estudiante ya respondió "
        f"{case['clarified']} pregunta(s) tuya(s) de aclaración: si ya tienes lo necesario, "
        "resuelve o escala sin volver a preguntar lo mismo."
    )


def hold_escalation(result: MoodleSupportResult, case: dict, lang: str) -> MoodleSupportResult:
    """No escalation before the student answered one clarifying question."""
    if result.outcome not in ESCALATIONS or case.get("clarified", 0) >= 1:
        return result
    case["leaning"] = result.outcome.value
    generic = {
        MoodleSupportOutcome.TEACHER: "support_clarify_teacher",
        MoodleSupportOutcome.REDIRECT: "support_clarify_redirect",
    }.get(result.outcome, "support_clarify")
    question = result.clarifying_question.strip() or text(generic, lang)
    return MoodleSupportResult(reply=question, outcome=MoodleSupportOutcome.ASK)


def ask_message(result: MoodleSupportResult, lang: str) -> str:
    """The reply to an ``ask``: it must actually ask something."""
    message = result.reply.strip()
    question = result.clarifying_question.strip()
    if not message:
        return question or text("support_clarify", lang)
    if "?" not in message and question:
        # The reply circles the question without asking it: keep its opening
        # sentence (the acknowledgement) and ask the question itself.
        opening = re.split(r"(?<=[.!])\s+", message, maxsplit=1)[0]
        return f"{opening} {question}"
    return message


async def ismael_moodle_support_node(state: AgentState, config: RunnableConfig) -> dict:
    """Answer a Moodle question; guide a ticket or redirect when it is not ours."""
    record_node_invocation("ismael_moodle_support")
    lang = lang_of(state)
    ctx = user_ctx(state)
    ismael = ismael_dict(state)
    persona = resolve_persona(state, DEFAULT_PERSONA)
    # Set by the teacher flow when the student accepted help with the reason
    # they gave earlier: this turn's own text is just "Sí, ayúdame".
    handed_over = ismael.pop("support_question", None)
    question = handed_over or ismael.get("question") or latest_user_text(state)

    case = {} if handed_over else open_case(ismael)
    if case.get("awaiting"):
        # This message answers what ismael asked (or reports on the steps) last turn.
        case["clarified"] = int(case.get("clarified") or 0) + 1
        case["awaiting"] = False
    if not case:
        case = {"question": question, "clarified": 0}

    system = (
        resolve_prompt(config, "THEOLOGY_MOODLE_SUPPORT", MOODLE_SUPPORT_PROMPT, state)
        .replace("{persona}", persona)
        .replace("{language_rule}", _language_rule(lang))
        + format_user_context(state)
        + support_data_block(ctx)
        + case_block(case)
    )
    usage: list = []
    try:
        provider = get_provider(config)
        model = resolve_background_model(config)
        result = await bounded_structured(
            provider,
            [
                {"role": "system", "content": system},
                *history_messages(state),
                {"role": "user", "content": handed_over or latest_user_text(state) or question},
            ],
            model,
            MoodleSupportResult,
            timeout=settings.ismael_timeout_answer_seconds,
            thinking_budget=settings.ismael_thinking_support,
        )
        usage.append(make_usage_record(node="ismael_moodle_support", provider=provider, model=model))
    except Exception as exc:  # noqa: BLE001 — a fallback answer, never silence
        logger.warning("ismael_moodle_support_failed", error=str(exc) or type(exc).__name__)
        result = fallback_support_result(ctx, question, lang)

    result = hold_escalation(result, case, lang)
    logger.info(
        "ismael_moodle_support",
        outcome=result.outcome.value,
        clarified=case.get("clarified", 0),
        held=case.get("leaning") if result.outcome == MoodleSupportOutcome.ASK else None,
    )

    if result.outcome == MoodleSupportOutcome.TEACHER:
        ismael.pop("support", None)  # the teacher flow takes it from here
        reason = result.ticket_description.strip() or case.get("question", "") or question
        ismael["teacher"] = teacher.from_support(reason)
        return {"ismael": ismael, "turn_usage": usage, "ismael_route": "teacher"}

    if result.outcome == MoodleSupportOutcome.ASK:
        message = ask_message(result, lang)
    else:
        message = result.reply.strip()
        mark_helped(ismael)
    # The case stays open even after a ticket: more detail next turn ("es en el
    # foro de la tarea 3") continues it instead of being asked about again.
    # "gracias" or the TTL closes it.
    case["awaiting"] = True
    case["touched_at"] = time.time()
    ismael["support"] = case
    if result.outcome == MoodleSupportOutcome.NEEDS_TICKET:
        message = f"{message}\n\n{ticket_block(ctx, result.ticket_description, lang)}".strip()
    return reply(message, ismael=ismael, turn_usage=usage, ismael_route="done")
