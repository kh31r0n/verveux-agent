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
"""

from __future__ import annotations

import structlog
from langchain_core.runnables import RunnableConfig

from ...graphs.state import AgentState
from ...observability import record_node_invocation
from ...providers.registry import get_provider, resolve_model
from ...schemas.ismael import MoodleSupportOutcome, MoodleSupportResult
from ...usage import make_usage_record
from ..utils import format_user_context, latest_user_text, resolve_persona, resolve_prompt
from .common import DEFAULT_PERSONA, history_messages, ismael_dict, reply, user_ctx
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

    system = (
        resolve_prompt(config, "THEOLOGY_MOODLE_SUPPORT", MOODLE_SUPPORT_PROMPT, state)
        .replace("{persona}", persona)
        .replace("{language_rule}", _language_rule(lang))
        + format_user_context(state)
        + support_data_block(ctx)
    )
    usage: list = []
    try:
        provider = get_provider(config)
        model = resolve_model(config)
        result = await provider.generate_structured(
            [
                {"role": "system", "content": system},
                *history_messages(state),
                {"role": "user", "content": handed_over or latest_user_text(state) or question},
            ],
            model,
            MoodleSupportResult,
        )
        usage.append(make_usage_record(node="ismael_moodle_support", provider=provider, model=model))
    except Exception as exc:  # noqa: BLE001 — fall back to the ticket, never to silence
        logger.warning("ismael_moodle_support_failed", error=str(exc))
        result = MoodleSupportResult(
            reply=text("support_failed", lang),
            outcome=MoodleSupportOutcome.NEEDS_TICKET,
            ticket_description=question,
        )

    message = result.reply.strip()
    if result.outcome == MoodleSupportOutcome.NEEDS_TICKET:
        message = f"{message}\n\n{ticket_block(ctx, result.ticket_description, lang)}".strip()
    logger.info("ismael_moodle_support", outcome=result.outcome.value)
    return reply(message, ismael=ismael, turn_usage=usage, ismael_route="done")
