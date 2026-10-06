"""ismael's FAQ branch: answer from the institution's own FAQs, before Brain.

The backend sends the top FAQs for every turn (``state["faqs"]``, retrieved for
the conversation's code name). Triage sees them as candidates and names the one
that answers (``TriageResult.faq_id``) — no extra LLM call to decide. This node
then adapts that FAQ's answer to the exact question, adding nothing; when the
model fails, the FAQ's answer goes out verbatim. A theology question answered
by a FAQ never reaches Brain: the institution's curated answer wins.

The chosen FAQ is kept whole in ``state["ismael"]["faq"]`` rather than by id,
because ``state["faqs"]`` is re-retrieved every turn: when the teacher flow
hands over a turn later, this turn's candidates belong to "Sí, ayúdame".
"""

from __future__ import annotations

import structlog
from langchain_core.runnables import RunnableConfig

from ...config import settings
from ...graphs.state import AgentState
from ...observability import record_node_invocation
from ...providers.registry import get_provider, resolve_model
from ...schemas.ismael import FaqAnswer
from ...usage import make_usage_record
from ..utils import latest_user_text, resolve_persona, resolve_prompt
from .common import bounded_structured, DEFAULT_PERSONA, history_messages, ismael_dict, reply
from .prompts import FAQ_PROMPT
from .texts import lang_of

logger = structlog.get_logger(__name__)

# Candidates shown to triage (the backend sends at most 3 today).
MAX_FAQ_CANDIDATES = 3
_MAX_ANSWER_CHARS = 1500


def faq_candidates(state: AgentState) -> list[dict]:
    """This turn's FAQs that carry an id and an answer."""
    return [
        faq
        for faq in (state.get("faqs") or [])[:MAX_FAQ_CANDIDATES]
        if isinstance(faq, dict) and faq.get("id") and faq.get("answer")
    ]


def faq_candidates_block(candidates: list[dict]) -> str:
    """The candidates as facts appended to the triage prompt."""
    if not candidates:
        return ""
    lines = [
        f"- id={faq['id']} · P: {faq.get('question', '')} · R: {str(faq.get('answer', ''))[:400]}"
        for faq in candidates
    ]
    return "\n\nPreguntas frecuentes de la institución (candidatas):\n" + "\n".join(lines)


def pick_faq(candidates: list[dict], faq_id: str) -> dict | None:
    """The candidate triage named — only ever one that was actually shown."""
    faq_id = (faq_id or "").strip()
    if not faq_id:
        return None
    return next((dict(faq) for faq in candidates if str(faq.get("id")) == faq_id), None)


def _language_rule(lang: str) -> str:
    return "Always respond in English." if lang == "en" else "Responde siempre en español."


async def ismael_faq_node(state: AgentState, config: RunnableConfig) -> dict:
    record_node_invocation("ismael_faq")
    ismael = ismael_dict(state)
    faq = ismael.pop("faq", None) or {}
    question = ismael.pop("faq_question", None) or latest_user_text(state)
    lang = lang_of(state)
    answer = str(faq.get("answer") or "")[:_MAX_ANSWER_CHARS]

    system = (
        resolve_prompt(config, "THEOLOGY_FAQ", FAQ_PROMPT, state)
        .replace("{persona}", resolve_persona(state, DEFAULT_PERSONA))
        .replace("{language_rule}", _language_rule(lang))
        + f"\n\nPregunta frecuente:\nP: {faq.get('question', '')}\nR: {answer}"
    )
    usage: list = []
    message = answer
    try:
        provider = get_provider(config)
        model = resolve_model(config)
        result = await bounded_structured(
            provider,
            [
                {"role": "system", "content": system},
                *history_messages(state),
                {"role": "user", "content": question},
            ],
            model,
            FaqAnswer,
            timeout=settings.ismael_timeout_answer_seconds,
            thinking_budget=settings.ismael_thinking_answer,
        )
        usage.append(make_usage_record(node="ismael_faq", provider=provider, model=model))
        message = result.reply.strip() or answer
    except Exception as exc:  # noqa: BLE001 — the curated answer itself still works
        logger.info("ismael_faq_rewrite_failed", error=str(exc))

    logger.info("ismael_faq", faq_id=faq.get("id"))
    faq_used = [
        {
            "id": faq.get("id"),
            "question": faq.get("question", ""),
            "confidence": faq.get("score", 0) or 0,
        }
    ] if faq.get("id") else None
    return reply(
        message, ismael=ismael, turn_usage=usage, faq_used=faq_used, ismael_route="done"
    )
