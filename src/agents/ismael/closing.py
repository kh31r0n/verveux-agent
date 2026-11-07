"""ismael's closing: a thank-you gets a short reply that asks the student to
rate the service, never the full introduction again.

    ismael_closing   "gracias" / "ok" / a goodbye — or a greeting in the middle
                     of a conversation. Once per 24 h, and only after ismael
                     actually helped (``helped_at``), the reply ends with two
                     buttons: «👍 Me ayudó» / «👎 No me ayudó». Otherwise it is
                     one line. No LLM call either way.
    ismael_rating    the answer to those buttons. A latch in
                     ``state["ismael"]["rating"]`` (``step`` pending → reason)
                     routes the next message here without triage, like the
                     survey and the teacher flow.

The rating is stored by the backend as ``message_feedback`` on the closing
question (the same table as the widget's thumbs), so it shows in the inbox
chip and the metrics. A 👎 asks once what was missing and stores it as the
reason. Saving never blocks the reply.
"""

from __future__ import annotations

import time

import structlog
from langchain_core.runnables import RunnableConfig

from ...config import settings
from ...graphs.state import AgentState
from ...observability import record_node_invocation
from ...providers.registry import get_provider, resolve_background_model
from ...schemas.ismael import TeacherStepAnswer
from ...usage import make_usage_record
from .. import backend_client
from ..utils import latest_user_text
from .common import ask_options, bounded_structured, fold, ismael_dict, latch_fresh, match_option, reply
from .prompts import STEP_PROMPT
from .texts import lang_of, text

logger = structlog.get_logger(__name__)

RATING_INTERVAL_SECONDS = 24 * 60 * 60
REASON_MAX = 500

# (es, en) visible labels, then extra words that pick the option. The number a
# user types is the position.
RATING = (("👍 Me ayudó", "👍 It helped"), ("👎 No me ayudó", "👎 It didn't help"))
_RATING_ALIASES = (
    ["👍", "me ayudo", "me sirvio", "it helped", "excelente", "muy bien"],
    ["👎", "no me ayudo", "no me sirvio", "it didn't help", "no sirvio", "no ayudo"],
)
# A bare yes/no to "¿Te sirvió mi ayuda?", folded.
_YES = {"si", "claro", "mucho", "si mucho", "si gracias", "yes", "sure"}
_NO = {"no", "nop", "no mucho", "nada", "no gracias"}
_VALUES = ("UP", "DOWN")


def rating_open(ismael: dict) -> bool:
    latch = ismael.get("rating")
    return isinstance(latch, dict) and bool(latch.get("step")) and latch_fresh(latch)


def _first_name(state: AgentState) -> str:
    ctx = state.get("user_context") or {}
    name = (ctx.get("name") or "").strip() if isinstance(ctx, dict) else ""
    return f", {name.split()[0]}" if name else ""


def _should_rate(ismael: dict) -> bool:
    helped = float(ismael.get("helped_at") or 0)
    asked = float(ismael.get("rating_asked_at") or 0)
    return helped > asked and time.time() - asked >= RATING_INTERVAL_SECONDS


async def ismael_closing_node(state: AgentState, config: RunnableConfig) -> dict:
    record_node_invocation("ismael_closing")
    ismael = ismael_dict(state)
    lang = lang_of(state)
    name = _first_name(state)
    ismael.pop("support", None)  # "gracias" closes the case

    if state.get("ismael_route") == "regreet":
        return reply(text("closing_again", lang, name=name), ismael=ismael, ismael_route="done")
    if not _should_rate(ismael):
        return reply(text("closing_plain", lang, name=name), ismael=ismael, ismael_route="done")

    now = time.time()
    ismael["rating_asked_at"] = now
    ismael["rating"] = {"step": "pending", "touched_at": now}
    labels = [pair[1] if lang == "en" else pair[0] for pair in RATING]
    return ask_options(text("closing_rate", lang, name=name), labels, ismael=ismael, ismael_route="done")


async def _classify(config: RunnableConfig, message: str, lang: str) -> tuple[int | None, bool, list]:
    """(0-based option, is_new_question, usage) — deterministic first, LLM otherwise."""
    bare = fold(message).strip(" .!¡¿?,")
    if bare in _YES:
        return 0, False, []
    if bare in _NO:
        return 1, False, []
    index = match_option(message, [[*pair, *aliases] for pair, aliases in zip(RATING, _RATING_ALIASES)])
    if index is not None:
        return index, False, []
    if not message.strip():
        return None, False, []
    labels = [pair[1] if lang == "en" else pair[0] for pair in RATING]
    options = "\n".join(f"{n}) {label}" for n, label in enumerate(labels, start=1))
    system = STEP_PROMPT.format(question=text("closing_rate", lang, name=""), options=options)
    try:
        provider = get_provider(config)
        model = resolve_background_model(config)
        result = await bounded_structured(
            provider,
            [{"role": "system", "content": system}, {"role": "user", "content": message}],
            model,
            TeacherStepAnswer,
            timeout=settings.ismael_timeout_step_seconds,
            thinking_budget=settings.ismael_thinking_classify,
        )
        usage = [make_usage_record(node="ismael_rating", provider=provider, model=model)]
    except Exception as exc:  # noqa: BLE001 — an unreadable rating is just not recorded
        logger.info("ismael_rating_classify_failed", error=str(exc))
        return None, False, []
    if 1 <= result.choice <= len(RATING):
        return result.choice - 1, False, usage
    return None, result.is_new_question, usage


async def _save(state: AgentState, rating: str, reason: str = "", message_id: str = "") -> str:
    """Store the rating; returns the rated message's id ("" on failure)."""
    try:
        result = await backend_client.save_ismael_rating(
            state.get("conversation_id") or "", rating, reason=reason, message_id=message_id
        )
    except Exception as exc:  # noqa: BLE001 — a lost rating must not cost the reply
        logger.warning("ismael_rating_save_failed", error=str(exc) or type(exc).__name__)
        return ""
    return str((result or {}).get("messageId") or "")


async def ismael_rating_node(state: AgentState, config: RunnableConfig) -> dict:
    record_node_invocation("ismael_rating")
    ismael = ismael_dict(state)
    lang = lang_of(state)
    latch = dict(ismael.get("rating") or {})
    message = latest_user_text(state)

    if latch.get("step") == "reason":
        ismael.pop("rating", None)
        reason = message.strip()[:REASON_MAX]
        if reason:
            await _save(state, "DOWN", reason=reason, message_id=str(latch.get("message_id") or ""))
        logger.info("ismael_rating", value="DOWN", reason=bool(reason))
        return reply(text("rating_reason_thanks", lang), ismael=ismael, ismael_route="done")

    index, new_question, usage = await _classify(config, message, lang)
    ismael.pop("rating", None)
    if index is None:
        # Not a rating (another question, or unreadable): answer it instead.
        logger.info("ismael_rating", value=None, new_question=new_question)
        return {"ismael": ismael, "turn_usage": usage, "ismael_route": "retriage"}

    value = _VALUES[index]
    message_id = await _save(state, value)
    logger.info("ismael_rating", value=value)
    if value == "UP":
        return reply(text("rating_thanks", lang), ismael=ismael, turn_usage=usage, ismael_route="done")
    ismael["rating"] = {"step": "reason", "touched_at": time.time(), "message_id": message_id}
    return reply(text("rating_ask_reason", lang), ismael=ismael, turn_usage=usage, ismael_route="done")
