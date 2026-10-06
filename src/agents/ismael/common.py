"""Helpers shared by ismael's nodes (triage/survey, Moodle support, teacher)."""

from __future__ import annotations

import asyncio
import re
import unicodedata

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer

from ...graphs.state import AgentState
from ..utils import emit_quick_replies

DEFAULT_PERSONA = "Ismael"


def ismael_dict(state: AgentState) -> dict:
    """A copy of ``state["ismael"]`` — nodes return the whole dict back."""
    value = state.get("ismael")
    return dict(value) if isinstance(value, dict) else {}


def user_ctx(state: AgentState) -> dict:
    ctx = state.get("user_context") or {}
    return ctx if isinstance(ctx, dict) else {}


def reply(message: str, **updates) -> dict:
    get_stream_writer()({"type": "token", "content": message})
    return {"messages": [AIMessage(content=message)], **updates}


def numbered(question: str, labels: list[str]) -> str:
    """A question plus the numbered list the backend matches quick replies on."""
    return "\n".join([question, *(f"{n}) {label}" for n, label in enumerate(labels, start=1))])


def ask_options(question: str, labels: list[str], **updates) -> dict:
    """Reply with a closed question: numbered text everywhere, buttons where drawn."""
    result = reply(numbered(question, labels), **updates)
    emit_quick_replies(labels)
    return result


def history_messages(state: AgentState, limit: int = 6) -> list[dict]:
    """The turns before the current one, as provider messages."""
    return [
        {"role": "assistant" if getattr(m, "type", "") == "ai" else "user", "content": m.content}
        for m in (state.get("messages") or [])[-(limit + 1) : -1]
        if getattr(m, "content", None)
    ]


def fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip()


def match_option(reply_text: str, options: list[list[str]]) -> int | None:
    """0-based index of the option a reply names, or None when it takes an LLM.

    ``options[i]`` holds every accepted label of option i (one per language).
    A quick-reply button sends its label verbatim, so an exact match comes
    first; then a bare number; then a label contained in the reply, only when
    exactly one option matches.
    """
    folded = fold(reply_text)
    if not folded:
        return None
    for index, labels in enumerate(options):
        if folded in (fold(label) for label in labels):
            return index
    number = re.fullmatch(r"\D{0,12}?(\d{1,2})\D{0,12}", folded)
    if number:
        n = int(number.group(1))
        return n - 1 if 1 <= n <= len(options) else None
    hits = {
        index
        for index, labels in enumerate(options)
        if any(fold(label) and fold(label) in folded for label in labels)
    }
    return hits.pop() if len(hits) == 1 else None



async def bounded_structured(
    provider,
    messages: list[dict],
    model: str,
    schema,
    *,
    timeout: float,
    thinking_budget: int | None,
):
    """``provider.generate_structured`` with a deadline and a thinking cap.

    ``asyncio.wait_for`` cancels the request when it overruns — that is what
    frees the conversation's lock — and raises, so the caller's ``except``
    sends its fallback reply within the backend's 60 s turn.
    """
    return await asyncio.wait_for(
        provider.generate_structured(messages, model, schema, thinking_budget=thinking_budget),
        timeout=timeout,
    )
