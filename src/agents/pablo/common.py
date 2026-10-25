"""Helpers shared by pablo's chat nodes and background jobs."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer

from ...graphs.state import AgentState
from ..utils import resolve_persona, resolve_prompt, resolve_prompt_provenance

DEFAULT_PERSONA = "Pablo"


def language_rule(lang: str) -> str:
    return "Always respond in English." if lang == "en" else "Responde siempre en español."


def system_prompt(
    config, key: str, fallback: str, lang: str, persona: str = DEFAULT_PERSONA
) -> tuple[str, dict]:
    """The resolved prompt text plus its provenance (for usage rows and the
    evaluation's audit trail), placeholders substituted."""
    text = (
        resolve_prompt(config, key, fallback)
        .replace("{persona}", persona or DEFAULT_PERSONA)
        .replace("{language_rule}", language_rule(lang))
    )
    return text, resolve_prompt_provenance(config, key, text, _version_of(key))


def _version_of(key: str) -> str:
    from . import prompts

    name = key.removeprefix("ASSESSMENT_") + "_VERSION"
    return str(getattr(prompts, name, "unknown"))


def fence(tag: str, body: str) -> str:
    """Data for a prompt, fenced so it is never read as an instruction. The
    closing marker is stripped from the body: an answer cannot close its fence."""
    clean = (body or "").replace(f"{tag}>>>", "").replace(f"<<<{tag}", "")
    return f"<<<{tag}\n{clean}\n{tag}>>>"


def pablo_dict(state: AgentState) -> dict:
    """A copy of ``state["pablo"]`` — nodes return the whole dict back."""
    value = state.get("pablo")
    return dict(value) if isinstance(value, dict) else {}


def assessment_turn(state: AgentState) -> dict:
    """This request's turn, sent by the backend in ``domain_state`` (reset
    every request, so a turn can never be replayed from the checkpoint)."""
    domain = state.get("domain_state") or {}
    turn = domain.get("assessment_turn") if isinstance(domain, dict) else None
    return turn if isinstance(turn, dict) else {}


def persona(state: AgentState) -> str:
    return resolve_persona(state, DEFAULT_PERSONA)


def reply(message: str, **updates) -> dict:
    get_stream_writer()({"type": "token", "content": message})
    return {"messages": [AIMessage(content=message)], **updates}


def emit_assessment(event: str, **data) -> None:
    """A stream event the backend reads to drive the attempt's clock and the
    activity UI (e.g. ``cycle_presented`` restarts the timer)."""
    get_stream_writer()({"type": "assessment", "event": event, **data})
