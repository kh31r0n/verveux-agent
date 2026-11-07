"""The acknowledgement after an answer: a literal quote of the student's idea.

The model only picks a fragment; what reaches the student is the fragment, and
only after code checked it is really in the answer. Everything around it (the
template, the thanks) is fixed text, so the acknowledgement can never praise,
grade or hint — and a model that fails or invents a sentence just means no
quote, never a different message.
"""

from __future__ import annotations

import asyncio
import re
import time

import structlog
from langchain_core.runnables import RunnableConfig

from ...config import settings
from ...providers.registry import get_provider, resolve_background_model
from ...schemas.pablo import AnswerEcho
from ...usage import make_usage_record
from .common import fence, json_contract, system_prompt
from .prompts import ACK_PROMPT

logger = structlog.get_logger(__name__)

MIN_WORDS = 3
MAX_WORDS = 25
MAX_CHARS = 200

_SPACE = re.compile(r"\s+")
_EDGE = "\"'«»“”‘’.,;:¡!¿?()[] "


def _norm(value: str) -> str:
    return _SPACE.sub(" ", value).strip().casefold()


def validated_quote(quote: str, answer: str) -> str | None:
    """The quote, trimmed of surrounding quotes and punctuation, when it is a
    literal fragment of the answer (case and spacing aside) of 3–25 words."""
    clean = _SPACE.sub(" ", quote or "").strip().strip(_EDGE).strip()
    if not clean or len(clean) > MAX_CHARS:
        return None
    if not MIN_WORDS <= len(clean.split(" ")) <= MAX_WORDS:
        return None
    if _norm(clean) not in _norm(answer):
        return None
    return clean


async def quote_of(
    config: RunnableConfig, answer: str, lang: str, persona: str
) -> tuple[str | None, list]:
    """(validated quote | None, usage). Never raises."""
    if len((answer or "").split()) < MIN_WORDS:
        return None, []
    system, provenance = system_prompt(config, "ASSESSMENT_ACK", ACK_PROMPT, lang, persona)
    try:
        provider = get_provider(config)
        # Picking a fragment needs no judgement: the platform's cheap model.
        model = resolve_background_model(config)
        started = time.monotonic()
        result = await asyncio.wait_for(
            provider.generate_structured(
                [
                    {"role": "system", "content": system + json_contract(AnswerEcho)},
                    {"role": "user", "content": fence("RESPUESTA", answer[:4000])},
                ],
                model,
                AnswerEcho,
                thinking_budget=settings.pablo_thinking_ack,
            ),
            timeout=settings.pablo_timeout_ack_seconds,
        )
        usage = [
            make_usage_record(
                node="pablo_ack",
                provider=provider,
                model=model,
                latency_ms=int((time.monotonic() - started) * 1000),
                provenance=provenance,
            )
        ]
    except Exception as exc:  # noqa: BLE001 — the plain acknowledgement still goes out
        logger.info("pablo_ack_failed", error=str(exc)[:300])
        return None, []
    quote = validated_quote(str(getattr(result, "quote", "") or ""), answer)
    if quote is None:
        logger.info("pablo_ack_quote_rejected")
    return quote, usage
