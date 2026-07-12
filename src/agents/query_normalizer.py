"""Query normalizer node — typo correction before triage/FAQ consumption.

Runs first in every agent graph (START → query_normalizer → triage). On most
turns it is a pure-Python passthrough that only stamps `original_text`; the
LLM correction runs ONLY when the deterministic trigger fires (per-tenant
flag on, initial backend FAQ retrieval empty/marginal, text long enough, no
explicit escalation keyword — see `shared_routing.should_normalize`).

When a correction is applied, the node re-runs FAQ retrieval once through the
backend's canonical search (`GET /internal/faqs/search`, conversation-scoped)
and overwrites `state["faqs"]` so downstream nodes consume the improved
matches with zero changes. It never rewrites `state["messages"]` — the
corrected text lives only in the `normalization` provenance dict.

Sole writer of `original_text` and `normalization` (write-once per turn).
"""

from __future__ import annotations

import json

import structlog
from langchain_core.runnables import RunnableConfig

from ..graphs.shared_routing import should_normalize
from ..graphs.state import AgentState
from ..observability import query_normalizations_total, record_node_invocation
from ..providers.registry import get_provider, resolve_model
from ..usage import make_usage_record
from ..json_utils import strip_json_fences
from . import backend_client
from .utils import latest_user_text, resolve_prompt

logger = structlog.get_logger(__name__)

# Corrections below this confidence are discarded — a wrong "fix" is worse
# than a missed FAQ (the legacy flow still answers from the original text).
NORMALIZATION_MIN_CONFIDENCE = 0.7

_VALID_RISKS = {"LOW", "MEDIUM", "HIGH"}

_NORMALIZER_SYSTEM_PROMPT = """You are a text normalizer for a customer-service assistant. The user's message may contain typos or misspellings that break keyword search.

Your ONLY job: fix spelling/typos. NEVER translate, rephrase, expand, shorten, or change the meaning. Keep the user's language exactly as written.

Respond with ONLY a one-line JSON object — no markdown, no extra text:
{{"corrected_text": "<the message with typos fixed>", "confidence": 0.0-1.0, "changed_meaning_risk": "LOW|MEDIUM|HIGH", "reason": "<what you corrected, briefly>"}}

- confidence: how sure you are the corrections are right.
- changed_meaning_risk: LOW = pure typo fixes; MEDIUM = a correction could shift nuance; HIGH = any correction might change what the user meant.
- If the text needs no correction, return it unchanged with confidence 1.0 and risk LOW."""


def _stub(enabled: bool, reason: str) -> dict:
    """Provenance dict for turns where no correction was applied pre-LLM."""
    return {
        "enabled": enabled,
        "model": None,
        "confidence": None,
        "changed_meaning_risk": None,
        "reason": reason,
        "applied": False,
        "corrected_text": None,
    }


def _sanitize_faqs(raw: list) -> list:
    """Backend search results → the same faq shape main.py produces."""
    return [
        {
            "id": str(f.get("id") or ""),
            "question": f.get("question", ""),
            "answer": f.get("answer", ""),
            "category": f.get("category") or "",
            "priority": 0,
            "score": float(f.get("score") or 0.0),
        }
        for f in raw
        if isinstance(f, dict)
    ]


def _max_score(faqs: list) -> float:
    return max(
        (float(f.get("score") or 0.0) for f in faqs if isinstance(f, dict)),
        default=0.0,
    )


async def query_normalizer_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("query_normalizer")

    thread_id = state.get("thread_id", "unknown")
    text = latest_user_text(state)
    faqs: list = state.get("faqs") or []
    flag = bool((config.get("configurable") or {}).get("normalization_enabled"))

    if not should_normalize(text, faqs, flag):
        query_normalizations_total.labels(outcome="no_trigger").inc()
        return {"original_text": text, "normalization": _stub(flag, "no_trigger")}

    provider = get_provider(config)
    model = resolve_model(config)
    prompt = resolve_prompt(
        config, "QUERY_NORMALIZER", _NORMALIZER_SYSTEM_PROMPT, state
    )

    stream = provider.stream_chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    )
    full_response = ""
    async for chunk in stream:
        full_response += chunk

    # Tokens were spent regardless of what happens next.
    usage_record = make_usage_record(
        node="query_normalizer", provider=provider, model=model
    )

    corrected: str = ""
    confidence: float = 0.0
    risk: str = "HIGH"
    llm_reason: str = ""
    outcome: str
    applied = False
    try:
        parsed = json.loads(strip_json_fences(full_response))
        corrected = str(parsed.get("corrected_text") or "").strip()
        confidence = float(parsed.get("confidence") or 0.0)
        risk = str(parsed.get("changed_meaning_risk") or "HIGH").upper()
        if risk not in _VALID_RISKS:
            risk = "HIGH"
        llm_reason = str(parsed.get("reason") or "")
        if not corrected:
            outcome = "error"
            llm_reason = "empty_corrected_text"
        elif risk == "HIGH":
            outcome = "discarded_high_risk"
        elif confidence < NORMALIZATION_MIN_CONFIDENCE:
            outcome = "discarded_low_confidence"
        elif corrected.strip() == text.strip():
            outcome = "no_trigger"
            llm_reason = "unchanged"
        else:
            applied = True
            outcome = "applied"
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "query_normalizer_parse_failed",
            thread_id=thread_id,
            error=str(exc),
            raw=full_response,
        )
        outcome = "error"
        llm_reason = "parse_error"

    update: dict = {
        "original_text": text,
        "turn_usage": [usage_record],
    }

    score_before = _max_score(faqs)
    score_after = score_before
    new_faqs: list = []

    if applied:
        conversation_id = state.get("conversation_id") or ""
        try:
            raw_results = await backend_client.search_faqs(
                conversation_id, corrected, limit=5
            )
            new_faqs = _sanitize_faqs(raw_results)
        except Exception as exc:  # noqa: BLE001 — HTTP failure must not kill the turn
            logger.warning(
                "query_normalizer_retrieval_failed",
                thread_id=thread_id,
                error=str(exc),
            )
            applied = False
            outcome = "retrieval_failed"
            llm_reason = "retrieval_failed"

    if applied and new_faqs:
        update["faqs"] = new_faqs
        score_after = _max_score(new_faqs)
    elif applied:
        # Correction stands (triage may still log it) but the original faqs —
        # weak as they were — stay in place.
        outcome = "retrieval_empty"
        score_after = 0.0

    update["normalization"] = {
        "enabled": True,
        "model": model,
        "confidence": confidence,
        "changed_meaning_risk": risk,
        "reason": llm_reason,
        "applied": applied,
        "corrected_text": corrected if applied else None,
    }

    query_normalizations_total.labels(outcome=outcome).inc()
    logger.info(
        "query_normalization",
        thread_id=thread_id,
        tenant_id=state.get("tenant_id", ""),
        outcome=outcome,
        original_text=text,
        corrected_text=corrected,
        confidence=confidence,
        risk=risk,
        applied=applied,
        score_before=score_before,
        score_after=score_after,
        faq_count_before=len(faqs),
        faq_count_after=len(new_faqs) if new_faqs else len(faqs),
        reason=llm_reason,
    )

    return update
