"""Shared name-capture routing — one policy for every agent graph.

`Contact.customName` (backend DB) is the single source of truth for the
customer's identity. When the backend snapshot carries no name, graphs route
brand-new users to `name_capture` — unless the intent is urgent (the task
matters more than the introduction) or the customer previously declined
(deferral, re-evaluated by the backend every turn so it expires after
NAME_CAPTURE_DEFER_DAYS).
"""

from __future__ import annotations

from ..config import settings
from ..schemas.intent import IntentType
from .state import AgentState

# Intents that skip the name ask: anything angry, urgent, or already headed
# to a human. The customer's problem outranks our curiosity about their name.
URGENT_INTENTS: frozenset[str] = frozenset(
    {
        IntentType.COMPLAINT.value,
        IntentType.ESCALATION.value,
        IntentType.PAYMENT_PROOF.value,
        IntentType.CORRECTION_REQUEST.value,
        IntentType.IDENTITY_CONFLICT.value,
        IntentType.APPOINTMENT_CANCEL.value,
    }
)


def has_name(state: AgentState) -> bool:
    """True when the contact's name is known.

    Sources, in order: the generic `name_captured` latch, camila's legacy
    `school_name_captured` latch (old checkpoints), or a non-empty
    `user_context.name` from the backend snapshot (Contact.customName).
    """
    if state.get("name_captured") or state.get("school_name_captured"):
        return True
    ctx = state.get("user_context") or {}
    if not isinstance(ctx, dict):
        return False
    return bool((ctx.get("name") or "").strip())


def is_deferred(state: AgentState) -> bool:
    """True while the customer's refusal to give a name is still in effect.

    The backend snapshot flag (`user_context.name_capture_deferred`) is
    authoritative — it already applies the NAME_CAPTURE_DEFER_DAYS window.
    The state flag covers the remainder of the turn/thread where the agent
    just recorded the refusal locally.
    """
    if state.get("name_capture_deferred"):
        return True
    ctx = state.get("user_context") or {}
    if not isinstance(ctx, dict):
        return False
    return bool(ctx.get("name_capture_deferred"))


# ── Query normalization trigger — one policy for every agent graph ──────────

# Best FAQ score at or above this means retrieval already worked; the
# normalizer only runs when the initial backend retrieval came back empty
# or marginal. Informed guess — tune via the query_normalization logs.
FAQ_SCORE_TRIGGER_THRESHOLD = 0.15

# Messages shorter than this ("ok", "sí", "3") carry too little signal for
# typo correction to help and too much risk of changing meaning.
MIN_NORMALIZATION_TEXT_LEN = 4

# Lexical escalation signals (es/en/pt). Intent is not classified yet when
# the normalizer runs, so this must be keyword-based. Deliberately
# conservative: any hit suppresses normalization so an angry or urgent
# message reaches triage untouched.
ESCALATION_KEYWORDS: frozenset[str] = frozenset(
    {
        # es
        "humano", "asesor", "agente", "supervisor", "gerente", "queja",
        "reclamo", "denuncia", "urgente", "demanda", "abogado", "cancelar",
        # en
        "human", "manager", "complaint", "refund", "lawyer", "urgent",
        "cancel",
        # pt
        "atendente", "reclamação", "advogado",
    }
)


def contains_escalation_keywords(text: str) -> bool:
    """True when the message carries an explicit escalation signal."""
    lowered = (text or "").lower()
    return any(kw in lowered for kw in ESCALATION_KEYWORDS)


def should_normalize(text: str, faqs: list, flag_enabled: bool) -> bool:
    """Deterministic trigger for LLM query normalization.

    Pure function of its inputs so the graph node and tests share one
    policy. Normalize ONLY IF the per-tenant flag AND the fleet-wide env
    switch are on, the text is long enough, it carries no explicit
    escalation keyword, and the initial backend FAQ retrieval came back
    empty or below FAQ_SCORE_TRIGGER_THRESHOLD.
    """
    if not flag_enabled or not settings.query_normalization_enabled:
        return False
    if len((text or "").strip()) < MIN_NORMALIZATION_TEXT_LEN:
        return False
    if contains_escalation_keywords(text):
        return False
    max_score = max(
        (float(f.get("score") or 0.0) for f in faqs if isinstance(f, dict)),
        default=0.0,
    )
    return max_score < FAQ_SCORE_TRIGGER_THRESHOLD


def should_capture_name(state: AgentState, intent: str | None = None) -> bool:
    """Single policy gate: route this turn to `name_capture`?

    False when the intent is urgent, the name is already known, or the
    customer declined within the deferral window. Graphs call this AFTER
    their urgent/attachment/in-progress-flow branches so mid-flow turns are
    never hijacked.
    """
    effective_intent = (
        intent if intent is not None else str(state.get("intent") or "")
    ).lower()
    if effective_intent in URGENT_INTENTS:
        return False
    return not has_name(state) and not is_deferred(state)
