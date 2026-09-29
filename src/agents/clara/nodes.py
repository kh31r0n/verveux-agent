"""Graph nodes for the email agent.

Every node is either pure Python (``precheck_node``) or ONE structured-output
call against a schema in ``src/schemas/email.py``. No node calls Gmail, the
backend or any tool: email content can at worst produce a wrong
classification, never an action. Security guards run after every node so a
parser flag cannot be reasoned away by the model.

Errors propagate. A provider or schema failure leaves the message unreported,
so the runner retries it on the next sync (and the checkpoint resumes after the
nodes that already succeeded) instead of recording a guess.
"""

from __future__ import annotations

import time

from langgraph.types import RunnableConfig

from ...config import settings
from ...observability import record_node_invocation
from ...providers.registry import get_provider, resolve_model
from ...schemas.email import (
    Category,
    ExtractedTask,
    Intent,
    ParsedEmail,
    PrecheckResult,
    ReplyDraft,
    SenderContext,
    Sentiment,
    Subcategory,
    TriageResult,
    Urgency,
)
from ...usage import make_usage_record
from ..utils import resolve_prompt, resolve_prompt_provenance
from . import prompts

_BOUNCE_SENDERS = ("mailer-daemon@", "postmaster@")
_HINT_LABELS = ("CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_UPDATES", "CATEGORY_FORUMS")


def _code_name(state: dict) -> str:
    return state.get("agent_code_name") or "unknown"


async def _structured_call(
    config: RunnableConfig,
    *,
    node: str,
    prompt_key: str,
    context: str,
    schema,
    thinking_budget: int | None,
):
    provider = get_provider(config)
    model = resolve_model(config)
    system = resolve_prompt(config, prompt_key, prompts.DEFAULT_PROMPTS[prompt_key])
    provenance = resolve_prompt_provenance(config, prompt_key, system, prompts.PROMPT_VERSION)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": context}]
    started = time.perf_counter()
    result = await provider.generate_structured(
        messages, model, schema, thinking_budget=thinking_budget
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    usage = dict(
        make_usage_record(
            node=node, provider=provider, model=model, latency_ms=latency_ms, provenance=provenance
        )
    )
    return result, usage


# ── Guards (ported from emailAi/agent.py) ────────────────────────────────────


def validate_task(task: ExtractedTask, email: ParsedEmail) -> ExtractedTask:
    """Evidence must be a literal substring of the email; a mismatch is flagged, not trusted."""
    flags = list(task.validation_flags)
    updates: dict[str, object] = {}
    if task.evidence_quote in email.body:
        updates["evidence_valid"] = True
    else:
        flags.append("evidence_quote_not_in_email")
        updates.update({"requires_human": True, "evidence_valid": False})
    if task.due_date and task.due_date_evidence not in email.body:
        flags.append("due_date_evidence_not_in_email")
        updates.update({"due_date": None, "due_date_evidence": None, "requires_human": True})
    if flags:
        updates["validation_flags"] = flags
    return task.model_copy(update=updates)


def guard_triage(triage: TriageResult, email: ParsedEmail) -> TriageResult:
    if not email.security_flags:
        return triage
    return triage.model_copy(
        update={
            "requires_human_review": True,
            "security_flags": sorted(set(triage.security_flags + email.security_flags)),
        }
    )


def guard_task(task: ExtractedTask, email: ParsedEmail) -> ExtractedTask:
    if not email.security_flags:
        return task
    return task.model_copy(
        update={
            "requires_human": True,
            "validation_flags": sorted(set(task.validation_flags + email.security_flags)),
        }
    )


def guard_draft(draft: ReplyDraft, email: ParsedEmail) -> ReplyDraft:
    if not email.security_flags:
        return draft
    return draft.model_copy(
        update={
            "requires_human_review": True,
            "safety_flags": sorted(set(draft.safety_flags + email.security_flags)),
        }
    )


# ── Nodes ────────────────────────────────────────────────────────────────────


def _precheck(email: ParsedEmail, sender: SenderContext) -> PrecheckResult:
    """Short-circuit only on strong signals; everything else goes to the model.

    Fail-open by construction: missing context (``known_contact is None``) or
    weak hints never end the run early. Filing a real customer email as noise is
    the costly mistake; spending one classification call is not.
    """
    signals = email.header_signals
    if signals.get("auto_submitted") or signals.get("autoreply"):
        return PrecheckResult(
            short_circuit=True,
            category=Category.NOISE,
            subcategory=Subcategory.NOTIFICATION,
            reasons=["auto_submitted"],
        )
    if email.from_address.email.startswith(_BOUNCE_SENDERS):
        return PrecheckResult(
            short_circuit=True,
            category=Category.NOISE,
            subcategory=Subcategory.NOTIFICATION,
            reasons=["bounce"],
        )
    if sender.is_internal:
        return PrecheckResult(
            short_circuit=True,
            category=Category.FYI,
            subcategory=Subcategory.OTHER,
            reasons=["internal_sender"],
        )
    reasons = [name for name, on in sorted(signals.items()) if on]
    reasons += [label for label in email.gmail_labels if label in _HINT_LABELS]
    if sender.known_contact:
        reasons.append("remitente_conocido_en_crm")
    return PrecheckResult(short_circuit=False, reasons=reasons)


def precheck_node(state: dict, config: RunnableConfig) -> dict:
    record_node_invocation("precheck", _code_name(state))
    email = ParsedEmail.model_validate(state["email"])
    sender = SenderContext.model_validate(state.get("sender_context") or {})
    result = _precheck(email, sender)
    update: dict = {"precheck": result.model_dump(mode="json")}
    if result.short_circuit:
        triage = TriageResult(
            category=result.category,
            subcategory=result.subcategory,
            intent=Intent.OTHER,
            urgency=Urgency.LOW,
            sentiment=Sentiment.NEUTRAL,
            summary=f"Clasificado sin LLM ({', '.join(result.reasons)}).",
            confidence=1.0,
        )
        update["triage"] = guard_triage(triage, email).model_dump(mode="json")
    return update


async def triage_node(state: dict, config: RunnableConfig) -> dict:
    record_node_invocation("triage", _code_name(state))
    email = ParsedEmail.model_validate(state["email"])
    hints = (state.get("precheck") or {}).get("reasons") or []
    result, usage = await _structured_call(
        config,
        node="triage",
        prompt_key="EMAIL_TRIAGE",
        context=prompts.triage_context(email, hints),
        schema=TriageResult,
        thinking_budget=settings.clara_thinking_triage,
    )
    return {
        "triage": guard_triage(result, email).model_dump(mode="json"),
        "turn_usage": [usage],
    }


async def extract_task_node(state: dict, config: RunnableConfig) -> dict:
    record_node_invocation("extract_task", _code_name(state))
    email = ParsedEmail.model_validate(state["email"])
    summary = state["triage"]["summary"]
    task, usage = await _structured_call(
        config,
        node="extract_task",
        prompt_key="EMAIL_EXTRACT_TASK",
        context=prompts.task_context(email, summary),
        schema=ExtractedTask,
        thinking_budget=settings.clara_thinking_extract_task,
    )
    task = guard_task(validate_task(task, email), email)
    return {"task": task.model_dump(mode="json"), "turn_usage": [usage]}


async def draft_reply_node(state: dict, config: RunnableConfig) -> dict:
    record_node_invocation("draft_reply", _code_name(state))
    email = ParsedEmail.model_validate(state["email"])
    summary = state["triage"]["summary"]
    draft, usage = await _structured_call(
        config,
        node="draft_reply",
        prompt_key="EMAIL_DRAFT_REPLY",
        context=prompts.draft_context(email, summary, state.get("sender_prefs")),
        schema=ReplyDraft,
        thinking_budget=settings.clara_thinking_draft,
    )
    return {"reply_draft": guard_draft(draft, email).model_dump(mode="json"), "turn_usage": [usage]}


async def draft_follow_up_node(state: dict, config: RunnableConfig) -> dict:
    record_node_invocation("draft_follow_up", _code_name(state))
    email = ParsedEmail.model_validate(state["email"])
    last_sent = (state.get("follow_up_context") or {}).get("last_sent_body") or ""
    draft, usage = await _structured_call(
        config,
        node="draft_follow_up",
        prompt_key="EMAIL_DRAFT_FOLLOW_UP",
        context=prompts.follow_up_context(email, last_sent, state.get("sender_prefs")),
        schema=ReplyDraft,
        thinking_budget=settings.clara_thinking_draft,
    )
    return {"reply_draft": guard_draft(draft, email).model_dump(mode="json"), "turn_usage": [usage]}


# ── Routing ──────────────────────────────────────────────────────────────────


def route_mode(state: dict) -> str:
    return "draft_follow_up" if state.get("mode") == "follow_up" else "precheck"


def route_after_precheck(state: dict) -> str:
    return "end" if (state.get("precheck") or {}).get("short_circuit") else "triage"


def route_after_triage(state: dict) -> str:
    category = state["triage"]["category"]
    if category in (Category.NOISE.value, Category.FYI.value):
        return "end"
    if category == Category.ACTION_REQUIRED.value:
        return "extract_task"
    return "draft_reply"
