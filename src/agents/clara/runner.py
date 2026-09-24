"""Side effects for the email agent — the ONLY module that talks to Gmail and the backend.

The graph (``src/graphs/clara_graph.py``) is pure reasoning over one sanitized
email. Everything with an effect happens here, in plain Python:

* ``sync_mailbox``: read new mail since the mailbox cursor, run the graph per
  inbound message, report results to the backend, report SENT messages so the
  backend can track "awaiting reply".
* ``generate_follow_up``: draft a follow-up for a thread the backend says is due.
* ``create_approved_draft``: after a human approved/edited a reply in the CRM,
  write it as a Gmail DRAFT (never sent) in the original thread.

Idempotency without a workflow engine:
* the backend dedupes on ``idempotency_key`` and answers ``alreadyIngested``
  before any model call;
* one checkpoint thread per message lets a crashed run resume after the nodes
  that already finished instead of paying for them again; the thread is deleted
  once the backend confirms ingestion (bodies and ``configurable`` metadata are
  not kept);
* drafts carry ``X-Clara-Approval-Id`` and a retry finds the existing one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parseaddr

import httpx
import structlog
from pydantic import BaseModel, Field

from ...config import settings
from ...graphs.registry import get_store_or_none
from ...observability import (
    email_drafts_created_total,
    email_messages_processed_total,
    email_security_flags_total,
    email_sync_runs_total,
)
from ...providers.errors import ProviderConfigError, is_provider_config_error
from ...schemas.email import ParsedEmail, SenderContext
from ...services.email_parser import parse_gmail_raw_response
from ...services.gmail import (
    GmailClient,
    GmailError,
    GmailHistoryExpired,
    assert_allowed_scopes,
    build_draft_raw,
    header_value,
)
from ..backend_client import (
    fetch_agent_credentials,
    get_email_access_token,
    get_email_sender_context,
    report_email_follow_up,
    report_email_message,
    report_email_outbound,
    report_email_sync,
)
from ..dedup import email_domain, is_public_email_domain

logger = structlog.get_logger(__name__)

PREFS_NAMESPACE = "clara_sender_prefs"
_PREFS_KEEP = 3
_PREFS_BODY_CHARS = 1500
_EXPIRED_HISTORY_QUERY = "in:inbox newer_than:2d"


# ── Requests (validated by the FastAPI endpoints) ────────────────────────────


class EmailSyncRequest(BaseModel):
    tenant_id: str
    mailbox_id: str
    agent_code_name: str = Field(min_length=1)
    sync_claim_token: str = Field(min_length=1)
    history_id: str | None = None
    prompts: dict = Field(default_factory=dict)


class EmailFollowUpRequest(BaseModel):
    tenant_id: str
    mailbox_id: str
    agent_code_name: str = Field(min_length=1)
    email_thread_id: str
    gmail_thread_id: str
    follow_up_number: int = Field(ge=1)
    prompts: dict = Field(default_factory=dict)


class EmailDraftRequest(BaseModel):
    tenant_id: str
    mailbox_id: str
    agent_code_name: str = Field(min_length=1)
    approval_id: str
    gmail_thread_id: str
    reply_to_gmail_message_id: str
    final_body: str = Field(min_length=1, max_length=20_000)
    include_cc: bool = False
    edited: bool = False
    reviewer_note: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────


def email_thread_id(code_name: str, tenant_id: str, mailbox_id: str, key: str) -> str:
    """Checkpoint thread for one email-agent run.

    The code name is the first segment so a connection switched to another email
    builder can never resume this graph's checkpoints with a different topology.
    """
    return f"{code_name}:{tenant_id}:{mailbox_id}:{key}"


async def llm_configurable(tenant_id: str) -> dict:
    """The tenant's LLM credentials for ``configurable``.

    Unlike /prospecting/run this never falls back to the platform OpenAI key: a
    silent fallback is exactly what made failed runs report success before
    (see ``providers/errors.py``). A failure here fails the run.
    """
    creds = await fetch_agent_credentials(tenant_id)
    provider = str(creds.get("provider") or "").lower()
    cfg: dict = {"llm_provider": provider, "llm_model": creds.get("model") or ""}
    if provider in ("openai", "anthropic"):
        key = creds.get("apiKey") or ""
        if not key:
            raise ProviderConfigError(f"tenant {tenant_id} has no {provider} API key")
        cfg[f"{provider}_api_key"] = key
    elif provider == "gemini":
        cfg["gemini_credentials"] = creds.get("geminiCredentials") or {}
        cfg["gemini_project_id"] = creds.get("geminiProjectId") or ""
        cfg["gemini_location"] = creds.get("geminiLocation") or ""
    else:
        raise ProviderConfigError(f"unsupported LLM provider {provider!r} for tenant {tenant_id}")
    return cfg


def _graph_config(thread_id: str, llm_cfg: dict, prompts: dict) -> dict:
    return {"configurable": {"thread_id": thread_id, "prompts": prompts or {}, **llm_cfg}}


async def run_graph_once(graph, config: dict, inputs: dict) -> dict:
    """Run a graph at most once per thread, resuming instead of repeating.

    With langgraph 1.1.x a thread that does not exist yet and one that finished
    both report ``next == ()``; only ``created_at`` tells them apart.
    """
    snapshot = await graph.aget_state(config)
    if snapshot.next:
        return await graph.ainvoke(None, config)
    if snapshot.created_at is not None:
        return dict(snapshot.values)
    return await graph.ainvoke(inputs, config)


async def _forget_thread(graph, thread_id: str) -> None:
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None or not hasattr(checkpointer, "adelete_thread"):
        return
    try:
        await checkpointer.adelete_thread(thread_id)
    except Exception as exc:  # noqa: BLE001 — retention cleanup must not fail the run
        logger.warning("clara_checkpoint_delete_failed", thread_id=thread_id, error=str(exc))


def sender_context(email: ParsedEmail, mailbox_address: str, backend_ctx: dict) -> SenderContext:
    domain = email_domain(email.from_address.email)
    public = is_public_email_domain(domain)
    return SenderContext(
        sender_domain=domain,
        is_public_provider=public,
        is_internal=bool(domain) and not public and domain == email_domain(mailbox_address),
        known_contact=backend_ctx.get("knownContact"),
        contact_id=backend_ctx.get("contactId") or "",
    )


async def load_sender_prefs(tenant_id: str, sender_email: str) -> dict:
    store = get_store_or_none()
    if store is None:
        return {}
    try:
        item = await store.aget((PREFS_NAMESPACE, tenant_id), sender_email)
    except Exception as exc:  # noqa: BLE001 — memory is best-effort
        logger.warning("clara_prefs_read_failed", error=str(exc))
        return {}
    return dict(item.value) if item is not None else {}


async def record_edited_reply(
    tenant_id: str, sender_email: str, approval_id: str, body: str, note: str
) -> None:
    """Remember how a human edited a reply to this sender.

    Deterministic and human-gated: written only when a person edited a draft in
    the CRM, keyed by approval so a re-driven draft is not stored twice. No model
    ever rewrites preferences, so a crafted email cannot poison them.
    """
    store = get_store_or_none()
    if store is None:
        return
    current = await load_sender_prefs(tenant_id, sender_email)
    examples = [e for e in current.get("examples", []) if e.get("approval_id") != approval_id]
    examples.append({"approval_id": approval_id, "body": body[:_PREFS_BODY_CHARS]})
    notes = [n for n in current.get("notes", []) if n.get("approval_id") != approval_id]
    if note.strip():
        notes.append({"approval_id": approval_id, "text": note.strip()[:300]})
    try:
        await store.aput(
            (PREFS_NAMESPACE, tenant_id),
            sender_email,
            {"examples": examples[-_PREFS_KEEP:], "notes": notes[-_PREFS_KEEP:]},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("clara_prefs_write_failed", error=str(exc))


def _received_at(raw: dict) -> str:
    try:
        millis = int(raw.get("internalDate") or 0)
    except (TypeError, ValueError):
        millis = 0
    moment = datetime.fromtimestamp(millis / 1000, tz=timezone.utc) if millis else datetime.now(timezone.utc)
    return moment.isoformat()


def _address(value) -> dict:
    return {"email": value.email, "name": value.name}


# ── Mailbox sync ─────────────────────────────────────────────────────────────


@dataclass
class _Item:
    message_id: str
    labels: list[str] | None
    cursor_after: str | None  # history id to store once this item is fully handled


class _StopSync(Exception):
    """Stop this sync, keeping the cursor before the current item (retried next sweep)."""


async def _collect(gmail: GmailClient, cursor: str | None) -> tuple[list[_Item], str]:
    """Items to handle in order, and the cursor to store once all are handled."""
    limit = settings.clara_max_messages_per_sync
    if cursor:
        try:
            return await _history_items(gmail, cursor, limit)
        except GmailHistoryExpired:
            logger.warning("clara_history_expired", cursor=cursor)
            return await _query_items(gmail, _EXPIRED_HISTORY_QUERY, limit)
    return await _query_items(gmail, settings.clara_initial_backfill_query, limit)


async def _query_items(gmail: GmailClient, query: str, limit: int) -> tuple[list[_Item], str]:
    # Baseline first: anything arriving while we work is picked up next sync,
    # and ``alreadyIngested`` absorbs the overlap.
    baseline = str((await gmail.get_profile()).get("historyId") or "")
    refs = await gmail.list_messages(query, limit) if query else []
    items = [_Item(ref["id"], None, None) for ref in reversed(refs)]
    return items, baseline


async def _history_items(gmail: GmailClient, cursor: str, limit: int) -> tuple[list[_Item], str]:
    items: list[_Item] = []
    latest = cursor
    page_token: str | None = None
    while True:
        data = await gmail.list_history(cursor, page_token)
        latest = str(data.get("historyId") or latest)
        for record in data.get("history", []):
            added = [a.get("message") or {} for a in record.get("messagesAdded", [])]
            if items and len(items) + len(added) > limit:
                # Stop on a record boundary so the stored cursor never skips a message.
                return items, items[-1].cursor_after or cursor
            for i, message in enumerate(added):
                last_in_record = i == len(added) - 1
                items.append(
                    _Item(
                        message["id"],
                        message.get("labelIds") or [],
                        str(record.get("id")) if last_in_record else None,
                    )
                )
        page_token = data.get("nextPageToken")
        if not page_token:
            return items, latest


async def sync_mailbox(req: EmailSyncRequest, graph) -> None:
    """One sync run. Always ends with ``report_email_sync`` so the claim is released."""
    code = req.agent_code_name
    counts = {"processed": 0, "skipped": 0, "failed": 0, "outbound": 0}
    cursor = req.history_id or None
    stored_cursor: str | None = None
    status, error = "COMPLETED", ""
    try:
        llm_cfg = await llm_configurable(req.tenant_id)
        token = await get_email_access_token(req.tenant_id, req.mailbox_id)
        assert_allowed_scopes(token.get("scopes") or [])
        mailbox_address = str(token.get("emailAddress") or "").lower()
        async with GmailClient(token["accessToken"], timeout=settings.clara_gmail_timeout_seconds) as gmail:
            items, final_cursor = await _collect(gmail, cursor)
            try:
                for item in items:
                    outcome = await _handle_item(req, graph, gmail, llm_cfg, mailbox_address, item)
                    counts[outcome] += 1
                    if item.cursor_after:
                        stored_cursor = item.cursor_after
                stored_cursor = final_cursor
            except _StopSync as stop:
                status, error = "FAILED", str(stop)
    except Exception as exc:  # noqa: BLE001 — the run must always report
        status, error = "FAILED", f"{type(exc).__name__}: {exc}"
        logger.error("clara_sync_failed", mailbox_id=req.mailbox_id, error=error)
    email_sync_runs_total.labels(agent_code_name=code, outcome=status.lower()).inc()
    try:
        await report_email_sync(
            req.mailbox_id,
            {
                "tenantId": req.tenant_id,
                "syncClaimToken": req.sync_claim_token,
                "status": status,
                "historyId": stored_cursor,
                "error": error[:1000] or None,
                "counts": counts,
            },
        )
    except Exception as exc:  # noqa: BLE001 — the claim expires on its own
        logger.error("clara_sync_report_failed", mailbox_id=req.mailbox_id, error=str(exc))


async def _handle_item(
    req: EmailSyncRequest,
    graph,
    gmail: GmailClient,
    llm_cfg: dict,
    mailbox_address: str,
    item: _Item,
) -> str:
    """Handle one message; returns the counts key. Raises _StopSync to retry later."""
    code = req.agent_code_name
    if item.labels is not None and "DRAFT" in item.labels:
        return "skipped"
    try:
        raw = await gmail.get_raw(item.message_id)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return "skipped"  # deleted since it was listed; it must not stall the cursor
        raise
    labels = raw.get("labelIds") or []
    if "DRAFT" in labels:
        return "skipped"
    try:
        email = parse_gmail_raw_response(raw, settings.clara_max_body_chars)
    except ValueError as exc:
        # A message we cannot parse will never parse: give up on it at once.
        await _report_failure(req, raw, None, f"parse_error: {exc}", permanent=True)
        email_messages_processed_total.labels(agent_code_name=code, category="unknown", outcome="failed").inc()
        return "failed"

    # SENT without INBOX also covers replies sent from a send-as alias.
    if email.from_address.email == mailbox_address or ("SENT" in labels and "INBOX" not in labels):
        await report_email_outbound(
            {
                "tenantId": req.tenant_id,
                "mailboxId": req.mailbox_id,
                "gmailMessageId": email.gmail_message_id,
                "gmailThreadId": email.gmail_thread_id,
                "sentAt": _received_at(raw),
                "subject": email.subject,
                "body": email.body,
            }
        )
        return "outbound"
    if "INBOX" not in labels:
        return "skipped"

    backend_ctx = await get_email_sender_context(
        req.tenant_id, req.mailbox_id, email.from_address.email, email.gmail_message_id
    )
    if backend_ctx.get("alreadyIngested"):
        return "skipped"

    thread_id = email_thread_id(code, req.tenant_id, req.mailbox_id, email.gmail_message_id)
    inputs = {
        "mode": "inbound",
        "agent_code_name": code,
        "tenant_id": req.tenant_id,
        "mailbox_id": req.mailbox_id,
        "email": email.model_dump(mode="json"),
        "sender_context": sender_context(email, mailbox_address, backend_ctx).model_dump(mode="json"),
        "sender_prefs": await load_sender_prefs(req.tenant_id, email.from_address.email),
    }
    try:
        values = await run_graph_once(graph, _graph_config(thread_id, llm_cfg, req.prompts), inputs)
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, ProviderConfigError) or is_provider_config_error(exc):
            raise  # every later message would fail the same way — fail the whole run
        result = await _report_failure(req, raw, email, f"{type(exc).__name__}: {exc}", permanent=False)
        email_messages_processed_total.labels(agent_code_name=code, category="unknown", outcome="failed").inc()
        if result.get("giveUp"):
            return "failed"
        raise _StopSync(f"message {email.gmail_message_id} failed and will be retried: {exc}") from exc

    try:
        await report_email_message(_message_payload(req, raw, email, inputs["sender_context"], values))
    except Exception as exc:  # noqa: BLE001 — checkpoint kept: the retry costs no LLM call
        raise _StopSync(f"backend did not accept message {email.gmail_message_id}: {exc}") from exc
    await _forget_thread(graph, thread_id)

    category = (values.get("triage") or {}).get("category", "unknown")
    email_messages_processed_total.labels(agent_code_name=code, category=category, outcome="processed").inc()
    for flag in email.security_flags:
        email_security_flags_total.labels(agent_code_name=code, flag=flag).inc()
    return "processed"


def _message_payload(req: EmailSyncRequest, raw: dict, email: ParsedEmail, ctx: dict, values: dict) -> dict:
    return {
        "idempotencyKey": email_thread_id(req.agent_code_name, req.tenant_id, req.mailbox_id, email.gmail_message_id),
        "tenantId": req.tenant_id,
        "mailboxId": req.mailbox_id,
        "agentCodeName": req.agent_code_name,
        "status": "PROCESSED",
        "gmailMessageId": email.gmail_message_id,
        "gmailThreadId": email.gmail_thread_id,
        "rfcMessageId": email.rfc_message_id,
        "from": _address(email.from_address),
        "cc": [_address(a) for a in email.cc],
        "subject": email.subject,
        "receivedAt": _received_at(raw),
        "body": email.body,
        "bodyTruncated": email.body_truncated,
        "attachmentNames": email.attachment_names,
        "senderDomain": ctx.get("sender_domain") or "",
        "isPublicProvider": bool(ctx.get("is_public_provider")),
        "securityFlags": email.security_flags,
        "triage": values.get("triage"),
        "task": values.get("task"),
        "draft": values.get("reply_draft"),
        "usage": values.get("turn_usage") or [],
    }


async def _report_failure(
    req: EmailSyncRequest, raw: dict, email: ParsedEmail | None, error: str, *, permanent: bool
) -> dict:
    message_id = email.gmail_message_id if email else str(raw.get("id") or "")
    payload = {
        "idempotencyKey": email_thread_id(req.agent_code_name, req.tenant_id, req.mailbox_id, message_id),
        "tenantId": req.tenant_id,
        "mailboxId": req.mailbox_id,
        "agentCodeName": req.agent_code_name,
        "status": "FAILED",
        "permanent": permanent,
        "error": error[:1000],
        "gmailMessageId": message_id,
        "gmailThreadId": email.gmail_thread_id if email else str(raw.get("threadId") or ""),
        "receivedAt": _received_at(raw),
    }
    if email is not None:
        payload["from"] = _address(email.from_address)
        payload["subject"] = email.subject
    try:
        return await report_email_message(payload)
    except Exception as exc:  # noqa: BLE001
        raise _StopSync(f"could not record failure of {message_id}: {exc}") from exc


# ── Follow-ups ───────────────────────────────────────────────────────────────


def _is_outbound(message: dict, mailbox_address: str) -> bool:
    labels = message.get("labelIds") or []
    sender = parseaddr(header_value(message, "From"))[1].lower()
    return sender == mailbox_address or ("SENT" in labels and "INBOX" not in labels)


async def generate_follow_up(req: EmailFollowUpRequest, graph) -> None:
    code = req.agent_code_name
    base = {
        "idempotencyKey": f"followup:{req.email_thread_id}:{req.follow_up_number}",
        "tenantId": req.tenant_id,
        "mailboxId": req.mailbox_id,
        "agentCodeName": code,
        "emailThreadId": req.email_thread_id,
        "followUpNumber": req.follow_up_number,
    }
    try:
        llm_cfg = await llm_configurable(req.tenant_id)
        token = await get_email_access_token(req.tenant_id, req.mailbox_id)
        assert_allowed_scopes(token.get("scopes") or [])
        mailbox_address = str(token.get("emailAddress") or "").lower()
        async with GmailClient(token["accessToken"], timeout=settings.clara_gmail_timeout_seconds) as gmail:
            thread = await gmail.get_thread(req.gmail_thread_id)
            messages = [m for m in thread.get("messages", []) if "DRAFT" not in (m.get("labelIds") or [])]
            messages.sort(key=lambda m: int(m.get("internalDate") or 0))
            sent = [m for m in messages if _is_outbound(m, mailbox_address)]
            inbound = [m for m in messages if not _is_outbound(m, mailbox_address)]
            if not sent or not inbound:
                await report_email_follow_up({**base, "status": "SKIPPED", "reason": "no_sent_or_inbound_message"})
                return
            last_sent = parse_gmail_raw_response(await gmail.get_raw(sent[-1]["id"]), settings.clara_max_body_chars)
            last_inbound = parse_gmail_raw_response(
                await gmail.get_raw(inbound[-1]["id"]), settings.clara_max_body_chars
            )
    except Exception as exc:  # noqa: BLE001 — the backend's claim expires and retries
        logger.error("clara_follow_up_failed", email_thread_id=req.email_thread_id, error=str(exc))
        return

    thread_id = email_thread_id(
        code, req.tenant_id, req.mailbox_id, f"followup:{req.email_thread_id}:{req.follow_up_number}"
    )
    inputs = {
        "mode": "follow_up",
        "agent_code_name": code,
        "tenant_id": req.tenant_id,
        "mailbox_id": req.mailbox_id,
        "email": last_inbound.model_dump(mode="json"),
        "follow_up_context": {"last_sent_body": last_sent.body},
        "sender_prefs": await load_sender_prefs(req.tenant_id, last_inbound.from_address.email),
    }
    try:
        values = await run_graph_once(graph, _graph_config(thread_id, llm_cfg, req.prompts), inputs)
        await report_email_follow_up(
            {
                **base,
                "status": "CREATED",
                "replyToGmailMessageId": last_inbound.gmail_message_id,
                "originalCc": [_address(a) for a in last_inbound.cc],
                "draft": values.get("reply_draft"),
                "usage": values.get("turn_usage") or [],
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("clara_follow_up_failed", email_thread_id=req.email_thread_id, error=str(exc))
        return
    await _forget_thread(graph, thread_id)


# ── Approved drafts ──────────────────────────────────────────────────────────


async def create_approved_draft(req: EmailDraftRequest) -> dict:
    """Write the human-approved reply as a Gmail draft. Never sends.

    Raises ``GmailError`` subclasses (``kind`` → HTTP status in the endpoint).
    """
    code = req.agent_code_name
    token = await get_email_access_token(req.tenant_id, req.mailbox_id)
    assert_allowed_scopes(token.get("scopes") or [])
    from_address = str(token.get("emailAddress") or "")
    try:
        async with GmailClient(token["accessToken"], timeout=settings.clara_gmail_timeout_seconds) as gmail:
            original = parse_gmail_raw_response(
                await gmail.get_raw(req.reply_to_gmail_message_id), settings.clara_max_body_chars
            )
            existing = await gmail.find_existing_draft(req.gmail_thread_id, req.approval_id)
            if existing:
                email_drafts_created_total.labels(agent_code_name=code, outcome="already_existed").inc()
                return {"gmail_draft_id": None, "gmail_draft_message_id": existing, "already_existed": True}
            raw = build_draft_raw(
                original,
                req.final_body,
                from_address,
                include_original_cc=req.include_cc,
                approval_id=req.approval_id,
            )
            created = await gmail.create_draft(raw, req.gmail_thread_id)
    except GmailError as exc:
        outcome = "refused" if exc.kind in ("missing_message_id", "gmail_scope") else "failed"
        email_drafts_created_total.labels(agent_code_name=code, outcome=outcome).inc()
        raise

    if req.edited:
        await record_edited_reply(
            req.tenant_id, original.from_address.email, req.approval_id, req.final_body, req.reviewer_note
        )
    email_drafts_created_total.labels(agent_code_name=code, outcome="created").inc()
    return {
        "gmail_draft_id": created.get("id"),
        "gmail_draft_message_id": (created.get("message") or {}).get("id"),
        "already_existed": False,
    }
