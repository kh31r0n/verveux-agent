"""Email agent runner: the only code with side effects.

Gmail is a fake with the GmailClient surface; backend calls are patched where
the runner imports them; the graph is the real clara graph over MemorySaver
with a fake provider. Nothing leaves the process.
"""

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from src.agents.clara import nodes, runner
from src.config import settings
from src.graphs import registry
from src.graphs.clara_graph import build_clara_graph
from src.providers.errors import ProviderConfigError, StructuredOutputError
from src.schemas.email import (
    Category,
    ExtractedTask,
    Intent,
    Priority,
    ReplyDraft,
    Sentiment,
    TriageResult,
    Urgency,
)
from src.services.gmail import COMPOSE_SCOPE, READONLY_SCOPE, GmailScopeError, MissingMessageIdError

FIXTURES = Path(__file__).parent / "fixtures" / "email"
MAILBOX = "ventas@example.test"
TOKEN = {"accessToken": "tok", "scopes": [READONLY_SCOPE, COMPOSE_SCOPE], "emailAddress": MAILBOX}


def _raw(name: str, message_id: str, labels=("INBOX",), thread_id="t1") -> dict:
    data = (FIXTURES / name).read_bytes()
    return {
        "id": message_id,
        "threadId": thread_id,
        "labelIds": list(labels),
        "internalDate": "1790000000000",
        "raw": base64.urlsafe_b64encode(data).decode().rstrip("="),
    }


class FakeGmail:
    def __init__(self, raws=None, *, history=None, messages=None, thread=None, existing=None, profile="500"):
        self.raws = raws or {}
        self.history = history
        self.messages = messages or []
        self.thread = thread or {}
        self.existing = existing
        self.profile = profile
        self.created: list = []
        self.fetched: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def get_profile(self):
        return {"historyId": self.profile}

    async def list_history(self, cursor, page_token=None):
        if isinstance(self.history, Exception):
            raise self.history
        return self.history

    async def list_messages(self, query, maximum):
        return self.messages[:maximum]

    async def get_raw(self, message_id):
        self.fetched.append(message_id)
        value = self.raws[message_id]
        if isinstance(value, Exception):
            raise value
        return value

    async def get_thread(self, thread_id, headers=("From", "Date")):
        return self.thread

    async def find_existing_draft(self, thread_id, approval_id):
        return self.existing

    async def create_draft(self, raw, thread_id):
        self.created.append((raw, thread_id))
        return {"id": "draft-1", "message": {"id": "draft-msg-1"}}


class FakeProvider:
    name = "gemini"

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls: list[str] = []
        self.last_usage = SimpleNamespace(input_tokens=10, output_tokens=5, cached_input_tokens=0, reasoning_tokens=0)

    async def generate_structured(self, messages, model, schema, *, thinking_budget=None, temperature=0.0):
        self.calls.append(schema.__name__)
        value = self.outputs[schema]
        if isinstance(value, Exception):
            raise value
        return value


TRIAGE = TriageResult(
    category=Category.ACTION_REQUIRED,
    intent=Intent.QUOTATION_REQUEST,
    urgency=Urgency.MEDIUM,
    sentiment=Sentiment.NEUTRAL,
    summary="Pide cotización.",
    confidence=0.9,
)
TASK = ExtractedTask(
    title="Cotizar",
    description="Cotizar 50 licencias.",
    intent=Intent.QUOTATION_REQUEST,
    priority=Priority.MEDIUM,
    requires_human=False,
    evidence_quote="necesitamos una cotización actualizada para 50 licencias",
)
DRAFT = ReplyDraft(body="Gracias, le enviamos la cotización: [PRECIO].", language="es")
OUTPUTS = {TriageResult: TRIAGE, ExtractedTask: TASK, ReplyDraft: DRAFT}


class Backend:
    """Records every backend call the runner makes."""

    def __init__(self, sender_ctx=None, message_result=None):
        self.sender_ctx = sender_ctx or {"knownContact": False, "contactId": "", "alreadyIngested": False}
        self.message_result = message_result or {"status": "created"}
        self.messages: list[dict] = []
        self.outbound: list[dict] = []
        self.follow_ups: list[dict] = []
        self.syncs: list[tuple] = []

    async def token(self, tenant_id, mailbox_id):
        return TOKEN

    async def sender_context(self, tenant_id, mailbox_id, email, gmail_message_id):
        return self.sender_ctx

    async def message(self, payload):
        self.messages.append(payload)
        return self.message_result

    async def outbound_(self, payload):
        self.outbound.append(payload)
        return {}

    async def follow_up(self, payload):
        self.follow_ups.append(payload)
        return {}

    async def sync(self, mailbox_id, payload):
        self.syncs.append((mailbox_id, payload))
        return {}


@pytest.fixture
def backend():
    return Backend()


@pytest.fixture
def wire(backend):
    """Patch the runner's collaborators; yields a function binding Gmail + provider."""

    def _wire(gmail: FakeGmail, provider: FakeProvider | None = None, creds=None):
        provider = provider or FakeProvider(OUTPUTS)
        creds_mock = AsyncMock(return_value=creds or {"provider": "GEMINI", "model": "gemini-test"})
        patches = [
            patch.object(runner, "GmailClient", lambda token, timeout=None: gmail),
            patch.object(runner, "fetch_agent_credentials", creds_mock),
            patch.object(runner, "get_email_access_token", backend.token),
            patch.object(runner, "get_email_sender_context", backend.sender_context),
            patch.object(runner, "report_email_message", backend.message),
            patch.object(runner, "report_email_outbound", backend.outbound_),
            patch.object(runner, "report_email_follow_up", backend.follow_up),
            patch.object(runner, "report_email_sync", backend.sync),
            patch.object(nodes, "get_provider", return_value=provider),
            patch.object(nodes, "resolve_model", return_value="gemini-test"),
        ]
        for p in patches:
            p.start()
        stack.extend(patches)
        return provider

    stack: list = []
    yield _wire
    for p in stack:
        p.stop()


@pytest.fixture
def graph():
    return build_clara_graph(MemorySaver())


def _sync_req(history_id=None) -> runner.EmailSyncRequest:
    return runner.EmailSyncRequest(
        tenant_id="ten", mailbox_id="mb", agent_code_name="clara", sync_claim_token="claim-1", history_id=history_id
    )


# ── run_graph_once ───────────────────────────────────────────────────────────


async def test_run_graph_once_runs_reuses_and_resumes(graph):
    provider = FakeProvider({**OUTPUTS, ReplyDraft: StructuredOutputError("bad", kind="schema_mismatch")})
    email = runner.parse_gmail_raw_response(_raw("quotation_es.eml", "m1"))
    inputs = {"mode": "inbound", "agent_code_name": "clara", "email": email.model_dump(mode="json")}
    config = {"configurable": {"thread_id": "clara:ten:mb:m1"}}
    with patch.object(nodes, "get_provider", return_value=provider), patch.object(nodes, "resolve_model", return_value="m"):
        with pytest.raises(StructuredOutputError):
            await runner.run_graph_once(graph, config, inputs)
        assert provider.calls == ["TriageResult", "ExtractedTask", "ReplyDraft"]

        provider.outputs[ReplyDraft] = DRAFT
        values = await runner.run_graph_once(graph, config, inputs)  # resumes at draft_reply only
        assert provider.calls[3:] == ["ReplyDraft"]
        assert values["reply_draft"]["body"] == DRAFT.body

        again = await runner.run_graph_once(graph, config, inputs)  # finished: no model call
        assert len(provider.calls) == 4
        assert again["reply_draft"] == values["reply_draft"]
        assert [row["node"] for row in again["turn_usage"]] == ["triage", "extract_task", "draft_reply"]


def test_thread_id_carries_the_code_name():
    assert runner.email_thread_id("clara", "ten", "mb", "m1") == "clara:ten:mb:m1"


# ── llm_configurable ─────────────────────────────────────────────────────────


async def test_credentials_failure_never_falls_back_to_the_platform_key():
    with patch.object(runner, "fetch_agent_credentials", AsyncMock(side_effect=httpx.ConnectError("down"))):
        with pytest.raises(httpx.ConnectError):
            await runner.llm_configurable("ten")
    with patch.object(runner, "fetch_agent_credentials", AsyncMock(return_value={"provider": "OPENAI"})):
        with pytest.raises(ProviderConfigError):
            await runner.llm_configurable("ten")


# ── sync_mailbox ─────────────────────────────────────────────────────────────


async def test_first_sync_backfills_processes_and_stores_the_baseline(wire, backend, graph):
    gmail = FakeGmail(
        {"m1": _raw("quotation_es.eml", "m1"), "m2": _raw("newsletter.eml", "m2")},
        messages=[{"id": "m2"}, {"id": "m1"}],
        profile="777",
    )
    provider = wire(gmail, FakeProvider({**OUTPUTS}))
    provider.outputs[TriageResult] = TRIAGE
    await runner.sync_mailbox(_sync_req(), graph)

    assert gmail.fetched == ["m1", "m2"]  # oldest first
    assert [m["gmailMessageId"] for m in backend.messages] == ["m1", "m2"]
    first = backend.messages[0]
    assert first["status"] == "PROCESSED"
    assert first["idempotencyKey"] == "clara:ten:mb:m1"
    assert first["triage"]["category"] == "action_required"
    assert first["task"]["evidence_valid"] is True
    assert first["draft"]["body"] == DRAFT.body
    assert first["usage"] and first["usage"][0]["node"] == "triage"
    assert first["from"]["email"] == "compras@cliente.example"
    assert backend.syncs == [
        ("mb", {"tenantId": "ten", "syncClaimToken": "claim-1", "status": "COMPLETED", "historyId": "777",
                "error": None, "counts": {"processed": 2, "skipped": 0, "failed": 0, "outbound": 0}})
    ]


async def test_checkpoint_is_deleted_once_the_backend_confirms(wire, graph):
    gmail = FakeGmail({"m1": _raw("quotation_es.eml", "m1")}, messages=[{"id": "m1"}])
    wire(gmail)
    await runner.sync_mailbox(_sync_req(), graph)
    snapshot = await graph.aget_state({"configurable": {"thread_id": "clara:ten:mb:m1"}})
    assert snapshot.created_at is None


async def test_already_ingested_messages_cost_no_model_call(wire, backend, graph):
    backend.sender_ctx = {"alreadyIngested": True}
    gmail = FakeGmail({"m1": _raw("quotation_es.eml", "m1")}, messages=[{"id": "m1"}])
    provider = wire(gmail)
    await runner.sync_mailbox(_sync_req(), graph)
    assert provider.calls == []
    assert backend.messages == []
    assert backend.syncs[0][1]["counts"]["skipped"] == 1


async def test_history_cursor_advances_on_record_boundaries(wire, backend, graph, monkeypatch):
    monkeypatch.setattr(settings, "clara_max_messages_per_sync", 2)
    history = {
        "historyId": "900",
        "history": [
            {"id": "801", "messagesAdded": [{"message": {"id": "m1", "labelIds": ["INBOX"]}}]},
            {"id": "802", "messagesAdded": [{"message": {"id": "m2", "labelIds": ["INBOX"]}}, {"message": {"id": "m3", "labelIds": ["INBOX"]}}]},
        ],
    }
    gmail = FakeGmail({mid: _raw("quotation_es.eml", mid) for mid in ("m1", "m2", "m3")}, history=history)
    wire(gmail)
    await runner.sync_mailbox(_sync_req(history_id="800"), graph)
    # Record 802 would overflow the cap: stop before it, never mid-record.
    assert gmail.fetched == ["m1"]
    assert backend.syncs[0][1]["historyId"] == "801"
    assert backend.syncs[0][1]["status"] == "COMPLETED"


async def test_expired_history_falls_back_to_a_recent_query(wire, backend, graph):
    from src.services.gmail import GmailHistoryExpired

    gmail = FakeGmail({"m1": _raw("quotation_es.eml", "m1")}, history=GmailHistoryExpired("old"), messages=[{"id": "m1"}], profile="950")
    wire(gmail)
    await runner.sync_mailbox(_sync_req(history_id="10"), graph)
    assert gmail.fetched == ["m1"]
    assert backend.syncs[0][1]["historyId"] == "950"


async def test_failing_message_stops_the_sync_until_it_gives_up(wire, backend, graph):
    history = {
        "historyId": "900",
        "history": [
            {"id": "801", "messagesAdded": [{"message": {"id": "ok", "labelIds": ["INBOX"]}}]},
            {"id": "802", "messagesAdded": [{"message": {"id": "bad", "labelIds": ["INBOX"]}}]},
        ],
    }
    gmail = FakeGmail({"ok": _raw("fyi.eml", "ok"), "bad": _raw("quotation_es.eml", "bad")}, history=history)
    provider = FakeProvider({**OUTPUTS})
    wire(gmail, provider)

    calls = {"n": 0}

    async def flaky(messages, model, schema, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return TRIAGE.model_copy(update={"category": Category.FYI})
        raise StructuredOutputError("broken json", kind="invalid_json")

    provider.generate_structured = flaky
    backend.message_result = {"giveUp": False}
    await runner.sync_mailbox(_sync_req(history_id="800"), graph)
    failed = [m for m in backend.messages if m["status"] == "FAILED"]
    assert failed and failed[0]["gmailMessageId"] == "bad" and failed[0]["permanent"] is False
    report = backend.syncs[0][1]
    assert report["status"] == "FAILED"
    assert report["historyId"] == "801"  # progress kept up to the failing message

    backend.syncs.clear()
    backend.message_result = {"giveUp": True}
    await runner.sync_mailbox(_sync_req(history_id="801"), graph)
    assert backend.syncs[0][1]["status"] == "COMPLETED"
    assert backend.syncs[0][1]["historyId"] == "900"


async def test_provider_config_error_fails_the_whole_run_without_counting_attempts(wire, backend, graph):
    class Forbidden(Exception):
        code = 403

    gmail = FakeGmail({"m1": _raw("quotation_es.eml", "m1")}, messages=[{"id": "m1"}])
    wire(gmail, FakeProvider({TriageResult: Forbidden("PERMISSION_DENIED")}))
    await runner.sync_mailbox(_sync_req(), graph)
    assert backend.messages == []
    assert backend.syncs[0][1]["status"] == "FAILED"
    assert backend.syncs[0][1]["historyId"] is None


async def test_scope_violation_fails_before_reading_mail(wire, backend, graph):
    gmail = FakeGmail({}, messages=[])
    wire(gmail)

    async def bad_token(tenant_id, mailbox_id):
        return {**TOKEN, "scopes": [*TOKEN["scopes"], "https://www.googleapis.com/auth/gmail.send"]}

    with patch.object(runner, "get_email_access_token", bad_token):
        await runner.sync_mailbox(_sync_req(), graph)
    assert gmail.fetched == []
    assert "GmailScopeError" in backend.syncs[0][1]["error"]


async def test_outbound_messages_are_reported_not_triaged(wire, backend, graph):
    sent = _raw("quotation_es.eml", "s1", labels=("SENT",))
    gmail = FakeGmail({"s1": sent}, messages=[{"id": "s1"}])
    provider = wire(gmail)
    await runner.sync_mailbox(_sync_req(), graph)
    assert provider.calls == []
    assert [o["gmailMessageId"] for o in backend.outbound] == ["s1"]
    assert backend.syncs[0][1]["counts"]["outbound"] == 1


async def test_deleted_and_draft_messages_are_skipped(wire, backend, graph):
    missing = httpx.HTTPStatusError(
        "404", request=httpx.Request("GET", "https://x"), response=httpx.Response(404)
    )
    history = {
        "historyId": "10",
        "history": [
            {"id": "2", "messagesAdded": [{"message": {"id": "gone", "labelIds": ["INBOX"]}}]},
            {"id": "3", "messagesAdded": [{"message": {"id": "d", "labelIds": ["DRAFT"]}}]},
        ],
    }
    gmail = FakeGmail({"gone": missing}, history=history)
    wire(gmail)
    await runner.sync_mailbox(_sync_req(history_id="1"), graph)
    assert gmail.fetched == ["gone"]
    assert backend.syncs[0][1]["counts"]["skipped"] == 2
    assert backend.syncs[0][1]["historyId"] == "10"


async def test_unparseable_message_is_given_up_at_once(wire, backend, graph):
    raw = _raw("quotation_es.eml", "m1")
    raw["raw"] = base64.urlsafe_b64encode(b"Subject: sin remitente\n\nhola").decode()
    gmail = FakeGmail({"m1": raw}, messages=[{"id": "m1"}])
    wire(gmail)
    await runner.sync_mailbox(_sync_req(), graph)
    assert backend.messages[0]["status"] == "FAILED" and backend.messages[0]["permanent"] is True
    assert backend.syncs[0][1]["status"] == "COMPLETED"


async def test_public_domain_sender_to_same_public_domain_is_not_internal():
    raw = _raw("public_domain_sender.eml", "m1")
    email = runner.parse_gmail_raw_response(raw)
    ctx = runner.sender_context(email, "ventas.tienda@gmail.com", {})
    assert ctx.is_public_provider is True and ctx.is_internal is False
    internal = runner.parse_gmail_raw_response(_raw("internal_sender.eml", "m2"))
    assert runner.sender_context(internal, MAILBOX, {}).is_internal is True


# ── Preferences ──────────────────────────────────────────────────────────────


@pytest.fixture
def store():
    s = InMemoryStore()
    registry.set_store(s)
    yield s
    registry.set_store(None)


async def test_edited_replies_are_kept_per_sender_deduped_and_capped(store):
    for i in range(5):
        await runner.record_edited_reply("ten", "ana@x.example", f"ap-{i}", f"cuerpo {i}", "")
    await runner.record_edited_reply("ten", "ana@x.example", "ap-4", "cuerpo 4 bis", "más breve")
    prefs = await runner.load_sender_prefs("ten", "ana@x.example")
    assert [e["approval_id"] for e in prefs["examples"]] == ["ap-2", "ap-3", "ap-4"]
    assert prefs["examples"][-1]["body"] == "cuerpo 4 bis"
    assert prefs["notes"] == [{"approval_id": "ap-4", "text": "más breve"}]
    assert await runner.load_sender_prefs("otro-tenant", "ana@x.example") == {}


async def test_missing_store_degrades_to_no_memory():
    registry.set_store(None)
    await runner.record_edited_reply("ten", "a@x.example", "ap", "b", "")
    assert await runner.load_sender_prefs("ten", "a@x.example") == {}


# ── Drafts ───────────────────────────────────────────────────────────────────


def _draft_req(**overrides) -> runner.EmailDraftRequest:
    data = dict(
        tenant_id="ten",
        mailbox_id="mb",
        agent_code_name="clara",
        approval_id="ap-1",
        gmail_thread_id="t1",
        reply_to_gmail_message_id="m1",
        final_body="Gracias por escribir.",
    )
    data.update(overrides)
    return runner.EmailDraftRequest(**data)


async def test_approved_draft_is_created_in_the_thread(wire, store):
    gmail = FakeGmail({"m1": _raw("quotation_es.eml", "m1")})
    wire(gmail)
    result = await runner.create_approved_draft(_draft_req())
    assert result == {"gmail_draft_id": "draft-1", "gmail_draft_message_id": "draft-msg-1", "already_existed": False}
    raw, thread_id = gmail.created[0]
    assert thread_id == "t1"
    decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode()
    assert "To: Compras Cliente <compras@cliente.example>" in decoded
    assert "X-Clara-Approval-Id: ap-1" in decoded
    assert f"From: {MAILBOX}" in decoded
    assert await runner.load_sender_prefs("ten", "compras@cliente.example") == {}


async def test_edited_draft_is_remembered(wire, store):
    gmail = FakeGmail({"m1": _raw("quotation_es.eml", "m1")})
    wire(gmail)
    await runner.create_approved_draft(_draft_req(edited=True, reviewer_note="tutear"))
    prefs = await runner.load_sender_prefs("ten", "compras@cliente.example")
    assert prefs["examples"][0]["body"] == "Gracias por escribir."


async def test_retried_draft_is_not_duplicated(wire):
    gmail = FakeGmail({"m1": _raw("quotation_es.eml", "m1")}, existing="draft-msg-0")
    wire(gmail)
    result = await runner.create_approved_draft(_draft_req())
    assert result["already_existed"] is True
    assert gmail.created == []


async def test_draft_refused_without_message_id(wire):
    raw = _raw("quotation_es.eml", "m1")
    body = base64.urlsafe_b64decode(raw["raw"] + "==").decode().replace("Message-ID: <quotation-es@cliente.example>\n", "")
    raw["raw"] = base64.urlsafe_b64encode(body.encode()).decode()
    gmail = FakeGmail({"m1": raw})
    wire(gmail)
    with pytest.raises(MissingMessageIdError):
        await runner.create_approved_draft(_draft_req())
    assert gmail.created == []


async def test_draft_refused_with_extra_scopes(wire):
    gmail = FakeGmail({"m1": _raw("quotation_es.eml", "m1")})
    wire(gmail)

    async def bad_token(tenant_id, mailbox_id):
        return {**TOKEN, "scopes": ["https://mail.google.com/"]}

    with patch.object(runner, "get_email_access_token", bad_token):
        with pytest.raises(GmailScopeError):
            await runner.create_approved_draft(_draft_req())
    assert gmail.fetched == []


# ── Follow-ups ───────────────────────────────────────────────────────────────


def _follow_req() -> runner.EmailFollowUpRequest:
    return runner.EmailFollowUpRequest(
        tenant_id="ten", mailbox_id="mb", agent_code_name="clara",
        email_thread_id="et-1", gmail_thread_id="t1", follow_up_number=1,
    )


def _meta(message_id, sender, labels, when):
    return {"id": message_id, "labelIds": labels, "internalDate": when,
            "payload": {"headers": [{"name": "From", "value": sender}]}}


async def test_follow_up_replies_to_the_last_customer_message(wire, backend, graph):
    thread = {"messages": [
        _meta("sent-1", f"Ventas <{MAILBOX}>", ["SENT"], "2"),
        _meta("in-1", "Compras <compras@cliente.example>", ["INBOX"], "1"),
        _meta("draft", f"<{MAILBOX}>", ["DRAFT"], "3"),
    ]}
    gmail = FakeGmail(
        {"sent-1": _raw("fyi.eml", "sent-1", labels=("SENT",)), "in-1": _raw("quotation_es.eml", "in-1")},
        thread=thread,
    )
    provider = wire(gmail)
    await runner.generate_follow_up(_follow_req(), graph)
    assert provider.calls == ["ReplyDraft"]
    report = backend.follow_ups[0]
    assert report["status"] == "CREATED"
    assert report["idempotencyKey"] == "followup:et-1:1"
    assert report["replyToGmailMessageId"] == "in-1"
    assert report["draft"]["body"] == DRAFT.body
    assert report["usage"][0]["node"] == "draft_follow_up"


async def test_follow_up_without_a_sent_message_is_skipped(wire, backend, graph):
    thread = {"messages": [_meta("in-1", "compras@cliente.example", ["INBOX"], "1")]}
    gmail = FakeGmail({}, thread=thread)
    provider = wire(gmail)
    await runner.generate_follow_up(_follow_req(), graph)
    assert provider.calls == []
    assert backend.follow_ups[0]["status"] == "SKIPPED"
