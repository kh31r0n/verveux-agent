"""Gmail client: least privilege, recipient allowlist, threading, idempotent drafts.

Ported from emailAi/tests/test_gmail_client.py; the client runs against an
httpx MockTransport, never the network.
"""

import base64
from email import policy
from email.parser import BytesParser

import httpx
import pytest

from src.schemas.email import EmailAddress, ParsedEmail
from src.services import gmail as gmail_module
from src.services.gmail import (
    ALLOWED_SCOPES,
    APPROVAL_HEADER,
    COMPOSE_SCOPE,
    READONLY_SCOPE,
    GmailAuthError,
    GmailClient,
    GmailHistoryExpired,
    GmailScopeError,
    GmailUnavailable,
    MissingMessageIdError,
    allowed_reply_recipients,
    assert_allowed_scopes,
    build_draft_raw,
)


def _email(**overrides) -> ParsedEmail:
    data = dict(
        gmail_message_id="g1",
        gmail_thread_id="t1",
        rfc_message_id="<original@example.com>",
        references=["<older@example.com>"],
        from_address=EmailAddress(email="ana@example.com", name="Ana"),
        cc=[EmailAddress(email="team@example.com"), EmailAddress(email="ana@example.com")],
        subject="Consulta",
        body="Hola",
    )
    data.update(overrides)
    return ParsedEmail(**data)


def _decode(raw: str):
    return BytesParser(policy=policy.default).parsebytes(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _instant(_seconds):
        return None

    monkeypatch.setattr(gmail_module.asyncio, "sleep", _instant)


def test_recipient_restriction_never_adds_addresses():
    to, cc = allowed_reply_recipients(_email(), include_original_cc=True)
    assert to.email == "ana@example.com"
    assert [address.email for address in cc] == ["team@example.com"]
    assert allowed_reply_recipients(_email())[1] == []


def test_draft_headers_stay_in_original_thread():
    raw = build_draft_raw(_email(), "Gracias.", "ventas@example.test", include_original_cc=True, approval_id="ap-1")
    message = _decode(raw)
    assert message["To"] == "Ana <ana@example.com>"
    assert message["Cc"] == "team@example.com"
    assert message["Subject"] == "Re: Consulta"
    assert message["In-Reply-To"] == "<original@example.com>"
    assert message["References"] == "<older@example.com> <original@example.com>"
    assert message[APPROVAL_HEADER] == "ap-1"


def test_cc_only_when_confirmed():
    message = _decode(build_draft_raw(_email(), "Gracias.", "ventas@example.test", approval_id="ap-1"))
    assert message["Cc"] is None


def test_draft_requires_message_id():
    with pytest.raises(MissingMessageIdError):
        build_draft_raw(_email(rfc_message_id=None), "Gracias.", "ventas@example.test", approval_id="ap")


def test_header_injection_in_subject_is_refused():
    with pytest.raises(ValueError):
        build_draft_raw(
            _email(subject="Hola\r\nBcc: attacker@example.com"), "x", "ventas@example.test", approval_id="ap"
        )


def test_non_ascii_subject_is_encoded():
    message = _decode(build_draft_raw(_email(subject="Cotización ñandú"), "x", "v@example.test", approval_id="a"))
    assert message["Subject"] == "Re: Cotización ñandú"


def test_scope_allowlist_is_exact():
    assert_allowed_scopes(ALLOWED_SCOPES)
    with pytest.raises(GmailScopeError):
        assert_allowed_scopes([READONLY_SCOPE, COMPOSE_SCOPE, "https://www.googleapis.com/auth/gmail.send"])
    with pytest.raises(GmailScopeError):
        assert_allowed_scopes([READONLY_SCOPE])
    with pytest.raises(GmailScopeError):
        assert_allowed_scopes(["https://mail.google.com/"])


def _client(handler) -> GmailClient:
    return GmailClient("token", transport=httpx.MockTransport(handler))


async def test_find_existing_draft_matches_only_drafts_with_our_header():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/threads/t1")
        assert request.headers["Authorization"] == "Bearer token"
        return httpx.Response(
            200,
            json={
                "messages": [
                    {"id": "sent", "labelIds": ["SENT"], "payload": {"headers": [{"name": APPROVAL_HEADER, "value": "ap-1"}]}},
                    {"id": "other-draft", "labelIds": ["DRAFT"], "payload": {"headers": [{"name": APPROVAL_HEADER, "value": "ap-2"}]}},
                    {"id": "ours", "labelIds": ["DRAFT"], "payload": {"headers": [{"name": APPROVAL_HEADER, "value": "ap-1"}]}},
                ]
            },
        )

    async with _client(handler) as gmail:
        assert await gmail.find_existing_draft("t1", "ap-1") == "ours"
        assert await gmail.find_existing_draft("t1", "ap-3") is None


async def test_auth_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, text="invalid")

    async with _client(handler) as gmail:
        with pytest.raises(GmailAuthError):
            await gmail.get_profile()
    assert len(calls) == 1


async def test_transient_errors_retry_then_fail():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="down")

    async with _client(handler) as gmail:
        with pytest.raises(GmailUnavailable):
            await gmail.get_profile()
    assert len(calls) == 3


async def test_transient_error_recovers():
    responses = iter([httpx.Response(429), httpx.Response(200, json={"historyId": "9"})])

    async with _client(lambda request: next(responses)) as gmail:
        assert (await gmail.get_profile())["historyId"] == "9"


async def test_expired_history_is_typed():
    async with _client(lambda request: httpx.Response(404, json={})) as gmail:
        with pytest.raises(GmailHistoryExpired):
            await gmail.list_history("1")


async def test_create_draft_posts_to_the_thread():
    seen = {}

    def handler(request):
        seen["body"] = request.read()
        return httpx.Response(200, json={"id": "d1", "message": {"id": "m9"}})

    async with _client(handler) as gmail:
        created = await gmail.create_draft("cmF3", "t1")
    assert created["id"] == "d1"
    assert b'"threadId":"t1"' in seen["body"].replace(b" ", b"")
