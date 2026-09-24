"""Minimal Gmail REST client for the email agent. Deliberately has no send operation.

The agent never holds a refresh token: every run asks the backend for a
short-lived access token and passes it here. Scopes are checked on every use —
a token carrying anything beyond read + compose is refused (fail closed), so
this codebase stays physically unable to send mail.
"""

from __future__ import annotations

import asyncio
import base64
import random
from email.message import EmailMessage
from email.policy import SMTP
from email.utils import formataddr
from typing import Iterable

import httpx

from ..schemas.email import EmailAddress, ParsedEmail

GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"
READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
COMPOSE_SCOPE = "https://www.googleapis.com/auth/gmail.compose"
ALLOWED_SCOPES = frozenset({READONLY_SCOPE, COMPOSE_SCOPE})

APPROVAL_HEADER = "X-Clara-Approval-Id"
_TRANSIENT_ATTEMPTS = 3


class GmailError(RuntimeError):
    kind = "gmail_error"


class GmailScopeError(GmailError):
    kind = "gmail_scope"


class GmailAuthError(GmailError):
    kind = "gmail_auth"


class GmailHistoryExpired(GmailError):
    kind = "gmail_history_expired"


class GmailUnavailable(GmailError):
    kind = "gmail_unavailable"


class MissingMessageIdError(GmailError):
    kind = "missing_message_id"


def assert_allowed_scopes(granted: Iterable[str]) -> None:
    """Exactly read + compose. Extra scopes (send, modify, full) and missing ones both fail."""
    scopes = {s for s in granted if s}
    if scopes != ALLOWED_SCOPES:
        extra = sorted(scopes - ALLOWED_SCOPES)
        missing = sorted(ALLOWED_SCOPES - scopes)
        raise GmailScopeError(f"Gmail token scopes rejected (extra={extra}, missing={missing})")


class GmailClient:
    """Async Gmail client bound to one short-lived access token."""

    def __init__(
        self,
        access_token: str,
        *,
        timeout: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=GMAIL_API,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> "GmailClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        last: Exception | None = None
        for attempt in range(1, _TRANSIENT_ATTEMPTS + 1):
            try:
                resp = await self._client.request(method, path, **kwargs)
            except httpx.RequestError as exc:
                last = exc
            else:
                if resp.status_code in (401, 403):
                    raise GmailAuthError(f"Gmail refused the token ({resp.status_code}): {resp.text[:300]}")
                if resp.status_code == 429 or resp.status_code >= 500:
                    last = GmailUnavailable(f"Gmail unavailable ({resp.status_code}): {resp.text[:300]}")
                else:
                    resp.raise_for_status()
                    return resp.json() if resp.content else {}
            if attempt < _TRANSIENT_ATTEMPTS:
                await asyncio.sleep(2 ** (attempt - 1) + random.uniform(0, 0.5))
        if isinstance(last, GmailUnavailable):
            raise last
        raise GmailUnavailable(f"Gmail request failed: {type(last).__name__}: {last}")

    async def get_profile(self) -> dict:
        return await self._request("GET", "/profile")

    async def list_history(self, start_history_id: str, page_token: str | None = None) -> dict:
        params = {"startHistoryId": start_history_id, "historyTypes": "messageAdded"}
        if page_token:
            params["pageToken"] = page_token
        try:
            return await self._request("GET", "/history", params=params)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise GmailHistoryExpired("Gmail history cursor is too old") from exc
            raise

    async def list_messages(self, query: str, maximum: int) -> list[dict]:
        found: list[dict] = []
        page_token: str | None = None
        while len(found) < maximum:
            params = {"q": query, "maxResults": min(500, maximum - len(found))}
            if page_token:
                params["pageToken"] = page_token
            data = await self._request("GET", "/messages", params=params)
            found.extend(data.get("messages", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        return found[:maximum]

    async def get_raw(self, message_id: str) -> dict:
        return await self._request("GET", f"/messages/{message_id}", params={"format": "raw"})

    async def get_thread(self, thread_id: str, headers: Iterable[str] = ("From", "Date")) -> dict:
        params = [("format", "metadata"), *(("metadataHeaders", h) for h in headers)]
        return await self._request("GET", f"/threads/{thread_id}", params=params)

    async def create_draft(self, raw: str, thread_id: str) -> dict:
        return await self._request(
            "POST", "/drafts", json={"message": {"raw": raw, "threadId": thread_id}}
        )

    async def find_existing_draft(self, thread_id: str, approval_id: str) -> str | None:
        """Message id of a draft in the thread already carrying this approval, if any.

        Makes draft creation idempotent across retries: a re-drive after a crash
        between drafts.create and the backend's acknowledgement finds it here.
        """
        thread = await self.get_thread(thread_id, headers=(APPROVAL_HEADER,))
        for message in thread.get("messages", []):
            if "DRAFT" not in (message.get("labelIds") or []):
                continue
            for header in (message.get("payload") or {}).get("headers", []):
                if header.get("name", "").lower() == APPROVAL_HEADER.lower() and header.get("value") == approval_id:
                    return message.get("id")
        return None


def header_value(message: dict, name: str) -> str:
    for header in (message.get("payload") or {}).get("headers", []):
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def allowed_reply_recipients(
    email: ParsedEmail, include_original_cc: bool = False
) -> tuple[EmailAddress, list[EmailAddress]]:
    """The only allowed recipients are the original sender and, if confirmed, original CC."""

    recipient = email.from_address
    cc: list[EmailAddress] = []
    if include_original_cc:
        seen = {recipient.email}
        for address in email.cc:
            if address.email not in seen:
                cc.append(address)
                seen.add(address.email)
    return recipient, cc


def _reply_subject(subject: str) -> str:
    return subject if subject.lower().startswith("re:") else f"Re: {subject}".strip()


def _format_address(address: EmailAddress) -> str:
    return formataddr((address.name, address.email)) if address.name else address.email


def build_draft_raw(
    email: ParsedEmail,
    body: str,
    from_address: str,
    *,
    include_original_cc: bool = False,
    approval_id: str,
) -> str:
    """MIME for a reply in the original thread; this only builds a draft payload.

    ``EmailMessage(policy=SMTP)`` encodes non-ASCII headers (RFC 2047) and
    refuses CR/LF inside header values, so text copied from the inbound email
    cannot inject headers.
    """

    if not email.rfc_message_id:
        raise MissingMessageIdError("Cannot safely reply: original email has no RFC Message-ID header.")
    reply_to, cc = allowed_reply_recipients(email, include_original_cc)
    message = EmailMessage(policy=SMTP)
    message["From"] = from_address
    message["To"] = _format_address(reply_to)
    if cc:
        message["Cc"] = ", ".join(_format_address(address) for address in cc)
    message["Subject"] = _reply_subject(email.subject)
    message["In-Reply-To"] = email.rfc_message_id
    message["References"] = " ".join(dict.fromkeys([*email.references, email.rfc_message_id]))
    message[APPROVAL_HEADER] = approval_id
    message.set_content(body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii").rstrip("=")
