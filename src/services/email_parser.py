"""MIME parsing and conservative sanitization for untrusted inbound email.

This is the only place raw email HTML/text is touched. Hidden HTML, quoted
history, invisible Unicode and common prompt-injection phrases are stripped or
flagged (``security_flags``) here, before anything reaches a model.
"""

from __future__ import annotations

import base64
import re
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses
from typing import Iterable

from bs4 import BeautifulSoup

from ..schemas.email import EmailAddress, ParsedEmail

DEFAULT_MAX_BODY_CHARS = 12_000

_QUOTE_MARKERS = re.compile(
    r"(?im)^(?:On .+ wrote:|El .+ escribi[oó]:|From: .+\nSent:|De: .+\nEnviado:|-{2,}\s*Original Message\s*-{2,})"
)
_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+|any\s+|previous\s+|prior\s+)?instructions?\b", re.I),
    re.compile(r"\bignora\s+(?:todas\s+)?(?:las\s+)?instrucciones\b", re.I),
    re.compile(r"\b(?:system|developer)\s+message\b", re.I),
    re.compile(r"\byou\s+are\s+now\b", re.I),
    re.compile(r"\bsend\s+(?:all\s+)?(?:customer\s+)?(?:data|information)\b", re.I),
)

# Zero-width characters are routine in marketing preheaders, so they are
# removed silently. Bidi controls reorder what a human sees against what the
# model reads — a known way to hide instructions — so removing one is flagged.
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
_BIDI_CONTROLS = re.compile("[‎‏‪-‮⁦-⁩]")

_NOREPLY_LOCAL_PART = re.compile(
    r"^(?:no-?reply|do-?not-?reply|donotreply|mailer-daemon|postmaster|bounce[s]?)(?:[+._-].*)?$",
    re.I,
)


def _addresses(values: Iterable[str | None]) -> list[EmailAddress]:
    parsed: list[EmailAddress] = []
    for name, address in getaddresses([value or "" for value in values]):
        if not address:
            continue
        try:
            parsed.append(EmailAddress(email=address, name=name.strip()))
        except ValueError:
            continue
    return parsed


def _is_hidden(tag) -> bool:
    style = (tag.get("style") or "").lower().replace(" ", "")
    color = (tag.get("color") or "").lower().replace(" ", "")
    hidden_style_markers = (
        "display:none",
        "visibility:hidden",
        "font-size:0",
        "font-size:0px",
        "color:transparent",
        "color:white",
        "color:#fff",
        "color:#ffffff",
        "rgba(0,0,0,0)",
    )
    return any(marker in style for marker in hidden_style_markers) or color in {
        "white",
        "#fff",
        "#ffffff",
        "transparent",
    }


def html_to_safe_text(html: str) -> tuple[str, bool]:
    """Drop active and visually hidden HTML before extracting visible text."""

    soup = BeautifulSoup(html, "html.parser")
    hidden_removed = False
    for tag in soup.find_all(["script", "style", "noscript", "template"]):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        # Descendants of a hidden tag are destroyed with it (attrs becomes None)
        # but are still in this snapshot list.
        if tag.decomposed:
            continue
        if _is_hidden(tag):
            hidden_removed = True
            tag.decompose()
    return soup.get_text("\n", strip=True), hidden_removed


def strip_invisible(text: str) -> tuple[str, bool]:
    """Remove zero-width and bidi control characters; report whether bidi ones existed."""
    had_bidi = bool(_BIDI_CONTROLS.search(text))
    text = _BIDI_CONTROLS.sub("", _ZERO_WIDTH.sub("", text))
    return text, had_bidi


def strip_quoted_history(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    marker = _QUOTE_MARKERS.search(text)
    if marker:
        text = text[: marker.start()]
    visible_lines = [line for line in text.splitlines() if not line.lstrip().startswith(">")]
    return "\n".join(visible_lines).strip()


def _security_flags(text: str, hidden_removed: bool, bidi_removed: bool) -> list[str]:
    flags: list[str] = []
    if hidden_removed:
        flags.append("hidden_content_removed")
    if bidi_removed:
        flags.append("bidi_control_removed")
    if any(pattern.search(text) for pattern in _INJECTION_PATTERNS):
        flags.append("prompt_injection_pattern")
    return flags


def _header_signals(message, sender: EmailAddress) -> dict[str, bool]:
    """Deterministic hints for the pre-LLM check. Each is a hint, never proof."""
    auto_submitted = (message.get("Auto-Submitted") or "").strip().lower()
    precedence = (message.get("Precedence") or "").strip().lower()
    local_part = sender.email.rsplit("@", 1)[0]
    return {
        "auto_submitted": bool(auto_submitted) and auto_submitted != "no",
        "autoreply": bool(message.get("X-Autoreply") or message.get("X-Autorespond")),
        "bulk": precedence in {"bulk", "list", "junk"},
        "list_unsubscribe": bool(message.get("List-Unsubscribe") or message.get("List-Id")),
        "noreply_sender": bool(_NOREPLY_LOCAL_PART.match(local_part)),
    }


def _part_text(part) -> str:
    content = part.get_content()
    if isinstance(content, bytes):
        return content.decode(part.get_content_charset() or "utf-8", errors="replace")
    return str(content)


def _message_ids(header_value: str | None) -> list[str]:
    """Extract RFC Message-ID tokens without trusting arbitrary header text."""

    return re.findall(r"<[^<>\s]+>", header_value or "")


def parse_raw_email(
    raw_message: bytes,
    *,
    gmail_message_id: str,
    gmail_thread_id: str,
    max_body_chars: int = DEFAULT_MAX_BODY_CHARS,
    gmail_labels: list[str] | None = None,
) -> ParsedEmail:
    """Parse a Gmail ``format=raw`` message without fetching attachments."""

    message = BytesParser(policy=policy.default).parsebytes(raw_message)
    plain_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[str] = []

    for part in message.walk():
        if part.is_multipart():
            continue
        disposition = (part.get_content_disposition() or "").lower()
        filename = part.get_filename()
        if filename:
            attachments.append(filename)
        if disposition == "attachment" or filename:
            continue
        content_type = part.get_content_type().lower()
        if content_type == "text/plain":
            plain_parts.append(_part_text(part))
        elif content_type == "text/html":
            html_parts.append(_part_text(part))

    # Inspect HTML even when text/plain wins: hidden HTML must still raise a
    # review flag rather than silently bypassing the security path.
    safe_html, hidden_removed = html_to_safe_text("\n".join(html_parts))
    body = "\n".join(plain_parts) if plain_parts else safe_html
    # Before the injection regexes, so "ig​nore instructions" still matches.
    body, bidi_removed = strip_invisible(body)
    body = strip_quoted_history(body)
    truncated = len(body) > max_body_chars
    if truncated:
        body = body[:max_body_chars]

    senders = _addresses([message.get("From")])
    if not senders:
        raise ValueError("email has no valid From address")
    subject, _ = strip_invisible(str(message.get("Subject") or ""))

    return ParsedEmail(
        gmail_message_id=gmail_message_id,
        gmail_thread_id=gmail_thread_id,
        rfc_message_id=(message.get("Message-ID") or "").strip() or None,
        references=_message_ids(message.get("References")),
        from_address=senders[0],
        to=_addresses(message.get_all("To", [])),
        cc=_addresses(message.get_all("Cc", [])),
        subject=subject,
        date=str(message.get("Date") or ""),
        body=body,
        attachment_names=attachments,
        security_flags=_security_flags(f"{subject}\n{body}", hidden_removed, bidi_removed),
        body_truncated=truncated,
        header_signals=_header_signals(message, senders[0]),
        gmail_labels=list(gmail_labels or []),
    )


def parse_gmail_raw_response(response: dict, max_body_chars: int = DEFAULT_MAX_BODY_CHARS) -> ParsedEmail:
    """Convert Gmail's base64url raw-message response to ``ParsedEmail``."""

    raw = response.get("raw")
    if not raw or not response.get("id") or not response.get("threadId"):
        raise ValueError("Gmail raw response is missing id, threadId, or raw content")
    raw_bytes = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    return parse_raw_email(
        raw_bytes,
        gmail_message_id=response["id"],
        gmail_thread_id=response["threadId"],
        max_body_chars=max_body_chars,
        gmail_labels=response.get("labelIds") or [],
    )
