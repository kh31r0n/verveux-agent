"""Shared Serper (Google SERP) client.

Two agents bill against the same Serper account: aurora (prospecting) discovers
organizations by niche, and sherlock (enrichment) resolves one known business
name to its official website. They send different payloads and have different
fallback rules, so only the parts that MUST be shared live here:

* the HTTP call itself, with its retry/auth policy — duplicating a billing-
  critical retry loop is how two code paths end up with two different ideas of
  what a 429 costs;
* the run-scoped credit counter — see the ContextVar note below, duplicating it
  would silently split the count in two;
* the response parsers, so both callers agree on the shape of a "hit".

Query construction, geo biasing and per-agent gating stay with each agent.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar

import httpx
import structlog

from ..config import settings

logger = structlog.get_logger(__name__)

SERPER_ENDPOINT = "https://google.serper.dev/search"
PLACES_ENDPOINT = "https://google.serper.dev/places"
ACCOUNT_ENDPOINT = "https://google.serper.dev/account"
# Page text through Serper (ismael's research): reads pages behind a bot
# challenge, e.g. every moodle.org host. ~10 credits a page, not 1.
SCRAPE_ENDPOINT = "https://scrape.serper.dev"


# ── Serper credit accounting ─────────────────────────────────────────────────

# Run-scoped counter of billable Serper requests.
#
# A ContextVar holding a MUTABLE dict, not an int: LangGraph runs nodes in child
# asyncio tasks, and each task gets a *copy* of the context — rebinding an int
# inside a node would never be visible to the report node. Every copy shares the
# same dict object, so mutating it is.
#
# The dict must be installed by the caller BEFORE `graph.ainvoke` (see the run
# wrappers in main.py); doing it from a node would only bind it in that node's
# own context copy. When it is absent the counter is a silent no-op, so nothing
# depends on the setup having happened.
_serper_calls: ContextVar[dict | None] = ContextVar("serper_calls", default=None)


def start_serper_accounting() -> None:
    """Install a fresh counter for the current run. Call before `ainvoke`."""
    _serper_calls.set({"calls": 0})


def serper_call_count() -> int:
    """Billable Serper requests made so far in this run (0 if not accounting)."""
    counter = _serper_calls.get()
    return int(counter["calls"]) if counter else 0


def _count_serper_call() -> None:
    counter = _serper_calls.get()
    if counter is not None:
        counter["calls"] += 1


class SerperAuthError(RuntimeError):
    """Serper rejected the credential itself (401/403).

    Deliberately fatal to the whole run rather than a per-query miss. A bad key
    fails every query identically, so swallowing it produces the one outcome
    that is indistinguishable from a legitimately empty result set: a COMPLETED
    run with nothing found. The boot validator only checks that SERPER_API_KEY
    is *set* — a set-but-invalid key gets this far — and on 2026-08-18 that gap
    turned a stored placeholder into ~4 minutes of 403s, 30-odd wasted retries
    and a run that would have reported success with nothing found.

    Not retried either: 401/403 is a verdict on the credential, and Serper bills
    every attempt including retries.
    """


class SerperQuotaExhaustedError(SerperAuthError):
    """The Serper account has no credits left.

    Serper answers it with ``400 {"message": "Not enough credits"}`` — not a 402,
    and not a 401/403, so before 2026-10-16 it fell into the generic retry path:
    three billed attempts per call, a geo-fallback that doubled them, and a run
    reported COMPLETED with nothing found. From 2026-10-13 to 10-16 every aurora
    run "completed" that way while the account sat at a negative balance.

    A subclass of :class:`SerperAuthError` so every path that already treats a
    rejected credential as fatal (both agents' nodes, the run wrappers) treats
    this one the same; the run wrappers check for this class first to report a
    distinct reason.
    """


def is_quota_exhausted_response(status_code: int, body: str) -> bool:
    """Is this Serper response "the account is out of credits"?

    402 is accepted whatever the body says; a 400 only when the body mentions
    credits, because Serper also answers 400 to a malformed request (e.g. a
    `location` it does not recognise) and that one is a per-query miss.
    """
    if status_code == 402:
        return True
    return status_code == 400 and "credit" in (body or "").lower()


async def serper_post(endpoint: str, payload: dict) -> dict | None:
    """POST to a Serper endpoint with exponential backoff. Returns parsed JSON
    or ``None`` on failure — callers tolerate a missing result set.

    This is the ONLY place a billable Serper request is made, so it is also the
    only place the credit counter is incremented — including every retry, which
    Serper charges for just like the first attempt.

    Raises :class:`SerperAuthError` when Serper rejects the key and
    :class:`SerperQuotaExhaustedError` when the account has no credits left;
    both fail the run instead of quietly returning no results. Any other 4xx is
    a verdict on the request itself and returns ``None`` without a retry; only
    429, 5xx and transport errors are retried.
    """
    if not settings.serper_api_key:
        return None
    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }
    delay = 0.6
    for attempt in range(3):
        status_code: int | None = None
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                _count_serper_call()
                resp = await client.post(endpoint, headers=headers, json=payload)
                status_code = resp.status_code
                if resp.status_code in (401, 403):
                    logger.error(
                        "serper_auth_failed",
                        endpoint=endpoint,
                        status_code=resp.status_code,
                    )
                    raise SerperAuthError(
                        f"Serper rejected SERPER_API_KEY with HTTP "
                        f"{resp.status_code} on {endpoint}; the stored secret is "
                        "invalid, revoked or out of plan"
                    )
                if is_quota_exhausted_response(resp.status_code, resp.text):
                    logger.error(
                        "serper_quota_exhausted",
                        endpoint=endpoint,
                        status_code=resp.status_code,
                        body=resp.text[:200],
                    )
                    raise SerperQuotaExhaustedError(
                        f"Serper has no credits left (HTTP {resp.status_code} on "
                        f"{endpoint}): {resp.text[:200]}"
                    )
                if resp.status_code == 429 and attempt < 2:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    # A verdict on THIS request (bad `location`, malformed query):
                    # the same payload fails the same way, and Serper bills every
                    # attempt — so it is a per-query miss, never retried.
                    logger.warning(
                        "serper_request_rejected",
                        endpoint=endpoint,
                        status_code=resp.status_code,
                        body=resp.text[:200],
                    )
                    return None
                resp.raise_for_status()
                data = resp.json()
            return data if isinstance(data, dict) else None
        except httpx.HTTPError as exc:
            if attempt < 2:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            logger.warning(
                "serper_request_failed",
                endpoint=endpoint,
                status_code=status_code,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return None
    return None


async def serper_account_balance() -> int | None:
    """Credits left on the Serper account, or None when it cannot be read.

    ``GET /account`` is free (not billed, not counted). Best effort: callers use
    it for an early warning, never to decide whether to run.
    """
    if not settings.serper_api_key:
        return None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                ACCOUNT_ENDPOINT, headers={"X-API-KEY": settings.serper_api_key}
            )
            resp.raise_for_status()
            balance = resp.json().get("balance")
        return int(balance) if isinstance(balance, (int, float)) else None
    except Exception as exc:  # noqa: BLE001 — advisory only
        logger.info("serper_balance_unavailable", error=str(exc))
        return None


# ── Response parsing ─────────────────────────────────────────────────────────


def parse_organic(data: dict | None) -> list[dict]:
    """Rows from a ``/search`` response: ``{title, url, snippet}``."""
    organic = (data or {}).get("organic", []) or []
    return [
        {
            "title": o.get("title", ""),
            "url": o.get("link", ""),
            "snippet": o.get("snippet", ""),
        }
        for o in organic
        if isinstance(o, dict) and o.get("link")
    ]


def parse_places(data: dict | None) -> list[dict]:
    """Rows from a ``/places`` response.

    Shaped like an organic hit (``url``/``snippet``) so callers can merge the two
    lists, plus the fields only a business listing carries. ``url`` may be empty:
    not every listing registers a website.
    """
    places = (data or {}).get("places", []) or []
    out: list[dict] = []
    for p in places:
        if not isinstance(p, dict):
            continue
        out.append(
            {
                "title": p.get("title", ""),
                "url": p.get("website", "") or "",
                "snippet": p.get("address", "") or "",
                "phone": p.get("phoneNumber", "") or "",
                "address": p.get("address", "") or "",
            }
        )
    return out
