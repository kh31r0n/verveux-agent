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


async def serper_post(endpoint: str, payload: dict) -> dict | None:
    """POST to a Serper endpoint with exponential backoff. Returns parsed JSON
    or ``None`` on failure — callers tolerate a missing result set.

    This is the ONLY place a billable Serper request is made, so it is also the
    only place the credit counter is incremented — including every retry, which
    Serper charges for just like the first attempt.

    Raises :class:`SerperAuthError` when Serper rejects the key, which fails the
    run instead of quietly returning no results — see that class.
    """
    if not settings.serper_api_key:
        return None
    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }
    delay = 0.6
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                _count_serper_call()
                resp = await client.post(endpoint, headers=headers, json=payload)
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
                if resp.status_code == 429 and attempt < 2:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
                data = resp.json()
            return data if isinstance(data, dict) else None
        except httpx.HTTPError as exc:
            if attempt < 2:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            logger.warning("serper_request_failed", endpoint=endpoint, error=str(exc))
            return None
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
