"""Google Places API (New) text search — aurora's business-listing source.

Replaces Serper ``/places`` for prospecting (2026-10-16). Two reasons:

* it halves the Serper credits a run spends — every query used to cost one
  ``/search`` AND one ``/places`` call on the same account, and that account
  running dry stopped prospecting for three days;
* the response is structured: ``addressComponents`` names the municipality
  (``locality``), so the region check no longer depends on the LLM reading a
  city off a web page.

Rows come out in the same shape as ``serper.parse_places`` (``title``, ``url``,
``snippet``, ``phone``, ``address``) plus ``locality``, so ``web_search`` merges
them with organic hits unchanged.

Billing goes to the Maps Platform project that owns ``GOOGLE_PLACES_API_KEY``,
not to Serper. The field mask includes contact fields (website, phone), which
puts every call on the Text Search *Enterprise* SKU — keep the mask minimal.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar

import httpx
import structlog

from ..config import settings

logger = structlog.get_logger(__name__)

TEXT_SEARCH_ENDPOINT = "https://places.googleapis.com/v1/places:searchText"

FIELD_MASK = ",".join(
    [
        "places.displayName",
        "places.websiteUri",
        "places.nationalPhoneNumber",
        "places.formattedAddress",
        "places.addressComponents",
        "places.businessStatus",
    ]
)

# Run-scoped call counter — same ContextVar-holding-a-dict pattern (and the
# same reason for it) as `services.serper._serper_calls`.
_places_calls: ContextVar[dict | None] = ContextVar("places_calls", default=None)


def start_places_accounting() -> None:
    _places_calls.set({"calls": 0})


def places_call_count() -> int:
    counter = _places_calls.get()
    return int(counter["calls"]) if counter else 0


def _count_places_call() -> None:
    counter = _places_calls.get()
    if counter is not None:
        counter["calls"] += 1


class PlacesAuthError(RuntimeError):
    """Google rejected the key (invalid, revoked, or not allowed to call the
    Places API). Fatal to the run, like ``SerperAuthError``: every query would
    fail the same way."""


class PlacesQuotaExhaustedError(PlacesAuthError):
    """The project's Places quota (or billing) is exhausted — HTTP 429 with
    ``RESOURCE_EXHAUSTED``. Fatal: a run cannot finish its search without it."""


def places_configured() -> bool:
    return bool(settings.google_places_api_key.strip())


def _error_status(resp: httpx.Response) -> str:
    try:
        return str(((resp.json() or {}).get("error") or {}).get("status") or "")
    except ValueError:
        return ""


async def places_text_search(
    query: str, *, region_code: str = "co", language_code: str = "es"
) -> list[dict]:
    """One text search. Returns parsed rows; ``[]`` on a transient failure.

    Raises :class:`PlacesAuthError` / :class:`PlacesQuotaExhaustedError` for the
    failures that would hit every query identically.
    """
    if not places_configured():
        return []
    headers = {
        "X-Goog-Api-Key": settings.google_places_api_key,
        "X-Goog-FieldMask": FIELD_MASK,
        "Content-Type": "application/json",
    }
    payload = {
        "textQuery": query,
        "languageCode": language_code or "es",
        "regionCode": (region_code or "co").upper(),
        "pageSize": settings.prospecting_places_page_size,
    }
    delay = 1.0
    for attempt in range(3):
        status_code: int | None = None
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                _count_places_call()
                resp = await client.post(
                    TEXT_SEARCH_ENDPOINT, headers=headers, json=payload
                )
            status_code = resp.status_code
            if resp.status_code in (401, 403):
                logger.error(
                    "google_places_auth_failed",
                    status_code=resp.status_code,
                    status=_error_status(resp),
                )
                raise PlacesAuthError(
                    f"Google Places rejected GOOGLE_PLACES_API_KEY with HTTP "
                    f"{resp.status_code} ({_error_status(resp) or 'no status'})"
                )
            if resp.status_code == 429:
                # Per-minute rate limits clear after a pause; a daily quota or
                # a billing cap does not. Retry first, give up as exhausted.
                if attempt < 2:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                logger.error("google_places_quota_exhausted", body=resp.text[:200])
                raise PlacesQuotaExhaustedError(
                    f"Google Places quota exhausted (HTTP 429): {resp.text[:200]}"
                )
            if 400 <= resp.status_code < 500:
                logger.warning(
                    "google_places_request_rejected",
                    status_code=resp.status_code,
                    query=query,
                    body=resp.text[:200],
                )
                return []
            resp.raise_for_status()
            return parse_places_response(resp.json())
        except httpx.HTTPError as exc:
            if attempt < 2:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            logger.warning(
                "google_places_request_failed",
                status_code=status_code,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return []
    return []


def _locality(components: list) -> str:
    """The municipality a listing sits in. ``locality`` first; Colombian
    addresses outside a city proper sometimes carry only the municipality as
    ``administrative_area_level_2``."""
    by_type: dict[str, str] = {}
    for comp in components or []:
        if not isinstance(comp, dict):
            continue
        for t in comp.get("types") or []:
            by_type.setdefault(t, comp.get("longText") or "")
    return by_type.get("locality") or by_type.get("administrative_area_level_2") or ""


def parse_places_response(data: dict | None) -> list[dict]:
    out: list[dict] = []
    for p in (data or {}).get("places") or []:
        if not isinstance(p, dict):
            continue
        if p.get("businessStatus") == "CLOSED_PERMANENTLY":
            continue
        title = ((p.get("displayName") or {}).get("text") or "").strip()
        if not title:
            continue
        address = p.get("formattedAddress") or ""
        out.append(
            {
                "title": title,
                "url": p.get("websiteUri") or "",
                "snippet": address,
                "phone": p.get("nationalPhoneNumber") or "",
                "address": address,
                "locality": _locality(p.get("addressComponents") or []),
                "source": "google_places",
            }
        )
    return out
