"""ismael's web lookup of a Moodle error it cannot explain.

Support answers from the chat model's general knowledge of Moodle. A concrete
error it does not recognize got a question about the course and a generic
ticket on 2026-10-18 (a course restore failing in ``question_answers``, which
Moodle documents with its fix). So such an error is looked up in Moodle's own
documentation: one Serper query restricted to ``ismael_research_domains``, the
results filtered by domain again in code (the ``site:`` operators are a hint to
Google, not a guarantee), and up to ``ismael_research_max_pages`` pages read
through the SSRF-hardened fetcher sherlock uses. Every moodle.org host answers
that fetcher with a Cloudflare challenge (403), so a refused page is read
through Serper's scrape endpoint instead — ~10 credits a page, so only the top
``ismael_research_scrape_pages`` of them; the others keep their snippet.

The pages reach the model fenced as data (``<<<FUENTE … FUENTE>>>``). The link
the reply relies on must be one of the pages actually read, and is appended in
code — the model never writes it.

Both halves are cached platform-wide in the shared LangGraph store (the pages
are public documentation, so one tenant's lookup serves every other one): a
search by its error code or normalized query, and a page's text by its URL,
for ``ismael_research_cache_days``; a search that found nothing for
``ismael_research_negative_cache_days``. A repeated error then costs no
credits — the scrape alone is ~10 — and skips the network. Snippets are never
cached as pages; the model's answer is not cached at all (it depends on the
student's case). Store down → an in-process cache, then nothing.

Fail-open everywhere: no key, no credits, no results or a timeout leave the
first answer as it was.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import re
import time
from dataclasses import dataclass

import httpx
import structlog

from ...config import settings
from ...observability import ismael_research_cache_total
from ...services import web_fetch
from ...services.serper import (
    SCRAPE_ENDPOINT,
    SERPER_ENDPOINT,
    SerperAuthError,
    parse_organic,
    serper_post,
)
from .common import fold

logger = structlog.get_logger(__name__)

# Moodle's error page links "More information about this error" to
# docs.moodle.org/…/error/<component>/<code>: text carrying it is a pasted
# error page, a concrete error whatever the model said.
_ERROR_PAGE = re.compile(
    r"m[aá]s informaci[oó]n sobre este error|more information about this error",
    re.IGNORECASE,
)
MAX_QUERY_CHARS = 200
# Moodle's error strings have codes ("error_question_answers_missing_in_db"),
# and docs.moodle.org has a page per code: searched alone, it comes up first.
_ERROR_CODE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){2,}\b")
# Listings, not answers: a forum's topic index or a search page.
_INDEX_PAGES = re.compile(r"/mod/forum/(?:view|search|index)\.php$")
# A Moodle docs page is ~100–300 KB uncompressed; the cap only truncates.
_MAX_PAGE_BYTES = 1_500_000


@dataclass(frozen=True)
class Source:
    url: str
    title: str
    text: str


def allowed_domains() -> tuple[str, ...]:
    return tuple(
        d.strip().lower() for d in settings.ismael_research_domains.split(",") if d.strip()
    )


def on_allowed_domain(url: str, domains: tuple[str, ...]) -> bool:
    """True for an https URL on one of ``domains`` or a subdomain of one."""
    try:
        parsed = httpx.URL(url)
    except Exception:  # noqa: BLE001 — an unparsable link is simply not allowed
        return False
    host = (parsed.host or "").lower()
    return parsed.scheme == "https" and any(host == d or host.endswith("." + d) for d in domains)


def pasted_error_page(text: str) -> bool:
    return bool(_ERROR_PAGE.search(text or ""))


def query_from_text(text: str) -> str:
    """A query from a pasted error itself, for when the model gave none: the
    page's footer, quotes and escaped markup ("&lt;p&gt;") removed."""
    plain = re.sub(r"<[^>]+>", " ", html.unescape(_ERROR_PAGE.sub(" ", text or "")))
    plain = re.sub(r"[«»\"]", " ", plain)
    return " ".join(plain.split())[:MAX_QUERY_CHARS]


# ── Cache ────────────────────────────────────────────────────────────────────

_SEARCH_NS = ("ismael_research", "searches")
_PAGE_NS = ("ismael_research", "pages")
_DAY = 86_400.0
# (namespace, key) → value: the fallback when the shared store is unavailable.
_local_cache: dict[tuple[tuple[str, ...], str], dict] = {}
_LOCAL_CACHE_MAX = 256


def _store():
    # Imported at call time, like rag_job: the registry imports the graphs.
    from ...graphs.registry import get_store_or_none

    return get_store_or_none()


def _cache_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def _cache_get(namespace: tuple[str, ...], key: str) -> tuple[dict, float] | None:
    """(value, age in seconds) of a cached entry, whatever its age; None when absent."""
    value = None
    store = _store()
    if store is not None:
        try:
            item = await store.aget(namespace, key)
            value = getattr(item, "value", None) if item else None
        except Exception as exc:  # noqa: BLE001 — a cache miss, never a failure
            logger.info("ismael_research_cache_read_failed", error=str(exc) or type(exc).__name__)
    value = value or _local_cache.get((namespace, key))
    if not isinstance(value, dict):
        return None
    return value, time.time() - float(value.get("stored_at") or 0)


async def _cache_put(namespace: tuple[str, ...], key: str, value: dict) -> None:
    value = {**value, "stored_at": time.time()}
    _local_cache[(namespace, key)] = value
    while len(_local_cache) > _LOCAL_CACHE_MAX:
        _local_cache.pop(next(iter(_local_cache)))
    store = _store()
    if store is None:
        return
    try:
        await store.aput(namespace, key, value)
    except Exception as exc:  # noqa: BLE001 — the local copy still holds
        logger.info("ismael_research_cache_write_failed", error=str(exc) or type(exc).__name__)


def _search_key(query: str, domains: tuple[str, ...]) -> str:
    """One key per question Serper would be asked: the error code when there is
    one (what the docs tier searches), else the folded query."""
    code = _ERROR_CODE.search(query)
    asked = f"code:{code.group(0)}" if code else "q:" + " ".join(fold(query).split())[:MAX_QUERY_CHARS]
    return f"{','.join(domains)}|{asked}"


def _page_key(url: str) -> str:
    """docs.moodle.org/en/X and docs.moodle.org/22/en/X are one page."""
    parsed = httpx.URL(url)
    return f"{parsed.host}/{parsed.path.rstrip('/').rsplit('/', 1)[-1]}?{parsed.query.decode()}"


async def _search(query: str, domains: tuple[str, ...]) -> tuple[list[dict], bool]:
    """(hits, complete): the first domain alone, then the others only if it
    found nothing. ``complete`` is False when a query failed, so an empty
    result is not a verdict worth caching.

    One ``site:`` per query on purpose: OR-ing three of them (2026-10-21) gave
    forum index pages instead of the docs page named after the error code,
    which a ``site:docs.moodle.org`` query returns first. Usually one credit.
    """
    tiers = [domains[:1], domains[1:]] if len(domains) > 1 else [domains]
    code = _ERROR_CODE.search(query)
    complete = True
    for n, tier in enumerate(tiers):
        terms = f"error {code.group(0)}" if code and n == 0 else query[:MAX_QUERY_CHARS]
        sites = " OR ".join(f"site:{d}" for d in tier)
        data = await serper_post(SERPER_ENDPOINT, {"q": f"{terms} {sites}", "num": 5})
        complete = complete and data is not None
        hits: list[dict] = []
        for row in parse_organic(data):
            if (
                on_allowed_domain(row["url"], tier)
                and not _INDEX_PAGES.search(httpx.URL(row["url"]).path)
                and _page_key(row["url"]) not in {_page_key(h["url"]) for h in hits}
            ):
                hits.append(row)
        if hits:
            return hits[: max(1, settings.ismael_research_max_pages)], complete
    return [], complete


async def _cached_search(query: str, domains: tuple[str, ...]) -> tuple[list[dict], bool]:
    """(hits, from_cache). A search that found nothing is remembered for less
    time than one that found pages: the docs may gain a page for it."""
    key = _cache_key(_search_key(query, domains))
    cached = await _cache_get(_SEARCH_NS, key)
    if cached:
        value, age = cached
        hits = value.get("hits")
        max_age = (
            settings.ismael_research_cache_days if hits else settings.ismael_research_negative_cache_days
        ) * _DAY
        if isinstance(hits, list) and age < max_age:
            ismael_research_cache_total.labels(kind="search", result="hit").inc()
            return hits, True
    ismael_research_cache_total.labels(kind="search", result="miss").inc()
    hits, complete = await _search(query, domains)
    if hits or complete:
        await _cache_put(_SEARCH_NS, key, {"hits": hits})
    return hits, False


async def _cached_page(url: str) -> str:
    cached = await _cache_get(_PAGE_NS, _cache_key(url))
    if cached:
        value, age = cached
        text = value.get("text")
        if isinstance(text, str) and text and age < settings.ismael_research_cache_days * _DAY:
            ismael_research_cache_total.labels(kind="page", result="hit").inc()
            return text
    ismael_research_cache_total.labels(kind="page", result="miss").inc()
    return ""


async def _fetch_direct(
    client: httpx.AsyncClient, hit: dict, domains: tuple[str, ...]
) -> tuple[str, bool]:
    """(page text, refused): ``refused`` when the server answered with an error
    (a bot challenge), which Serper's scrape may still get past."""
    try:
        page = await web_fetch.fetch_page(
            client,
            hit["url"],
            max_bytes=_MAX_PAGE_BYTES,
            max_chars=settings.ismael_research_page_chars,
        )
    except web_fetch.FetchBlocked as exc:
        logger.info("ismael_research_fetch_failed", url=hit["url"], error=str(exc))
        return "", exc.reason == "http_error"
    except Exception as exc:  # noqa: BLE001 — TLS, DNS: fall back to the snippet
        logger.info("ismael_research_fetch_failed", url=hit["url"], error=str(exc) or type(exc).__name__)
        return "", False
    # A redirect off the allowed domains is not Moodle's documentation.
    return (page.text.strip() if on_allowed_domain(page.url, domains) else ""), False


async def _scrape(url: str) -> str:
    try:
        data = await serper_post(SCRAPE_ENDPOINT, {"url": url})
    except SerperAuthError as exc:  # never voids the lookup: the snippet remains
        logger.info("ismael_research_scrape_failed", url=url, error=str(exc))
        return ""
    return ((data or {}).get("text") or "").strip()[: settings.ismael_research_page_chars]


async def _read(hits: list[dict], domains: tuple[str, ...]) -> tuple[list[Source], int]:
    """(sources, scrapes made). Cached pages first; the rest are fetched."""
    texts = list(await asyncio.gather(*(_cached_page(h["url"]) for h in hits)))
    missing = [i for i, text in enumerate(texts) if not text]
    refused: dict[int, bool] = {}
    if missing:
        async with web_fetch.build_client(settings.ismael_research_timeout_seconds) as client:
            fetched = await asyncio.gather(*(_fetch_direct(client, hits[i], domains) for i in missing))
        for i, (text, was_refused) in zip(missing, fetched):
            texts[i], refused[i] = text, was_refused
    scrapes_left = settings.ismael_research_scrape_pages
    scrapes = 0
    sources = []
    for i, hit in enumerate(hits):  # in rank order: the top hit is scraped first
        text = texts[i]
        if not text and refused.get(i) and scrapes_left > 0:
            scrapes_left -= 1
            scrapes += 1
            text = await _scrape(hit["url"])
        if text and i in missing:
            # Page text only: a snippet is not the page, and is not cached as one.
            await _cache_put(_PAGE_NS, _cache_key(hit["url"]), {"url": hit["url"], "text": text})
        text = text or (hit.get("snippet") or "").strip()
        if text:
            sources.append(Source(url=hit["url"], title=hit.get("title") or "", text=text))
    return sources, scrapes


async def look_up(query: str) -> tuple[list[Source], str]:
    """(sources, outcome) for ``query``. Never raises.

    Outcomes: read · cached (read without spending a Serper credit) ·
    disabled · no_results · no_pages · search_failed · timeout · error.
    """
    if not (settings.ismael_research_enabled and settings.serper_api_key and query.strip()):
        return [], "disabled"
    domains = allowed_domains()
    try:
        async with asyncio.timeout(settings.ismael_research_timeout_seconds):
            hits, search_cached = await _cached_search(query, domains)
            if not hits:
                return [], "no_results"
            sources, scrapes = await _read(hits, domains)
    except TimeoutError:
        return [], "timeout"
    except SerperAuthError as exc:  # no credits or a bad key: answer without it
        logger.warning("ismael_research_search_failed", error=str(exc))
        return [], "search_failed"
    except Exception as exc:  # noqa: BLE001 — research is a bonus, never a failure
        logger.warning("ismael_research_error", error=str(exc) or type(exc).__name__)
        return [], "error"
    if not sources:
        return [], "no_pages"
    return sources, ("cached" if search_cached and not scrapes else "read")


def _defang(value: str) -> str:
    """Page text must not be able to close its own fence."""
    return value.replace("<<<", "‹‹‹").replace(">>>", "›››")


def sources_block(sources: list[Source]) -> str:
    """The pages, fenced as data, for the second support call."""
    parts = [
        f"<<<FUENTE {n} · {source.url}\n{_defang(source.title)}\n{_defang(source.text)}\nFUENTE>>>"
        for n, source in enumerate(sources, start=1)
    ]
    return "\n\nFuentes consultadas:\n" + "\n\n".join(parts)


def cited_url(source_url: str, sources: list[Source]) -> str:
    """``source_url`` when it is one of the pages read, else ""."""
    wanted = (source_url or "").strip().rstrip("/")
    return next((s.url for s in sources if s.url.rstrip("/") == wanted), "") if wanted else ""
