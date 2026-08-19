"""Nodes for the autonomous prospecting graph (aurora).

Non-conversational: triggered by the backend scheduler, not a customer. The
graph discovers organizations for a tenant-configured **niche** in a configurable
**location** via web search (Serper ``/search`` + ``/places``), extracts each
organization's data with the LLM (fanned out per search result via LangGraph
``Send``), deduplicates against the CRM, and files the survivors as
``AI_PROSPECTING`` contacts for human review.

Self-improvement (three mechanisms):
  1. **Cross-run strategy memory** — `load_memory` reads, and `reflect_memory`
     writes, per-niche `{best_queries, avoid_patterns, notes}` in a LangGraph
     ``BaseStore`` (Postgres in prod), so successive runs of the same niche start
     from what worked before.
  2. **In-run query refinement** — when a run's quality survivors fall short of
     the threshold, a conditional edge routes to `refine_queries` (LLM) and back
     to `web_search`, up to `prospecting_max_iterations` total passes.
  3. **Human feedback** — good/bad verdicts recorded in the CRM are read at run
     start and injected as concrete examples into the extraction prompt.

All external effects (dedupe, contact creation, feedback read, run reporting) go
through the NestJS internal endpoints in ``backend_client`` — the agent never
touches the CRM database directly. The strategy store is the ONE piece of state
the agent owns itself.
"""

from __future__ import annotations

import json
import operator
import re
import unicodedata
from datetime import datetime, timezone
from typing import Annotated, TypedDict

import httpx
import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from ..config import settings
from ..providers.registry import get_provider, resolve_model
from ..services.serper import (
    PLACES_ENDPOINT,
    SERPER_ENDPOINT,
    SerperAuthError,
    _count_serper_call,
    _serper_calls,
    parse_organic,
    parse_places,
    serper_call_count,
    start_serper_accounting,
)
from ..services.serper import serper_post as _serper_post
from ..usage import make_usage_record
from . import backend_client
from .dedup import normalize_domain, normalize_name, prospect_external_id
from .utils import resolve_prompt

logger = structlog.get_logger(__name__)

# Rotated across run dates so successive days cover different metro areas
# instead of re-scanning the first city every morning.
COLOMBIA_CITIES: list[str] = [
    "Bogotá",
    "Medellín",
    "Cali",
    "Barranquilla",
    "Cartagena",
    "Bucaramanga",
    "Pereira",
    "Manizales",
    "Cúcuta",
    "Ibagué",
    "Santa Marta",
    "Villavicencio",
]

# Fallback when the backend sends no location. There is deliberately NO default
# niche: aurora is industry-agnostic, so WHAT to prospect must always come from
# the tenant's configuration — a built-in fallback would silently prospect
# somebody else's industry. `/prospecting/run` rejects a request without one.
DEFAULT_LOCATION: dict = {
    "country": "Colombia",
    "gl": "co",
    "hl": "es",
    "cities": COLOMBIA_CITIES,
}

DEFAULT_EXTRACTION_PROMPT = (
    "Eres un asistente de extracción de datos. A partir del contenido de una "
    "página web, extrae los datos de la organización.\n\n"
    "Devuelve EXCLUSIVAMENTE un objeto JSON válido con esta forma exacta:\n"
    "{\n"
    '  "name": "nombre oficial (string, obligatorio)",\n'
    '  "email": "correo o null",\n'
    '  "website": "URL del sitio oficial o null",\n'
    '  "city": "ciudad/municipio o null",\n'
    '  "notes": "una frase corta o null",\n'
    '  "is_match": true | false\n'
    "}\n\n"
    "Reglas: NO inventes datos (usa null si no aparece). is_match es false si la "
    "página no corresponde al nicho objetivo indicado. Responde SOLO con el JSON."
)

# Runtime-injected geographic context for the extraction prompt.
#
# Deliberately NOT part of the tenant-editable prompt: a tenant that already has
# an ACTIVE PROSPECTING_EXTRACTION row would never pick up an edit to the default,
# so the region rule would silently not apply to exactly the tenants that have
# customized their prompt. Injected at call time like `niche_label`, it always
# applies. Same reason it carries no `{placeholder}` — that would couple it to
# the backend's prompt-variable validation.
REGION_CONTEXT_TEMPLATE = (
    "REGIÓN OBJETIVO: {cities} ({country}).\n"
    "El campo \"city\" es OBLIGATORIO en tu respuesta y es la señal más "
    "importante después del nombre. La página puede ser un directorio nacional "
    "que lista organizaciones de MUCHAS ciudades: extrae la ciudad de LA "
    "organización que estás describiendo (búscala en la dirección, el pie de "
    "página, el dominio o la URL), NO la ciudad del directorio ni una de la "
    "lista de arriba. Si de verdad no puedes determinarla, usa null — nunca "
    "adivines ni copies una ciudad objetivo por defecto."
)

# Internal reasoning prompt (NOT tenant-editable) for the refinement loop.
REFINE_QUERIES_PROMPT = (
    "Eres un estratega de búsqueda para prospección B2B. El nicho objetivo es: "
    "{niche_label}, en {country} (ciudades: {cities}).\n"
    "TÉRMINOS DEL NICHO (definidos por el cliente, son la definición "
    "autoritativa del rubro): {terms}.\n"
    "Dadas las búsquedas ya intentadas y cuántos prospectos de calidad produjo "
    "cada una, propón hasta {max_new} NUEVAS consultas de búsqueda de Google que "
    "probablemente encuentren MÁS organizaciones de este nicho con datos de "
    "contacto. Varía barrios/zonas y modificadores (por ejemplo 'directorio', "
    "'listado', 'asociación', 'contacto'). NO repitas las búsquedas ya "
    "intentadas.\n"
    "RESTRICCIÓN DE RUBRO OBLIGATORIA: cada consulta debe contener al menos uno "
    "de los TÉRMINOS DEL NICHO. NO uses sinónimos, traducciones ni términos "
    "relacionados que cambien el rubro. Si un término es ambiguo, interprétalo "
    "SIEMPRE según el nicho indicado arriba y nunca según otro sector.\n"
    "RESTRICCIÓN GEOGRÁFICA OBLIGATORIA: cada consulta debe nombrar "
    "explícitamente una de las ciudades objetivo listadas arriba. Nunca "
    "propongas una consulta nacional o de otra ciudad — el objetivo es "
    "profundizar en la región, no ampliarla.\n"
    'Responde EXCLUSIVAMENTE con un arreglo JSON de strings, por ejemplo: '
    '["consulta 1", "consulta 2"].'
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_MAX_PAGE_CHARS = 6000
_MAX_BEST_QUERIES = 15
_MAX_AVOID_PATTERNS = 30


class ProspectingState(TypedDict, total=False):
    """Per-run state. JSON-native only (checkpointer serde is strict)."""

    tenant_id: str
    run_id: str
    run_date: str
    # Tenant-configurable targeting. `niche` is required (validated at the
    # endpoint); `location` falls back to DEFAULT_LOCATION.
    niche: dict
    location: dict
    # Self-improvement inputs, loaded once by load_memory.
    strategy_memory: dict
    feedback_block: str
    # Refinement-loop bookkeeping.
    iteration: int
    queries: list[str]
    # Reduced across loop iterations (each web_search returns only its new items).
    searched_queries: Annotated[list[str], operator.add]
    seen_urls: Annotated[list[str], operator.add]
    search_results: list[dict]
    # Reduced across the parallel extract fan-out AND across loop iterations.
    candidates: Annotated[list[dict], operator.add]
    deduped_candidates: list[dict]
    quality_count: int
    enough_quality: bool
    created: list[dict]
    turn_usage: Annotated[list[dict], operator.add]
    metrics: dict


# ── small helpers ────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def niche_key_of(niche: dict | None) -> str:
    """Niche key, or "" when unconfigured.

    "" is a deliberate dead end, not a placeholder: the key namespaces the
    cross-run strategy store, so a stand-in would pool unrelated industries into
    one learned query set. `_load_strategy`/`_save_strategy` already no-op on a
    falsy key.
    """
    key = (niche or {}).get("key")
    return key.strip() if isinstance(key, str) else ""


def niche_label_of(niche: dict | None) -> str:
    """Human-readable niche for the prompts. Falls back to the key, then "" —
    never to a built-in industry."""
    label = (niche or {}).get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    return niche_key_of(niche)


def _fold(text: str) -> str:
    """Accent- and case-insensitive form for place-name comparison.

    The CRM already holds both `Chia`/`Chía` and `Cajica`/`Cajicá` for the same
    municipality (whichever spelling the source page used), so a raw `==` against
    the configured city list mislabels roughly half the matches as out-of-region.
    """
    stripped = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    return _WS_RE.sub(" ", stripped).strip().casefold()


def city_in_region(city: str | None, location: dict | None) -> bool | None:
    """Is ``city`` one of the tenant's configured cities?

    Returns ``None`` — not ``False`` — when the answer is unknowable (no city
    extracted, or no cities configured). The caller must treat that as "don't
    tag": region is advisory here, and tagging on missing data would flag the
    ~28% of prospects whose page never states a city, which is a data-quality
    problem, not a targeting one.

    Substring matching is intentional and one-directional: a page saying
    "Zipaquirá, Cundinamarca" or "Chía (Sabana Centro)" is the configured city,
    while a bare "Bogotá" matches no configured entry and is correctly flagged.
    """
    configured = [c for c in ((location or {}).get("cities") or []) if c]
    if not city or not city.strip() or not configured:
        return None
    folded = _fold(city)
    if not folded:
        return None
    return any(_fold(c) in folded for c in configured)


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")


def clean_website(raw: str | None) -> str | None:
    """Coerce an LLM-extracted website into something the backend accepts.

    `ProspectContactInput.website` is `@IsUrl({require_protocol: true})`, and a
    single failing field 400s the WHOLE contact — a run once lost 26 of 47
    prospects to schemeless values like "acme.com.co". A missing scheme is a
    formatting artifact, not a data problem, so add it; anything still unusable
    is dropped so the prospect is filed without a website rather than lost.
    """
    value = (raw or "").strip().rstrip(".,;")
    if not value or " " in value:
        return None
    if not value.startswith(("http://", "https://")):
        if "://" in value:  # some other scheme — not fetchable, drop it
            return None
        value = f"https://{value}"
    host = value.split("://", 1)[1].split("/", 1)[0].split("@")[-1]
    if "." not in host.strip("."):
        return None
    return value


def clean_email(raw: str | None) -> str | None:
    """Drop an email the backend's `@IsEmail` would reject (same 400 trap).

    Extractions routinely yield partials like "info@acme" or a mailto: prefix.
    """
    value = (raw or "").strip().rstrip(".,;")
    if value.lower().startswith("mailto:"):
        value = value[7:].strip()
    return value if _EMAIL_RE.match(value) else None


def _stem(word: str) -> str:
    """Crude Spanish plural stem, enough to match a term against a query word.

    The tenant writes "Colegio" but a good query says "colegios privados"; the
    reverse also happens. Stripping a trailing "es"/"s" from both sides makes the
    two directions symmetric without pulling in a stemming dependency.
    """
    if len(word) > 5 and word.endswith("es"):
        return word[:-2]
    if len(word) > 4 and word.endswith("s"):
        return word[:-1]
    return word


def query_matches_niche(query: str, niche: dict | None) -> bool:
    """Does ``query`` still target the tenant's own industry?

    The LLM refiner is asked for query *variations* and will happily reinterpret
    an ambiguous word: with the niche "Colegios" it proposed fitness-gym queries
    and crawled spinningcentergym.com, because in Colombia a "Gimnasio Campestre"
    is a school. Prompt wording alone can't be relied on for that — this is the
    deterministic gate, anchored on the tenant's own `search_terms`, which are
    the authoritative definition of the niche.

    Returns True when no terms are configured: this is a drift filter, not an
    authorization check, and the scheduler already refuses to dispatch a run
    without terms.
    """
    terms = [t for t in ((niche or {}).get("search_terms") or []) if t]
    if not terms:
        return True
    folded_query = _fold(query)
    query_stems = {_stem(w) for w in folded_query.split()}
    for term in terms:
        # A multi-word term must match every one of its words, so "jardines
        # infantiles" doesn't green-light a query about botanical gardens.
        words = [_stem(w) for w in _fold(term).split() if w]
        if words and all(
            w in query_stems or w in folded_query for w in words
        ):
            return True
    return False


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


# ── strategy store (best-effort; a missing store degrades to stateless) ──────


async def _load_strategy(tenant_id: str, niche_key: str) -> dict:
    # Imported lazily to avoid a circular import (registry → prospecting_graph →
    # prospecting_nodes).
    from ..graphs.registry import get_store_or_none

    store = get_store_or_none()
    if store is None or not tenant_id or not niche_key:
        return {}
    try:
        item = await store.aget((tenant_id, niche_key, "strategy"), "latest")
        if item is not None and getattr(item, "value", None):
            return dict(item.value)
    except Exception as exc:  # noqa: BLE001 — memory is optional
        logger.warning("prospecting_strategy_load_failed", error=str(exc))
    return {}


async def _save_strategy(tenant_id: str, niche_key: str, value: dict) -> None:
    from ..graphs.registry import get_store_or_none

    store = get_store_or_none()
    if store is None or not tenant_id or not niche_key:
        return
    try:
        await store.aput((tenant_id, niche_key, "strategy"), "latest", value)
    except Exception as exc:  # noqa: BLE001 — memory is optional
        logger.warning("prospecting_strategy_save_failed", error=str(exc))


def _render_feedback_block(feedback: list[dict], cap: int) -> str:
    """Turn CRM good/bad verdicts into a compact prompt block."""
    good = [f for f in feedback if str(f.get("verdict", "")).upper() == "GOOD"][:cap]
    bad = [f for f in feedback if str(f.get("verdict", "")).upper() == "BAD"][:cap]
    if not good and not bad:
        return ""
    lines = ["EJEMPLOS DE REVISIÓN HUMANA (aprende de estos):"]
    for f in good:
        note = f" — {f.get('note')}" if f.get("note") else ""
        lines.append(f"- BUENO: {f.get('customName') or '?'}{note}")
    for f in bad:
        note = f" — {f.get('note')}" if f.get("note") else ""
        lines.append(f"- MALO: {f.get('customName') or '?'}{note}")
    lines.append(
        "Prioriza organizaciones parecidas a los BUENOS y descarta (is_match=false) "
        "las parecidas a los MALOS."
    )
    return "\n".join(lines)


# ── load_memory ──────────────────────────────────────────────────────────────


async def load_memory_node(state: ProspectingState, config: RunnableConfig) -> dict:
    """Seed the run with cross-run strategy memory + human feedback. Fail-open."""
    tenant_id = state.get("tenant_id", "")
    niche = state.get("niche") or {}
    niche_key = niche_key_of(niche)

    strategy = await _load_strategy(tenant_id, niche_key)

    feedback_block = ""
    try:
        feedback = await backend_client.get_prospect_feedback(tenant_id, niche_key)
        feedback_block = _render_feedback_block(
            feedback, settings.prospecting_feedback_examples
        )
    except Exception as exc:  # noqa: BLE001 — feedback is optional
        logger.warning("prospecting_feedback_load_failed", error=str(exc))

    logger.info(
        "prospecting_load_memory",
        run_id=state.get("run_id"),
        niche=niche_key,
        best_queries=len(strategy.get("best_queries") or []),
        has_feedback=bool(feedback_block),
    )
    return {
        "iteration": 0,
        "strategy_memory": strategy,
        "feedback_block": feedback_block,
    }


# ── plan_searches ───────────────────────────────────────────────────────────


def _plan_queries(
    run_date: str,
    niche: dict | None = None,
    location: dict | None = None,
    strategy: dict | None = None,
) -> list[str]:
    """Query set for the run: learned best queries first, then a deterministic
    niche×city floor rotated by run date. ``avoid_patterns`` filters ONLY the
    learned seeds, never the deterministic floor — so memory can augment but
    never empty the discovery surface."""
    niche = niche or {}
    location = location or DEFAULT_LOCATION
    strategy = strategy or {}

    terms = [t for t in (niche.get("search_terms") or []) if t]
    cities = location.get("cities") or DEFAULT_LOCATION["cities"]
    country = location.get("country") or DEFAULT_LOCATION["country"]
    budget = settings.prospecting_max_searches

    # Date-derived offset so consecutive days start at different cities.
    offset = sum(ord(c) for c in (run_date or "")) % max(len(cities), 1)
    rotated = cities[offset:] + cities[:offset]

    avoid = {a.lower() for a in (strategy.get("avoid_patterns") or [])}
    queries: list[str] = []

    # 1) Learned best queries (avoid- AND niche-filtered), highest signal first.
    # The niche filter also retro-cleans memory written before the anchor
    # existed, and covers a tenant who edits `search_terms` without resetting the
    # store — the seeds are then simply ignored and the floor takes over.
    for q in strategy.get("best_queries") or []:
        if (
            q
            and q.lower() not in avoid
            and q not in queries
            and query_matches_niche(q, niche)
        ):
            queries.append(q)
            if len(queries) >= budget:
                return queries

    # 2) Deterministic niche×city floor (never avoid-filtered).
    for city in rotated:
        for term in terms:
            q = f"{term} en {city} {country}".strip()
            if q in queries:
                continue
            queries.append(q)
            if len(queries) >= budget:
                return queries
    return queries


async def plan_searches_node(state: ProspectingState, config: RunnableConfig) -> dict:
    niche = state.get("niche") or {}
    location = state.get("location") or DEFAULT_LOCATION
    strategy = state.get("strategy_memory") or {}
    queries = _plan_queries(state.get("run_date", ""), niche, location, strategy)
    logger.info(
        "prospecting_plan",
        run_id=state.get("run_id"),
        niche=niche.get("key"),
        queries=len(queries),
        seeded=len(strategy.get("best_queries") or []),
    )
    return {"queries": queries, "metrics": {"planned_queries": len(queries)}}


# ── web_search (Serper /search + /places) ────────────────────────────────────
#
# The HTTP call, its retry/auth policy, the run-scoped credit counter and the
# response parsers live in ``services.serper`` — sherlock's website discovery
# bills against the same account. Query construction and the geo fallback below
# stay here: they are prospecting's own policy.


def serper_location_for(query: str, location: dict | None) -> str | None:
    """Serper's `location` param for a query, or None to leave it national.

    `gl`/`hl` only pin the country, so a query like "<término> en Chía Colombia"
    still competes against national directory pages — which is how prospects from
    Manizales and Cartagena ended up in a Sabana-Centro run. `location` geo-biases
    the SERP itself.

    The city is recovered from the query text rather than threaded through state:
    queries are built as "{term} en {city} {country}", and the learned
    `best_queries` in the strategy store are plain strings of that same shape, so
    a structural change would have to migrate stored memory for no extra benefit.
    Longest match first, so "Santa Marta" wins over a configured "Marta".
    """
    if not settings.prospecting_geo_targeting_enabled:
        return None
    location = location or DEFAULT_LOCATION
    country = location.get("country") or DEFAULT_LOCATION["country"]
    folded_query = _fold(query)
    cities = sorted(
        (c for c in (location.get("cities") or []) if c), key=len, reverse=True
    )
    for city in cities:
        if _fold(city) in folded_query:
            return f"{city}, {country}"
    return None


async def _serper_search(query: str, location: dict | None = None) -> list[dict]:
    location = location or DEFAULT_LOCATION
    payload = {
        "q": query,
        "num": settings.prospecting_max_results_per_search,
        "gl": location.get("gl") or "co",
        "hl": location.get("hl") or "es",
    }
    geo = serper_location_for(query, location)
    if geo:
        payload["location"] = geo
    data = await _serper_post(SERPER_ENDPOINT, payload)
    # Serper rejects (or empties) an unrecognized canonical location. Retry once
    # nationally so a small municipality Google doesn't know still yields results
    # — the extraction-side region check is what actually enforces targeting.
    if geo and not (data or {}).get("organic"):
        logger.info("prospecting_geo_fallback", query=query, location=geo)
        payload.pop("location", None)
        data = await _serper_post(SERPER_ENDPOINT, payload)
    return parse_organic(data)


async def _serper_places(query: str, location: dict | None = None) -> list[dict]:
    """Local-business results (name, address, phone, website). Gated by
    ``prospecting_places_enabled``; transient failures return [] so /search still
    stands. A rejected credential (:class:`SerperAuthError`) is not transient and
    propagates — /search cannot stand either when the key itself is bad."""
    if not settings.prospecting_places_enabled:
        return []
    location = location or DEFAULT_LOCATION
    payload = {
        "q": query,
        "gl": location.get("gl") or "co",
        "hl": location.get("hl") or "es",
    }
    geo = serper_location_for(query, location)
    if geo:
        payload["location"] = geo
    data = await _serper_post(PLACES_ENDPOINT, payload)
    if geo and not (data or {}).get("places"):
        payload.pop("location", None)
        data = await _serper_post(PLACES_ENDPOINT, payload)
    return parse_places(data)


async def web_search_node(state: ProspectingState, config: RunnableConfig) -> dict:
    queries = state.get("queries", [])
    location = state.get("location") or DEFAULT_LOCATION
    prior_seen = set(state.get("seen_urls") or [])  # accumulated across iterations
    seen: set[str] = set()
    results: list[dict] = []
    new_urls: list[str] = []

    for query in queries:
        combined = list(await _serper_search(query, location)) + list(
            await _serper_places(query, location)
        )
        for hit in combined:
            url = hit.get("url", "")
            # Places rows may have no website; dedupe those by title instead.
            key = url or ("title:" + (hit.get("title", "") or "").strip().lower())
            if not key or key in seen or (url and url in prior_seen):
                continue
            seen.add(key)
            results.append({**hit, "query": query})
            if url:
                new_urls.append(url)

    logger.info(
        "prospecting_search",
        run_id=state.get("run_id"),
        iteration=state.get("iteration"),
        queries=len(queries),
        results=len(results),
    )
    return {
        "search_results": results,
        "searched_queries": list(queries),
        "seen_urls": new_urls,
        "metrics": {
            **(state.get("metrics") or {}),
            "search_results": len(results),
        },
    }


def fan_out_to_extract(state: ProspectingState) -> list:
    """Conditional edge: one extract_and_enrich invocation per search result.

    With zero results the fan-out would strand ``dedupe_check`` (its only static
    incoming edge is from ``extract_and_enrich``), so route straight there to
    guarantee the run advances to evaluation/reporting.
    """
    results = state.get("search_results", [])
    if not results:
        return ["dedupe_check"]
    tenant_id = state.get("tenant_id", "")
    niche = state.get("niche") or {}
    niche_label = niche_label_of(niche)
    location = state.get("location") or DEFAULT_LOCATION
    feedback_block = state.get("feedback_block", "")
    return [
        Send(
            "extract_and_enrich",
            {
                "result": r,
                "tenant_id": tenant_id,
                "niche_label": niche_label,
                "location": location,
                "feedback_block": feedback_block,
                "query": r.get("query", ""),
            },
        )
        for r in results
    ]


# ── extract_and_enrich (Send target, runs in parallel) ──────────────────────


def _strip_html(html: str) -> str:
    text = _TAG_RE.sub(" ", html)
    return _WS_RE.sub(" ", text).strip()[:_MAX_PAGE_CHARS]


async def _fetch_page_text(url: str) -> str:
    try:
        async with httpx.AsyncClient(
            timeout=settings.prospecting_fetch_timeout_seconds,
            follow_redirects=True,
        ) as client:
            resp = await client.get(url, headers={"User-Agent": "VerveuxProspector/1.0"})
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "html" not in ctype and "text" not in ctype:
                return ""
            return _strip_html(resp.text)
    except httpx.HTTPError as exc:
        # The exception TYPE is logged alongside the message because httpx raises
        # timeouts with an empty `str(exc)` — on 2026-08-18 every failed fetch in
        # a live run logged `error: ""`, which cannot tell a read timeout from a
        # DNS failure from a 404. `error_type` always says something.
        logger.info(
            "prospecting_fetch_failed",
            url=url,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return ""


def _parse_extraction(raw: str) -> dict | None:
    text = raw.strip()
    # Strip ```json fences if the model added them.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


async def extract_and_enrich_node(state: dict, config: RunnableConfig) -> dict:
    """Extract one organization from one search result. Receives a Send payload
    (``{result, tenant_id, niche_label, feedback_block, query}``), not the full
    graph state."""
    result: dict = state.get("result") or {}
    url = result.get("url", "")
    page_text = await _fetch_page_text(url) if url else ""
    if not page_text:
        # Fall back to the SERP snippet so a fetch failure still yields a name.
        page_text = f"{result.get('title', '')}\n{result.get('snippet', '')}".strip()
    if not page_text:
        return {}

    provider = get_provider(config)
    model = resolve_model(config)
    system_prompt = resolve_prompt(
        config, "PROSPECTING_EXTRACTION", DEFAULT_EXTRACTION_PROMPT
    )
    # Inject niche context + human feedback at runtime (no prompt placeholder →
    # avoids the backend prompt-variable-validation coupling).
    niche_label = state.get("niche_label") or ""
    location = state.get("location") or DEFAULT_LOCATION
    system_prompt = f"Nicho objetivo: {niche_label}.\n{system_prompt}"
    cities = [c for c in (location.get("cities") or []) if c]
    if cities:
        system_prompt = (
            f"{system_prompt}\n\n"
            + REGION_CONTEXT_TEMPLATE.format(
                cities=", ".join(cities),
                country=location.get("country") or DEFAULT_LOCATION["country"],
            )
        )
    feedback_block = state.get("feedback_block") or ""
    if feedback_block:
        system_prompt = f"{system_prompt}\n\n{feedback_block}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": page_text},
    ]

    text = ""
    try:
        async for chunk in provider.stream_chat(model=model, messages=messages):
            text += chunk
    except Exception as exc:  # provider/network failure — skip this result
        # `error_type` for the same reason as the fetch above: a bare timeout
        # stringifies to "" and would otherwise be unattributable.
        logger.warning(
            "prospecting_extract_failed",
            url=url,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return {}

    usage = dict(
        make_usage_record(
            node="extract_and_enrich", provider=provider, model=model
        )
    )

    parsed = _parse_extraction(text)
    if not parsed:
        return {"turn_usage": [usage]}
    if not parsed.get("is_match"):
        return {"turn_usage": [usage]}

    name = (parsed.get("name") or "").strip()
    external_id = prospect_external_id(name)
    if not external_id:
        return {"turn_usage": [usage]}

    # `url` is a Serper `link` and always carries a scheme, so it is the safe
    # fallback when the extracted website is unusable.
    website = clean_website(parsed.get("website")) or clean_website(url)
    email = clean_email(parsed.get("email"))
    notes = parsed.get("notes") or None
    phone = result.get("phone") or ""
    if phone:
        notes = f"{notes} · Tel: {phone}" if notes else f"Tel: {phone}"

    city = (parsed.get("city") or None)
    # None (unknown city / no cities configured) must NOT become True: the tag
    # means "we placed it elsewhere", not "we couldn't place it".
    in_region = city_in_region(city, location)
    out_of_region = in_region is False
    if out_of_region:
        logger.info(
            "prospecting_out_of_region",
            url=url,
            city=city,
            cities=cities,
        )

    candidate = {
        "externalId": external_id,
        "customName": name,
        "normalizedName": normalize_name(name),
        "email": email,
        "website": website,
        "domain": normalize_domain(website),
        "city": city,
        "notes": notes,
        "sourceUrl": url,
        "is_match": True,
        "outOfRegion": out_of_region,
        "query": state.get("query", ""),
    }
    return {"candidates": [candidate], "turn_usage": [usage]}


# ── dedupe_check ────────────────────────────────────────────────────────────


async def dedupe_check_node(state: ProspectingState, config: RunnableConfig) -> dict:
    candidates = state.get("candidates", [])
    tenant_id = state.get("tenant_id", "")

    # 1) Intra-run dedupe by synthetic identity (two pages, same organization).
    by_id: dict[str, dict] = {}
    for c in candidates:
        by_id.setdefault(c["externalId"], c)
    unique = list(by_id.values())

    # 2) Cross-run dedupe against the CRM (bulk backend call).
    survivors: list[dict] = unique
    if unique and tenant_id:
        try:
            dedup_map = await backend_client.check_prospect_duplicates(
                tenant_id,
                [
                    {
                        "externalId": c["externalId"],
                        "normalizedName": c.get("normalizedName") or "",
                        "domain": c.get("domain") or "",
                        "email": c.get("email") or "",
                    }
                    for c in unique
                ],
            )
            survivors = [
                c
                for c in unique
                if not dedup_map.get(c["externalId"], {}).get("exists")
            ]
        except Exception as exc:
            # On dedupe failure, prefer NOT creating (the backend create is
            # idempotent on externalId, but skipping avoids email/domain dupes).
            logger.warning("prospecting_dedupe_failed", error=str(exc))
            survivors = []

    out_of_region = len([c for c in survivors if c.get("outOfRegion")])
    unknown_city = len([c for c in survivors if not c.get("city")])
    logger.info(
        "prospecting_dedupe",
        run_id=state.get("run_id"),
        candidates=len(candidates),
        unique=len(unique),
        survivors=len(survivors),
        out_of_region=out_of_region,
    )
    return {
        "deduped_candidates": survivors,
        "metrics": {
            **(state.get("metrics") or {}),
            "found": len(candidates),
            "unique": len(unique),
            "duplicates": len(unique) - len(survivors),
            # Targeting health: how much of the run drifted outside the tenant's
            # cities, and how much couldn't be placed at all.
            "out_of_region": out_of_region,
            "unknown_city": unknown_city,
        },
    }


# ── evaluate_quality (loop decision source) ─────────────────────────────────


def _is_quality(candidate: dict) -> bool:
    """A survivor counts toward the threshold if it matches the niche, has a
    contact channel (email or resolvable website), AND was not placed outside the
    tenant's cities. Survivors are already non-CRM-duplicates by construction.

    Excluding out-of-region prospects here is what makes the drift self-correcting
    rather than merely visible: a run that fills its quota with out-of-region hits
    no longer looks "good enough", so `evaluate_quality` triggers the refinement
    loop, and `reflect_memory` demotes the query that produced them into
    `avoid_patterns` instead of promoting it into `best_queries`.
    """
    if candidate.get("outOfRegion"):
        return False
    return bool(candidate.get("email") or candidate.get("domain"))


async def evaluate_quality_node(
    state: ProspectingState, config: RunnableConfig
) -> dict:
    survivors = state.get("deduped_candidates", [])
    quality = len([c for c in survivors if _is_quality(c)])
    enough = quality >= settings.prospecting_min_quality_prospects
    logger.info(
        "prospecting_evaluate",
        run_id=state.get("run_id"),
        iteration=state.get("iteration"),
        quality=quality,
        threshold=settings.prospecting_min_quality_prospects,
        enough=enough,
    )
    return {
        "quality_count": quality,
        "enough_quality": enough,
        "metrics": {**(state.get("metrics") or {}), "quality_count": quality},
    }


def route_from_evaluate(state: ProspectingState) -> str:
    """Conditional edge: stop when we have enough quality prospects OR we've hit
    the iteration cap; otherwise refine the queries and search again."""
    if state.get("enough_quality"):
        return "create_contacts"
    if int(state.get("iteration", 0)) + 1 >= settings.prospecting_max_iterations:
        return "create_contacts"
    return "refine_queries"


# ── refine_queries (LLM; loops back to web_search) ──────────────────────────


def _parse_query_list(raw: str) -> list[str]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if isinstance(x, str) and str(x).strip()]
    return []


async def refine_queries_node(
    state: ProspectingState, config: RunnableConfig
) -> dict:
    iteration = int(state.get("iteration", 0)) + 1
    niche = state.get("niche") or {}
    location = state.get("location") or DEFAULT_LOCATION
    strategy = state.get("strategy_memory") or {}
    searched = set(state.get("searched_queries") or [])
    avoid = {a.lower() for a in (strategy.get("avoid_patterns") or [])}

    # Per-query quality yield so the model can see what worked.
    yield_by_q: dict[str, int] = {}
    for c in state.get("candidates", []):
        q = c.get("query") or ""
        yield_by_q[q] = yield_by_q.get(q, 0) + (1 if _is_quality(c) else 0)

    provider = get_provider(config)
    model = resolve_model(config)
    system_prompt = REFINE_QUERIES_PROMPT.format(
        niche_label=niche_label_of(niche),
        terms=", ".join(t for t in (niche.get("search_terms") or []) if t),
        country=location.get("country") or DEFAULT_LOCATION["country"],
        cities=", ".join((location.get("cities") or [])[:10]),
        max_new=settings.prospecting_max_searches,
    )
    tried = (
        "\n".join(f"- {q}: {n}" for q, n in yield_by_q.items()) or "ninguna todavía"
    )
    user = f"Búsquedas ya intentadas y prospectos de calidad por búsqueda:\n{tried}"

    text = ""
    try:
        async for chunk in provider.stream_chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
        ):
            text += chunk
    except Exception as exc:  # provider failure — advance iteration so the loop
        # guard can terminate instead of retrying the same empty query set.
        logger.warning("prospecting_refine_failed", error=str(exc))
        return {"iteration": iteration, "queries": [], "search_results": []}

    usage = dict(
        make_usage_record(node="refine_queries", provider=provider, model=model)
    )
    proposed = _parse_query_list(text)
    on_niche = [q for q in proposed if query_matches_niche(q, niche)]
    off_niche = len(proposed) - len(on_niche)
    kept = [
        q
        for q in on_niche
        if q and q.lower() not in avoid and q not in searched
    ][: settings.prospecting_max_searches]
    logger.info(
        "prospecting_refine",
        run_id=state.get("run_id"),
        iteration=iteration,
        proposed=len(proposed),
        off_niche=off_niche,
        kept=len(kept),
    )
    # An empty `kept` is a safe outcome, not a failure: web_search finds nothing,
    # evaluate_quality routes on the unchanged survivor set, and the iteration
    # cap terminates the loop. Searching off-niche would be the costly branch.
    return {
        "queries": kept,
        "search_results": [],
        "iteration": iteration,
        "turn_usage": [usage],
        "metrics": {
            **(state.get("metrics") or {}),
            "off_niche_queries": int(
                (state.get("metrics") or {}).get("off_niche_queries") or 0
            )
            + off_niche,
        },
    }


# ── create_contacts ─────────────────────────────────────────────────────────


async def create_contacts_node(state: ProspectingState, config: RunnableConfig) -> dict:
    survivors = state.get("deduped_candidates", [])
    tenant_id = state.get("tenant_id", "")
    run_id = state.get("run_id", "")
    created: list[dict] = []
    errors = 0

    for c in survivors:
        contact = {
            "externalId": c["externalId"],
            "customName": c.get("customName"),
            "sourceUrl": c.get("sourceUrl"),
            "email": c.get("email"),
            "website": c.get("website"),
            "city": c.get("city"),
            "notes": c.get("notes"),
            "outOfRegion": bool(c.get("outOfRegion")),
            "runId": run_id,
        }
        try:
            resp = await backend_client.create_prospect_contact(tenant_id, contact)
            if resp.get("ok"):
                created.append(
                    {
                        "contactId": resp.get("contactId"),
                        "externalId": c["externalId"],
                        "deduped": bool(resp.get("deduped")),
                    }
                )
        except Exception as exc:
            errors += 1
            logger.warning(
                "prospecting_create_failed",
                external_id=c["externalId"],
                error=str(exc),
            )

    logger.info(
        "prospecting_created",
        run_id=run_id,
        created=len(created),
        errors=errors,
    )
    return {
        "created": created,
        "metrics": {
            **(state.get("metrics") or {}),
            "created": len([c for c in created if not c["deduped"]]),
            "create_errors": errors,
        },
    }


# ── reflect_memory (update cross-run strategy) ──────────────────────────────


async def reflect_memory_node(
    state: ProspectingState, config: RunnableConfig
) -> dict:
    """Attribute per-query quality yield and fold it into the niche's strategy
    memory. Deterministic; best-effort persistence."""
    tenant_id = state.get("tenant_id", "")
    niche = state.get("niche") or {}
    niche_key = niche_key_of(niche)
    candidates = state.get("candidates", [])
    searched = state.get("searched_queries") or []

    yield_by_q: dict[str, int] = {}
    for c in candidates:
        q = c.get("query") or ""
        if not q:
            continue
        yield_by_q[q] = yield_by_q.get(q, 0) + (1 if _is_quality(c) else 0)

    prev = state.get("strategy_memory") or {}
    prev_best = list(prev.get("best_queries") or [])
    prev_avoid = list(prev.get("avoid_patterns") or [])

    good_now = [
        q
        for q, n in sorted(yield_by_q.items(), key=lambda kv: kv[1], reverse=True)
        if q and n > 0
    ]
    zero_now = [q for q in searched if q and yield_by_q.get(q, 0) == 0]

    # `best_queries` is capped at _MAX_BEST_QUERIES and seeded ahead of the
    # deterministic floor, so an off-niche query that once yielded something
    # would occupy a scarce slot indefinitely and crowd out real ones. Filter at
    # promotion, not just at use.
    best = [
        q
        for q in _dedup_keep_order(good_now + prev_best)
        if query_matches_niche(q, niche)
    ][:_MAX_BEST_QUERIES]
    best_set = set(best)
    avoid = _dedup_keep_order(
        [a for a in (prev_avoid + zero_now) if a not in best_set]
    )[:_MAX_AVOID_PATTERNS]

    runs_observed = int(prev.get("runs_observed") or 0) + 1
    total_quality = sum(yield_by_q.values())
    notes = (
        f"Última corrida: {total_quality} prospecto(s) de calidad de "
        f"{len(searched)} búsqueda(s). Mejores términos: "
        f"{', '.join(best[:3]) or 'n/d'}."
    )
    value = {
        "best_queries": best,
        "avoid_patterns": avoid,
        "notes": notes,
        "runs_observed": runs_observed,
        "updated_at": _now_iso(),
    }
    await _save_strategy(tenant_id, niche_key, value)

    logger.info(
        "prospecting_reflect",
        run_id=state.get("run_id"),
        niche=niche_key,
        best=len(best),
        avoid=len(avoid),
    )
    return {
        "strategy_memory": value,
        "metrics": {
            **(state.get("metrics") or {}),
            "best_query_count": len(best),
        },
    }


# ── report ──────────────────────────────────────────────────────────────────


async def report_node(state: ProspectingState, config: RunnableConfig) -> dict:
    run_id = state.get("run_id", "")
    metrics = dict(state.get("metrics") or {})
    usage: list[dict] = list(state.get("turn_usage") or [])
    metrics["invocations"] = len(usage)
    # Billable Serper requests (retries included). The backend prices these from
    # its own snapshotted GlobalPlatformSettings.serperSearchPrice — the agent
    # only reports the count, never a cost.
    metrics["serper_calls"] = serper_call_count()
    metrics["iterations"] = int(state.get("iteration", 0)) + 1
    metrics["niche"] = niche_key_of(state.get("niche"))
    metrics["location"] = (state.get("location") or DEFAULT_LOCATION).get("country")

    try:
        await backend_client.report_prospecting_run(
            run_id, "COMPLETED", metrics=metrics, usage=usage
        )
    except Exception as exc:
        # The endpoint's failure path will mark the run FAILED if this never
        # lands; log and move on rather than crash the graph.
        logger.warning("prospecting_report_failed", run_id=run_id, error=str(exc))

    return {"metrics": metrics}
