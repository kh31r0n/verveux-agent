"""Nodes for the autonomous prospecting graph (aurora).

Non-conversational: triggered by the backend scheduler, not a customer. The
graph discovers private schools/kindergartens in Colombia via web search,
extracts each institution's data with the LLM (fanned out per search result via
LangGraph ``Send``), deduplicates against the CRM, and files the survivors as
``AI_PROSPECTING`` contacts for human review.

All external effects (dedupe, contact creation, run reporting) go through the
NestJS internal endpoints in ``backend_client`` — the agent never touches the
database directly.
"""

from __future__ import annotations

import asyncio
import json
import operator
import re
from typing import Annotated, TypedDict

import httpx
import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from ..config import settings
from ..providers.registry import get_provider, resolve_model
from ..usage import make_usage_record
from . import backend_client
from .dedup import normalize_domain, normalize_name, prospect_external_id
from .utils import resolve_prompt

logger = structlog.get_logger(__name__)

SERPER_ENDPOINT = "https://google.serper.dev/search"

# Rotated across run dates so successive days cover different metro areas
# instead of re-scanning Bogotá every morning.
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

# Query templates. Constrained to school-ish sites to cut noise. Hardcoded (not
# prompt-editable) so a prompt edit can never break the discovery surface.
QUERY_TEMPLATES: list[str] = [
    "colegios privados en {city} Colombia",
    "jardines infantiles privados en {city} Colombia",
]

DEFAULT_EXTRACTION_PROMPT = (
    "Eres un asistente de extracción de datos. A partir del contenido de una "
    "página web de un colegio o jardín infantil privado en Colombia, extrae los "
    "datos de la institución.\n\n"
    "Devuelve EXCLUSIVAMENTE un objeto JSON válido con esta forma exacta:\n"
    "{\n"
    '  "name": "nombre oficial (string, obligatorio)",\n'
    '  "email": "correo o null",\n'
    '  "website": "URL del sitio oficial o null",\n'
    '  "city": "ciudad/municipio o null",\n'
    '  "notes": "una frase corta o null",\n'
    '  "is_school": true | false\n'
    "}\n\n"
    "Reglas: NO inventes datos (usa null si no aparece). is_school es false si la "
    "página no corresponde a un colegio/jardín privado. Responde SOLO con el JSON."
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_MAX_PAGE_CHARS = 6000


class ProspectingState(TypedDict, total=False):
    """Per-run state. JSON-native only (checkpointer serde is strict)."""

    tenant_id: str
    run_id: str
    run_date: str
    queries: list[str]
    search_results: list[dict]
    # Reduced across the parallel extract fan-out.
    candidates: Annotated[list[dict], operator.add]
    deduped_candidates: list[dict]
    created: list[dict]
    turn_usage: Annotated[list[dict], operator.add]
    metrics: dict


# ── plan_searches ───────────────────────────────────────────────────────────


def _plan_queries(run_date: str) -> list[str]:
    """Deterministic query set for the run, rotating cities by run date."""
    # Cheap date-derived offset so consecutive days start at different cities.
    offset = sum(ord(c) for c in (run_date or "")) % len(COLOMBIA_CITIES)
    rotated = COLOMBIA_CITIES[offset:] + COLOMBIA_CITIES[:offset]
    queries: list[str] = []
    for city in rotated:
        for template in QUERY_TEMPLATES:
            queries.append(template.format(city=city))
            if len(queries) >= settings.prospecting_max_searches:
                return queries
    return queries


async def plan_searches_node(state: ProspectingState, config: RunnableConfig) -> dict:
    queries = _plan_queries(state.get("run_date", ""))
    logger.info(
        "prospecting_plan", run_id=state.get("run_id"), queries=len(queries)
    )
    return {"queries": queries, "metrics": {"planned_queries": len(queries)}}


# ── web_search ──────────────────────────────────────────────────────────────


async def _serper_search(query: str) -> list[dict]:
    if not settings.serper_api_key:
        return []
    payload = {
        "q": query,
        "num": settings.prospecting_max_results_per_search,
        "gl": "co",
        "hl": "es",
    }
    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }
    # Exponential backoff on transient failures / rate limits.
    delay = 0.6
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    SERPER_ENDPOINT, headers=headers, json=payload
                )
                if resp.status_code == 429 and attempt < 2:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
                data = resp.json()
            organic = data.get("organic", []) if isinstance(data, dict) else []
            return [
                {
                    "title": o.get("title", ""),
                    "url": o.get("link", ""),
                    "snippet": o.get("snippet", ""),
                }
                for o in organic
                if isinstance(o, dict) and o.get("link")
            ]
        except httpx.HTTPError as exc:
            if attempt < 2:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            logger.warning("serper_search_failed", query=query, error=str(exc))
            return []
    return []


async def web_search_node(state: ProspectingState, config: RunnableConfig) -> dict:
    queries = state.get("queries", [])
    seen_urls: set[str] = set()
    results: list[dict] = []
    for query in queries:
        for hit in await _serper_search(query):
            url = hit["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(hit)

    logger.info(
        "prospecting_search",
        run_id=state.get("run_id"),
        queries=len(queries),
        results=len(results),
    )
    return {
        "search_results": results,
        "metrics": {
            **(state.get("metrics") or {}),
            "search_results": len(results),
        },
    }


def fan_out_to_extract(state: ProspectingState) -> list:
    """Conditional edge: one extract_and_enrich invocation per search result.

    With zero results the fan-out would strand ``dedupe_check`` (its only
    incoming edge is from ``extract_and_enrich``), so route straight there to
    guarantee the run still reports COMPLETED with 0 found.
    """
    results = state.get("search_results", [])
    if not results:
        return ["dedupe_check"]
    tenant_id = state.get("tenant_id", "")
    return [
        Send("extract_and_enrich", {"result": r, "tenant_id": tenant_id})
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
        logger.info("prospecting_fetch_failed", url=url, error=str(exc))
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
    """Extract one institution from one search result. Receives a Send payload
    (``{result, tenant_id}``), not the full graph state."""
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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": page_text},
    ]

    text = ""
    try:
        async for chunk in provider.stream_chat(model=model, messages=messages):
            text += chunk
    except Exception as exc:  # provider/network failure — skip this result
        logger.warning("prospecting_extract_failed", url=url, error=str(exc))
        return {}

    usage = dict(
        make_usage_record(
            node="extract_and_enrich", provider=provider, model=model
        )
    )

    parsed = _parse_extraction(text)
    if not parsed or not parsed.get("is_school"):
        return {"turn_usage": [usage]}

    name = (parsed.get("name") or "").strip()
    external_id = prospect_external_id(name)
    if not external_id:
        return {"turn_usage": [usage]}

    website = parsed.get("website") or url
    candidate = {
        "externalId": external_id,
        "customName": name,
        "normalizedName": normalize_name(name),
        "email": (parsed.get("email") or None),
        "website": website,
        "domain": normalize_domain(website),
        "city": (parsed.get("city") or None),
        "notes": (parsed.get("notes") or None),
        "sourceUrl": url,
    }
    return {"candidates": [candidate], "turn_usage": [usage]}


# ── dedupe_check ────────────────────────────────────────────────────────────


async def dedupe_check_node(state: ProspectingState, config: RunnableConfig) -> dict:
    candidates = state.get("candidates", [])
    tenant_id = state.get("tenant_id", "")

    # 1) Intra-run dedupe by synthetic identity (two pages, same school).
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

    logger.info(
        "prospecting_dedupe",
        run_id=state.get("run_id"),
        candidates=len(candidates),
        unique=len(unique),
        survivors=len(survivors),
    )
    return {
        "deduped_candidates": survivors,
        "metrics": {
            **(state.get("metrics") or {}),
            "found": len(candidates),
            "unique": len(unique),
            "duplicates": len(unique) - len(survivors),
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


# ── report ──────────────────────────────────────────────────────────────────


async def report_node(state: ProspectingState, config: RunnableConfig) -> dict:
    run_id = state.get("run_id", "")
    metrics = dict(state.get("metrics") or {})
    usage: list[dict] = list(state.get("turn_usage") or [])
    metrics["invocations"] = len(usage)

    try:
        await backend_client.report_prospecting_run(
            run_id, "COMPLETED", metrics=metrics, usage=usage
        )
    except Exception as exc:
        # The endpoint's failure path will mark the run FAILED if this never
        # lands; log and move on rather than crash the graph.
        logger.warning("prospecting_report_failed", run_id=run_id, error=str(exc))

    return {"metrics": metrics}
