"""Web Search (Serper) and Google Places connector with cache, circuit breaker, and budget integration."""

from __future__ import annotations

from typing import Any

from ..budget import CalebBudgetTracker
from .base import (
    CalebCache,
    CircuitBreaker,
    ConnectorResult,
    HttpTransportFn,
    RateLimiter,
    SourceCitation,
    is_waf_or_forbidden,
)

JUNK_DOMAINS: frozenset[str] = frozenset(
    {
        "facebook.com",
        "instagram.com",
        "linkedin.com",
        "tiktok.com",
        "youtube.com",
        "wikipedia.org",
        "mercadolibre.com.co",
        "computrabajo.com",
        "elempleo.com",
    }
)


class WebAndPlacesConnector:
    def __init__(
        self,
        transport: HttpTransportFn,
        cache: CalebCache | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        rate_limiter: RateLimiter | None = None,
        budget_tracker: CalebBudgetTracker | None = None,
        cost_per_query_usd: float = 0.002,
    ) -> None:
        self.source_key = "SERPER_GOOGLE"
        self.transport = transport
        self.cache = cache or CalebCache()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.budget_tracker = budget_tracker
        self.cost_per_query_usd = cost_per_query_usd

    async def fetch(self, params: dict[str, Any]) -> ConnectorResult:
        query = str(params.get("query") or "").strip()
        country = str(params.get("country") or "CO").upper()
        city = str(params.get("city") or "")
        zone_code = str(params.get("zoneCode") or "")
        mode = str(params.get("mode") or "search")  # "search" | "places"
        source_key = "GOOGLE_PLACES" if mode == "places" else "SERPER_GOOGLE"

        cache_key = self.cache.build_key(
            source_key, {"query": query, "country": country, "city": city, "mode": mode}
        )
        cached_hit = self.cache.get(cache_key)
        if cached_hit is not None:
            return cached_hit

        if self.circuit_breaker.is_open():
            msg = f"Circuit breaker abierto para {source_key} tras fallos consecutivos."
            return ConnectorResult(
                source_key=source_key,
                status="CIRCUIT_OPEN",
                data=[],
                citations=[
                    SourceCitation(
                        source_key=source_key,
                        url=f"https://google.serper.dev/{mode}",
                        status="CIRCUIT_OPEN",
                        warning=msg,
                    )
                ],
                warnings=[msg],
            )

        op_key = f"{source_key}:{cache_key[:16]}"
        if self.budget_tracker is not None:
            eval_res = self.budget_tracker.reserve(op_key, self.cost_per_query_usd)
            if not eval_res.allowed:
                msg = eval_res.reason or "Omitido por límite de presupuesto interno."
                return ConnectorResult(
                    source_key=source_key,
                    status="BUDGET_SKIPPED",
                    data=[],
                    citations=[
                        SourceCitation(
                            source_key=source_key,
                            url=f"https://google.serper.dev/{mode}",
                            status="BUDGET_SKIPPED",
                            warning=msg,
                        )
                    ],
                    warnings=[msg],
                )

        await self.rate_limiter.acquire()
        endpoint = f"https://google.serper.dev/{mode}"

        try:
            resp = await self.transport(
                "POST",
                endpoint,
                {"q": query, "gl": country.lower(), "hl": "es"},
            )
        except Exception as exc:
            self.circuit_breaker.record_failure()
            if self.budget_tracker is not None:
                self.budget_tracker.settle(op_key, 0.0)
            warn = f"Timeout o error de red en {source_key} ({query}): {exc}"
            return ConnectorResult(
                source_key=source_key,
                status="TIMEOUT_OR_ERROR",
                data=[],
                citations=[
                    SourceCitation(
                        source_key=source_key,
                        url=endpoint,
                        status="TIMEOUT_OR_ERROR",
                        warning=warn,
                    )
                ],
                warnings=[warn],
            )

        if is_waf_or_forbidden(resp) or resp.status_code >= 400:
            self.circuit_breaker.record_failure()
            if self.budget_tracker is not None:
                self.budget_tracker.settle(op_key, 0.0)
            status_code_label = (
                "BLOCKED_WAF_OR_403"
                if is_waf_or_forbidden(resp)
                else "TIMEOUT_OR_ERROR"
            )
            warn = f"{source_key} respondió HTTP {resp.status_code} para '{query}'."
            return ConnectorResult(
                source_key=source_key,
                status=status_code_label,
                data=[],
                citations=[
                    SourceCitation(
                        source_key=source_key,
                        url=endpoint,
                        status=status_code_label,
                        warning=warn,
                    )
                ],
                warnings=[warn],
            )

        self.circuit_breaker.record_success()
        if self.budget_tracker is not None:
            self.budget_tracker.settle(op_key, self.cost_per_query_usd)

        payload = resp.json() if isinstance(resp.json(), dict) else {}
        items_raw = (
            payload.get("places", [])
            if mode == "places"
            else payload.get("organic", [])
        )

        entities: list[dict[str, Any]] = []
        for idx, item in enumerate(items_raw):
            link = str(item.get("website") or item.get("link") or "").strip()
            if any(junk in link.lower() for junk in JUNK_DOMAINS):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            snippet = str(item.get("snippet") or item.get("address") or "").strip()
            phone = str(item.get("phoneNumber") or item.get("phone") or "sin dato").strip()
            entities.append(
                {
                    "entityId": f"{source_key.lower()}-{idx + 1}",
                    "name": title,
                    "description": snippet or "sin dato",
                    "website": link or "sin dato",
                    "phone": phone or "sin dato",
                    "email": "sin dato",
                    "country": country,
                    "city": city or "sin dato",
                    "zoneCode": zone_code or "sin dato",
                    "source": source_key,
                    "signals": [s for s in [snippet] if s],
                }
            )

        status = "OK" if entities else "EMPTY"
        result = ConnectorResult(
            source_key=source_key,
            status=status,
            data=entities,
            citations=[
                SourceCitation(
                    source_key=source_key,
                    url=endpoint,
                    status=status,
                )
            ],
        )
        self.cache.set(cache_key, result)
        return result
