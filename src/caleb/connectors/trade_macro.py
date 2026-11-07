"""Foreign Trade (UN Comtrade), Tariffs (WITS), and Macro Indicators (World Bank) connector."""

from __future__ import annotations

from typing import Any

from .base import (
    CalebCache,
    CircuitBreaker,
    ConnectorResult,
    HttpTransportFn,
    RateLimiter,
    SourceCitation,
    is_waf_or_forbidden,
)

COMTRADE_ENDPOINT = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
WITS_ENDPOINT = "https://wits.worldbank.org/API/V1/SDMX/V21/datasource/TRN"
WORLD_BANK_ENDPOINT = "https://api.worldbank.org/v2/country"


class TradeAndMacroConnector:
    def __init__(
        self,
        transport: HttpTransportFn,
        cache: CalebCache | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.source_key = "TRADE_UN_COMTRADE"
        self.transport = transport
        self.cache = cache or CalebCache()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.rate_limiter = rate_limiter or RateLimiter()

    async def fetch(self, params: dict[str, Any]) -> ConnectorResult:
        country = str(params.get("country") or "PA").upper()
        hs_codes = list(params.get("hsCodes6") or ["731815"])
        kind = str(params.get("kind") or "comtrade")  # "comtrade" | "wits" | "world_bank"

        source_key = (
            "TRADE_WITS_TARIFF"
            if kind == "wits"
            else ("MACRO_WORLD_BANK" if kind == "world_bank" else "TRADE_UN_COMTRADE")
        )
        endpoint = (
            WITS_ENDPOINT
            if kind == "wits"
            else (
                f"{WORLD_BANK_ENDPOINT}/{country.lower()}/indicator"
                if kind == "world_bank"
                else COMTRADE_ENDPOINT
            )
        )

        cache_key = self.cache.build_key(
            source_key, {"country": country, "hsCodes6": hs_codes, "kind": kind}
        )
        cached_hit = self.cache.get(cache_key)
        if cached_hit is not None:
            return cached_hit

        if self.circuit_breaker.is_open():
            warn = f"Circuit breaker abierto para {source_key}."
            return ConnectorResult(
                source_key=source_key,
                status="CIRCUIT_OPEN",
                data={
                    "country": country,
                    "importsUsd": "sin dato",
                    "appliedTariffPct": "sin dato",
                    "gdpUsd": "sin dato",
                },
                citations=[
                    SourceCitation(
                        source_key=source_key,
                        url=endpoint,
                        status="CIRCUIT_OPEN",
                        warning=warn,
                    )
                ],
                warnings=[warn],
            )

        await self.rate_limiter.acquire()
        try:
            resp = await self.transport(
                "GET",
                endpoint,
                {"country": country, "hsCodes6": ",".join(hs_codes)},
            )
        except Exception as exc:
            self.circuit_breaker.record_failure()
            warn = f"Error consultando {source_key} para {country}: {exc}. Se reporta 'sin dato'."
            return ConnectorResult(
                source_key=source_key,
                status="TIMEOUT_OR_ERROR",
                data={
                    "country": country,
                    "importsUsd": "sin dato",
                    "appliedTariffPct": "sin dato",
                    "gdpUsd": "sin dato",
                },
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
            st = (
                "BLOCKED_WAF_OR_403"
                if is_waf_or_forbidden(resp)
                else "TIMEOUT_OR_ERROR"
            )
            warn = f"{source_key} respondió HTTP {resp.status_code} para {country}; series marcadas como 'sin dato'."
            return ConnectorResult(
                source_key=source_key,
                status=st,
                data={
                    "country": country,
                    "importsUsd": "sin dato",
                    "appliedTariffPct": "sin dato",
                    "gdpUsd": "sin dato",
                },
                citations=[
                    SourceCitation(
                        source_key=source_key,
                        url=endpoint,
                        status=st,
                        warning=warn,
                    )
                ],
                warnings=[warn],
            )

        self.circuit_breaker.record_success()
        payload = resp.json() if isinstance(resp.json(), dict) else {}

        data_out = {
            "country": country,
            "hsCodes6": hs_codes,
            "importsUsd": payload.get("importsUsd", "sin dato"),
            "exportsUsd": payload.get("exportsUsd", "sin dato"),
            "appliedTariffPct": payload.get("appliedTariffPct", "sin dato"),
            "gdpUsd": payload.get("gdpUsd", "sin dato"),
            "logisticsPerformanceIndex": payload.get(
                "logisticsPerformanceIndex", "sin dato"
            ),
        }
        status = (
            "OK"
            if any(
                v != "sin dato"
                for k, v in data_out.items()
                if k not in ("country", "hsCodes6")
            )
            else "EMPTY"
        )
        result = ConnectorResult(
            source_key=source_key,
            status=status,
            data=data_out,
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
