"""Public Business Registries connector (Colombia RUES / Socrata Open Data API + regional fallback)."""

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

RUES_SOCRATA_ENDPOINT = "https://www.datos.gov.co/resource/rues-empresas.json"


class PublicRegistryConnector:
    def __init__(
        self,
        transport: HttpTransportFn,
        cache: CalebCache | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.source_key = "PUBLIC_REGISTRY_CO_RUES"
        self.transport = transport
        self.cache = cache or CalebCache()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.rate_limiter = rate_limiter or RateLimiter()

    async def fetch(self, params: dict[str, Any]) -> ConnectorResult:
        country = str(params.get("country") or "CO").upper()
        city = str(params.get("city") or "")
        keyword = str(params.get("keyword") or "")

        if country != "CO":
            warn = (
                f"El país {country} no cuenta con API abierta de registro mercantil "
                "automatizado; campos de registro mercantil se reportan como 'sin dato'."
            )
            return ConnectorResult(
                source_key=self.source_key,
                status="EMPTY",
                data=[],
                citations=[
                    SourceCitation(
                        source_key=self.source_key,
                        url=f"registry://{country.lower()}/no-open-api",
                        status="EMPTY",
                        warning=warn,
                    )
                ],
                warnings=[warn],
            )

        cache_key = self.cache.build_key(
            self.source_key, {"country": country, "city": city, "keyword": keyword}
        )
        cached_hit = self.cache.get(cache_key)
        if cached_hit is not None:
            return cached_hit

        if self.circuit_breaker.is_open():
            warn = "Circuit breaker abierto para PUBLIC_REGISTRY_CO_RUES."
            return ConnectorResult(
                source_key=self.source_key,
                status="CIRCUIT_OPEN",
                data=[],
                citations=[
                    SourceCitation(
                        source_key=self.source_key,
                        url=RUES_SOCRATA_ENDPOINT,
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
                RUES_SOCRATA_ENDPOINT,
                {"municipio": city, "q": keyword},
            )
        except Exception as exc:
            self.circuit_breaker.record_failure()
            warn = f"Error consultando registro público RUES/Socrata: {exc}"
            return ConnectorResult(
                source_key=self.source_key,
                status="TIMEOUT_OR_ERROR",
                data=[],
                citations=[
                    SourceCitation(
                        source_key=self.source_key,
                        url=RUES_SOCRATA_ENDPOINT,
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
            warn = f"RUES/Socrata respondió HTTP {resp.status_code}; datos de registro marcados como 'sin dato'."
            return ConnectorResult(
                source_key=self.source_key,
                status=st,
                data=[],
                citations=[
                    SourceCitation(
                        source_key=self.source_key,
                        url=RUES_SOCRATA_ENDPOINT,
                        status=st,
                        warning=warn,
                    )
                ],
                warnings=[warn],
            )

        self.circuit_breaker.record_success()
        rows = resp.json() if isinstance(resp.json(), list) else []
        normalized: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            razon = str(row.get("razon_social") or row.get("nombre") or "").strip()
            if not razon:
                continue
            nit = str(row.get("nit") or row.get("identificacion") or "sin dato").strip()
            ciiu = str(row.get("ciiu") or "sin dato").strip()
            estado = str(row.get("estado_matricula") or "ACTIVA").strip()
            normalized.append(
                {
                    "entityId": f"rues-{idx + 1}",
                    "name": razon,
                    "taxId": nit or "sin dato",
                    "ciiu": ciiu or "sin dato",
                    "registryStatus": estado or "sin dato",
                    "country": "CO",
                    "city": str(row.get("municipio") or city or "sin dato"),
                    "phone": str(row.get("telefono_comercial_1") or "sin dato"),
                    "email": str(row.get("correo_electronico_comercial") or "sin dato"),
                    "website": "sin dato",
                    "source": self.source_key,
                    "signals": [f"ciiu:{ciiu}", f"matricula:{estado}"],
                }
            )

        status = "OK" if normalized else "EMPTY"
        result = ConnectorResult(
            source_key=self.source_key,
            status=status,
            data=normalized,
            citations=[
                SourceCitation(
                    source_key=self.source_key,
                    url=RUES_SOCRATA_ENDPOINT,
                    status=status,
                )
            ],
        )
        self.cache.set(cache_key, result)
        return result
