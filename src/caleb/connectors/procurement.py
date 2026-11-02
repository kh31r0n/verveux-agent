"""Public Procurement connector across all 19 regional countries (CO SECOP II, Central America, South America).

Explicitly handles WAF/Cloudflare/403 blocks and portals without open APIs by
reporting ``"sin dato"`` and attaching a documented warning.
"""

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

PROCUREMENT_PORTAL_CATALOG: dict[str, dict[str, str]] = {
    "CO": {
        "source_key": "PROCUREMENT_CO_SECOP",
        "name": "SECOP II (Datos Abiertos Colombia)",
        "url": "https://www.datos.gov.co/resource/jbjy-vk9h.json",
        "access": "OPEN_API",
    },
    # Central America (7)
    "GT": {
        "source_key": "PROCUREMENT_GT_GUATECOMPRAS",
        "name": "Guatecompras",
        "url": "https://www.guatecompras.gt/consultas/api",
        "access": "PUBLIC_PORTAL",
    },
    "SV": {
        "source_key": "PROCUREMENT_SV_COMPRASAL",
        "name": "Comprasal",
        "url": "https://www.comprasal.gob.sv/comprasalweb/api",
        "access": "PUBLIC_PORTAL",
    },
    "HN": {
        "source_key": "PROCUREMENT_HN_HONDUCOMPRAS",
        "name": "HonduCompras",
        "url": "https://www.honducompras.gob.hn/api",
        "access": "PUBLIC_PORTAL",
    },
    "NI": {
        "source_key": "PROCUREMENT_NI_SISCAE",
        "name": "SISCAE Nicaragua",
        "url": "https://www.nicaraguacompra.gob.ni",
        "access": "NO_OPEN_API",
    },
    "CR": {
        "source_key": "PROCUREMENT_CR_SICOP",
        "name": "SICOP Costa Rica",
        "url": "https://www.sicop.go.cr/api",
        "access": "PUBLIC_PORTAL",
    },
    "PA": {
        "source_key": "PROCUREMENT_PA_PANAMACOMPRAS",
        "name": "PanamaCompras",
        "url": "https://www.panamacompra.gob.pa/api",
        "access": "PUBLIC_PORTAL",
    },
    "BZ": {
        "source_key": "PROCUREMENT_SOUTH_AMERICA_OPEN",
        "name": "Belize Procurement Portal",
        "url": "https://procurement.gov.bz",
        "access": "NO_OPEN_API",
    },
    # South America (11)
    "EC": {
        "source_key": "PROCUREMENT_SOUTH_AMERICA_OPEN",
        "name": "SERCOP Datos Abiertos Ecuador",
        "url": "https://datosabiertos.compraspublicas.gob.ec/PLATAFORMA/api",
        "access": "OPEN_API",
    },
    "PE": {
        "source_key": "PROCUREMENT_SOUTH_AMERICA_OPEN",
        "name": "SEACE / OSCE Perú",
        "url": "https://prodapp2.seace.gob.pe/seacebus-uiwd-pub",
        "access": "PUBLIC_PORTAL",
    },
    "CL": {
        "source_key": "PROCUREMENT_SOUTH_AMERICA_OPEN",
        "name": "MercadoPúblico ChileCompra",
        "url": "https://api.mercadopublico.cl/servicios/v1/publico",
        "access": "OPEN_API",
    },
    "AR": {
        "source_key": "PROCUREMENT_SOUTH_AMERICA_OPEN",
        "name": "COMPR.AR Argentina",
        "url": "https://comprar.gob.ar/api",
        "access": "PUBLIC_PORTAL",
    },
    "BR": {
        "source_key": "PROCUREMENT_SOUTH_AMERICA_OPEN",
        "name": "PNCP / Compras.gov.br Brasil",
        "url": "https://pncp.gov.br/api/consulta/v1",
        "access": "OPEN_API",
    },
    "BO": {
        "source_key": "PROCUREMENT_SOUTH_AMERICA_OPEN",
        "name": "SICOES Bolivia",
        "url": "https://www.sicoes.gob.bo",
        "access": "NO_OPEN_API",
    },
    "PY": {
        "source_key": "PROCUREMENT_SOUTH_AMERICA_OPEN",
        "name": "DNCP Contrataciones Paraguay (OCDS)",
        "url": "https://www.contrataciones.gov.py/datos/api/v3",
        "access": "OPEN_API",
    },
    "UY": {
        "source_key": "PROCUREMENT_SOUTH_AMERICA_OPEN",
        "name": "ACCE Compras Estatales Uruguay",
        "url": "https://www.comprasestatales.gub.uy/api",
        "access": "OPEN_API",
    },
    "VE": {
        "source_key": "PROCUREMENT_SOUTH_AMERICA_OPEN",
        "name": "SNC Venezuela",
        "url": "http://www.snc.gob.ve",
        "access": "NO_OPEN_API",
    },
    "GY": {
        "source_key": "PROCUREMENT_SOUTH_AMERICA_OPEN",
        "name": "NPTAB Guyana",
        "url": "https://www.nptab.gov.gy",
        "access": "NO_OPEN_API",
    },
    "SR": {
        "source_key": "PROCUREMENT_SOUTH_AMERICA_OPEN",
        "name": "Suriname Public Procurement",
        "url": "https://gov.sr",
        "access": "NO_OPEN_API",
    },
}


class ProcurementConnector:
    def __init__(
        self,
        transport: HttpTransportFn,
        cache: CalebCache | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.source_key = "PROCUREMENT_CO_SECOP"
        self.transport = transport
        self.cache = cache or CalebCache()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.rate_limiter = rate_limiter or RateLimiter()

    async def fetch(self, params: dict[str, Any]) -> ConnectorResult:
        country = str(params.get("country") or "CO").upper()
        keyword = str(params.get("keyword") or "")
        unspsc = str(params.get("unspsc") or "")

        portal = PROCUREMENT_PORTAL_CATALOG.get(country)
        if portal is None:
            warn = f"País {country} sin portal de compras públicas registrado; se reporta 'sin dato'."
            return ConnectorResult(
                source_key=self.source_key,
                status="EMPTY",
                data={"tendersCount": "sin dato", "suppliers": []},
                warnings=[warn],
            )

        src_key = portal["source_key"]
        portal_url = portal["url"]
        portal_name = portal["name"]

        if portal["access"] == "NO_OPEN_API":
            warn = (
                f"{portal_name} ({country}) no ofrece API abierta automatizable; "
                "actividad de compras públicas marcada como 'sin dato'."
            )
            return ConnectorResult(
                source_key=src_key,
                status="EMPTY",
                data={"country": country, "tendersCount": "sin dato", "suppliers": []},
                citations=[
                    SourceCitation(
                        source_key=src_key,
                        url=portal_url,
                        status="EMPTY",
                        warning=warn,
                    )
                ],
                warnings=[warn],
            )

        cache_key = self.cache.build_key(
            src_key, {"country": country, "keyword": keyword, "unspsc": unspsc}
        )
        cached_hit = self.cache.get(cache_key)
        if cached_hit is not None:
            return cached_hit

        if self.circuit_breaker.is_open():
            warn = f"Circuit breaker abierto para {portal_name} ({country})."
            return ConnectorResult(
                source_key=src_key,
                status="CIRCUIT_OPEN",
                data={"country": country, "tendersCount": "sin dato", "suppliers": []},
                citations=[
                    SourceCitation(
                        source_key=src_key,
                        url=portal_url,
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
                portal_url,
                {"q": keyword, "unspsc": unspsc, "country": country},
            )
        except Exception as exc:
            self.circuit_breaker.record_failure()
            warn = f"Timeout/error en {portal_name} ({country}): {exc}. Se reporta 'sin dato'."
            return ConnectorResult(
                source_key=src_key,
                status="TIMEOUT_OR_ERROR",
                data={"country": country, "tendersCount": "sin dato", "suppliers": []},
                citations=[
                    SourceCitation(
                        source_key=src_key,
                        url=portal_url,
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
            warn = (
                f"Portal {portal_name} ({country}) bloqueó la consulta automatizada "
                f"(HTTP {resp.status_code} / WAF). Se reporta 'sin dato' sin inventar adjudicaciones."
            )
            return ConnectorResult(
                source_key=src_key,
                status=st,
                data={"country": country, "tendersCount": "sin dato", "suppliers": []},
                citations=[
                    SourceCitation(
                        source_key=src_key,
                        url=portal_url,
                        status=st,
                        warning=warn,
                    )
                ],
                warnings=[warn],
            )

        self.circuit_breaker.record_success()
        rows = resp.json() if isinstance(resp.json(), list) else []
        suppliers: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            supplier_name = str(
                row.get("proveedor_adjudicado")
                or row.get("supplier_name")
                or row.get("adjudicatario")
                or ""
            ).strip()
            if not supplier_name:
                continue
            suppliers.append(
                {
                    "entityId": f"proc-{country.lower()}-{idx + 1}",
                    "name": supplier_name,
                    "taxId": str(
                        row.get("documento_proveedor") or row.get("tax_id") or "sin dato"
                    ),
                    "country": country,
                    "city": str(row.get("ciudad") or "sin dato"),
                    "publicContractsCount": int(row.get("contratos_count") or 1),
                    "contractAmountFormatted": str(
                        row.get("valor_del_contrato") or "sin dato"
                    ),
                    "source": src_key,
                    "signals": [f"adjudicatario publico {portal_name}"],
                }
            )

        status = "OK" if suppliers else "EMPTY"
        result = ConnectorResult(
            source_key=src_key,
            status=status,
            data={
                "country": country,
                "portalName": portal_name,
                "tendersCount": len(rows),
                "suppliers": suppliers,
            },
            citations=[
                SourceCitation(
                    source_key=src_key,
                    url=portal_url,
                    status=status,
                )
            ],
        )
        self.cache.set(cache_key, result)
        return result
