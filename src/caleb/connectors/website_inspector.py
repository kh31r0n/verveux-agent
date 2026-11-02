"""B2B Website Inspector connector with strict SSRF protection and deterministic brand/contact/signal extraction."""

from __future__ import annotations

import ipaddress
import re
from typing import Any
from urllib.parse import urlparse

from .base import (
    CalebCache,
    CircuitBreaker,
    ConnectorResult,
    HttpTransportFn,
    RateLimiter,
    SourceCitation,
    is_waf_or_forbidden,
)

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,10}")
PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}")
IGNORED_EMAIL_DOMAINS = ("example.com", "sentry.io", "wixpress.com", "schema.org")
IGNORED_EMAIL_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")

B2B_COMMERCIAL_SIGNALS: tuple[str, ...] = (
    "importador directo",
    "ventas al por mayor",
    "mayorista",
    "distribuidor autorizado",
    "catálogo industrial",
    "ficha técnica",
    "mantenimiento industrial",
    "cotización",
    "cobertura nacional",
    "certificación iso",
)

FORBIDDEN_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        "metadata",
        "169.254.169.254",
        "0.0.0.0",
    }
)


def validate_public_http_url(url: str) -> tuple[bool, str | None]:
    """Validates that a URL uses http/https and does not target private/loopback/link-local SSRF addresses."""
    if not url or not isinstance(url, str):
        return False, "URL vacía o inválida"
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in ("http", "https"):
        return False, f"Esquema no permitido ({parsed.scheme}): solo se permite http/https"

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        return False, "URL sin hostname válido"

    if (
        hostname in FORBIDDEN_HOSTNAMES
        or hostname.endswith(".local")
        or hostname.endswith(".internal")
        or hostname.endswith(".localhost")
    ):
        return False, f"Hostname bloqueado por protección SSRF ({hostname})"

    try:
        ip_obj = ipaddress.ip_address(hostname)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
        ):
            return False, f"Dirección IP no pública bloqueada por protección SSRF ({hostname})"
    except ValueError:
        # Hostname is a domain name, not an IP literal
        pass

    return True, None


class WebsiteInspectorConnector:
    def __init__(
        self,
        transport: HttpTransportFn,
        cache: CalebCache | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.source_key = "COMPANY_WEBSITES"
        self.transport = transport
        self.cache = cache or CalebCache()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.rate_limiter = rate_limiter or RateLimiter()

    async def fetch(self, params: dict[str, Any]) -> ConnectorResult:
        url = str(params.get("url") or "").strip()
        reference_brands: list[str] = [
            str(b.get("name") if isinstance(b, dict) else b).strip()
            for b in (params.get("referenceBrands") or [])
            if b
        ]
        product_keywords: list[str] = [
            str(k).strip() for k in (params.get("productKeywords") or []) if k
        ]

        allowed, ssrf_reason = validate_public_http_url(url)
        if not allowed:
            warn = f"URL rechazada por seguridad SSRF ({url}): {ssrf_reason}"
            return ConnectorResult(
                source_key=self.source_key,
                status="SSRF_BLOCKED",
                data={
                    "url": url,
                    "emails": "sin dato",
                    "phones": "sin dato",
                    "brandsFound": [],
                    "signalsFound": [],
                },
                citations=[
                    SourceCitation(
                        source_key=self.source_key,
                        url=url,
                        status="SSRF_BLOCKED",
                        warning=warn,
                    )
                ],
                warnings=[warn],
            )

        cache_key = self.cache.build_key(
            self.source_key,
            {
                "url": url,
                "referenceBrands": sorted(reference_brands),
                "productKeywords": sorted(product_keywords),
            },
        )
        cached_hit = self.cache.get(cache_key)
        if cached_hit is not None:
            return cached_hit

        if self.circuit_breaker.is_open():
            warn = "Circuit breaker abierto para COMPANY_WEBSITES."
            return ConnectorResult(
                source_key=self.source_key,
                status="CIRCUIT_OPEN",
                data={
                    "url": url,
                    "emails": "sin dato",
                    "phones": "sin dato",
                    "brandsFound": [],
                    "signalsFound": [],
                },
                warnings=[warn],
            )

        await self.rate_limiter.acquire()
        try:
            resp = await self.transport("GET", url, {})
        except Exception as exc:
            self.circuit_breaker.record_failure()
            warn = f"Timeout/error inspeccionando sitio web {url}: {exc}"
            return ConnectorResult(
                source_key=self.source_key,
                status="TIMEOUT_OR_ERROR",
                data={
                    "url": url,
                    "emails": "sin dato",
                    "phones": "sin dato",
                    "brandsFound": [],
                    "signalsFound": [],
                },
                citations=[
                    SourceCitation(
                        source_key=self.source_key,
                        url=url,
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
            warn = f"Sitio web {url} respondió HTTP {resp.status_code}; datos de sitio web marcados como 'sin dato'."
            return ConnectorResult(
                source_key=self.source_key,
                status=st,
                data={
                    "url": url,
                    "emails": "sin dato",
                    "phones": "sin dato",
                    "brandsFound": [],
                    "signalsFound": [],
                },
                citations=[
                    SourceCitation(
                        source_key=self.source_key,
                        url=url,
                        status=st,
                        warning=warn,
                    )
                ],
                warnings=[warn],
            )

        self.circuit_breaker.record_success()
        html = resp.text or ""
        text_lower = html.lower()

        emails: list[str] = []
        for match in EMAIL_RE.findall(html):
            m_low = match.lower()
            if any(m_low.endswith(ext) for ext in IGNORED_EMAIL_EXTENSIONS):
                continue
            if any(dom in m_low for dom in IGNORED_EMAIL_DOMAINS):
                continue
            if m_low not in emails:
                emails.append(m_low)

        phones: list[str] = []
        for match in PHONE_RE.findall(html):
            cleaned = match.strip()
            digits_only = re.sub(r"\D", "", cleaned)
            if 7 <= len(digits_only) <= 15 and cleaned not in phones:
                phones.append(cleaned)

        brands_found = [
            b for b in reference_brands if b and b.lower() in text_lower
        ]
        signals_found = [
            sig for sig in B2B_COMMERCIAL_SIGNALS if sig in text_lower
        ]
        for kw in product_keywords:
            if kw.lower() in text_lower and kw not in signals_found:
                signals_found.append(kw)

        has_whatsapp = "wa.me/" in text_lower or "api.whatsapp.com" in text_lower

        result_data = {
            "url": url,
            "email": emails[0] if emails else "sin dato",
            "emails": emails if emails else "sin dato",
            "phone": phones[0] if phones else "sin dato",
            "phones": phones if phones else "sin dato",
            "hasWhatsapp": has_whatsapp,
            "brandsFound": brands_found,
            "signalsFound": signals_found,
        }
        result = ConnectorResult(
            source_key=self.source_key,
            status="OK",
            data=result_data,
            citations=[
                SourceCitation(
                    source_key=self.source_key,
                    url=url,
                    status="OK",
                )
            ],
        )
        self.cache.set(cache_key, result)
        return result
