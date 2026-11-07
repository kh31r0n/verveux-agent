"""Base protocol, deterministic cache, circuit breaker, rate limiter, and source citations for Caleb connectors."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal, Protocol

ConnectorStatus = Literal[
    "OK",
    "EMPTY",
    "BLOCKED_WAF_OR_403",
    "TIMEOUT_OR_ERROR",
    "CIRCUIT_OPEN",
    "SSRF_BLOCKED",
    "BUDGET_SKIPPED",
]


@dataclass
class SourceCitation:
    source_key: str
    url: str
    status: ConnectorStatus
    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    cached: bool = False
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceKey": self.source_key,
            "url": self.url,
            "status": self.status,
            "fetchedAt": self.fetched_at,
            "cached": self.cached,
            "warning": self.warning,
        }


@dataclass
class ConnectorResult:
    source_key: str
    status: ConnectorStatus
    data: Any
    citations: list[SourceCitation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceKey": self.source_key,
            "status": self.status,
            "data": self.data,
            "citations": [c.to_dict() for c in self.citations],
            "warnings": list(self.warnings),
            "cached": self.cached,
        }


@dataclass
class HttpResponseStub:
    """Normalized HTTP response returned by injectable transport callables."""

    status_code: int
    text: str = ""
    json_data: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    url: str = ""

    def json(self) -> Any:
        if self.json_data is not None:
            return self.json_data
        return json.loads(self.text or "{}")


HttpTransportFn = Callable[[str, str, dict[str, Any]], Awaitable[HttpResponseStub]]


class CalebCache:
    """Deterministic TTL cache for connector responses."""

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, ConnectorResult]] = {}

    @staticmethod
    def build_key(source_key: str, params: dict[str, Any]) -> str:
        canonical = json.dumps(params, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"{source_key}:{digest}"

    def get(self, key: str) -> ConnectorResult | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, result = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        cached_citations = [
            SourceCitation(
                source_key=c.source_key,
                url=c.url,
                status=c.status,
                fetched_at=c.fetched_at,
                cached=True,
                warning=c.warning,
            )
            for c in result.citations
        ]
        return ConnectorResult(
            source_key=result.source_key,
            status=result.status,
            data=result.data,
            citations=cached_citations,
            warnings=list(result.warnings),
            cached=True,
        )

    def set(self, key: str, result: ConnectorResult) -> None:
        self._store[key] = (time.monotonic() + self.ttl_seconds, result)


class CircuitBreaker:
    """Per-connector circuit breaker that prevents cascading timeouts/WAF blocks."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_sec: float = 60.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.failure_count = 0
        self.opened_at: float | None = None

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if (time.monotonic() - self.opened_at) >= self.recovery_timeout_sec:
            # Half-open: allow next attempt
            self.opened_at = None
            self.failure_count = 0
            return False
        return True

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.opened_at = time.monotonic()


class RateLimiter:
    """Async minimum-interval rate limiter."""

    def __init__(self, min_interval_sec: float = 0.0) -> None:
        self.min_interval_sec = min_interval_sec
        self._last_call: float = 0.0

    async def acquire(self) -> None:
        if self.min_interval_sec <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval_sec:
            await asyncio.sleep(self.min_interval_sec - elapsed)
        self._last_call = time.monotonic()


def is_waf_or_forbidden(resp: HttpResponseStub) -> bool:
    """Detects HTTP 403/429 or Cloudflare/Incapsula WAF challenge pages."""
    if resp.status_code in (401, 403, 429, 503):
        body_lower = (resp.text or "").lower()
        if (
            resp.status_code in (403, 429)
            or "cloudflare" in body_lower
            or "attention required" in body_lower
            or "access denied" in body_lower
            or "incapsula" in body_lower
            or "cf-ray" in {k.lower() for k in resp.headers}
        ):
            return True
    return False


class CalebDataConnector(Protocol):
    source_key: str

    async def fetch(self, params: dict[str, Any]) -> ConnectorResult: ...
