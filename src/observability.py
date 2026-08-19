import structlog
from prometheus_client import Counter, Histogram

from .config import settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus counters
# ---------------------------------------------------------------------------

agent_requests_total = Counter(
    "agent_requests_total",
    "Total number of chat requests processed",
    ["agent_code_name"],
)

agent_interrupt_events_total = Counter(
    "agent_interrupt_events_total",
    "Total number of interrupt events emitted",
)

agent_node_invocations_total = Counter(
    "agent_node_invocations_total",
    "Total number of node invocations",
    ["agent_code_name", "node"],
)

agent_tool_errors_total = Counter(
    "agent_tool_errors_total",
    "Total number of tool/node errors",
    ["agent_code_name", "tool"],
)

graph_compile_duration = Histogram(
    "graph_compile_duration_seconds",
    "Time spent compiling a LangGraph topology on first request",
    ["agent_code_name"],
)

# Phase-1 deprecation canary. Increments any time a request arrives without an
# agent_code_name and is resolved through the legacy agent_type fallback. When
# this stays at zero for 30 days, the fallback path is removed.
legacy_agent_type_fallback_total = Counter(
    "legacy_agent_type_fallback_total",
    "Requests that fell back from agent_type to a canonical code name",
)

orders_started_total = Counter(
    "orders_started_total",
    "Total number of sales orders initiated",
)

orders_confirmed_total = Counter(
    "orders_confirmed_total",
    "Total number of orders confirmed and executed",
)

tracking_requests_total = Counter(
    "tracking_requests_total",
    "Total number of tracking requests",
)

complaints_total = Counter(
    "complaints_total",
    "Total number of complaints registered",
)

# Outcomes: applied | discarded_high_risk | discarded_low_confidence |
# no_trigger | retrieval_empty | retrieval_failed | error
query_normalizations_total = Counter(
    "query_normalizations_total",
    "Query normalization outcomes per turn",
    ["outcome"],
)

# Reasons: no_credentials | invalid_token | caller_not_allowed |
# email_unverified | token_decode_error | jwks_unavailable |
# allowlist_unconfigured
service_auth_rejects_total = Counter(
    "service_auth_rejects_total",
    "Rejected service-to-service authentication attempts",
    ["reason"],
)


def record_node_invocation(node: str, agent_code_name: str = "unknown") -> None:
    agent_node_invocations_total.labels(
        agent_code_name=agent_code_name, node=node
    ).inc()


def record_tool_error(tool: str, agent_code_name: str = "unknown") -> None:
    agent_tool_errors_total.labels(
        agent_code_name=agent_code_name, tool=tool
    ).inc()


# ---------------------------------------------------------------------------
# No-op tracing stubs — used when Langfuse is unconfigured or the installed
# SDK version has an incompatible API (e.g. langfuse 4.x removed .trace()).
# ---------------------------------------------------------------------------


class _NoOpGeneration:
    def end(self, **kwargs) -> None:
        pass


class _NoOpTrace:
    def generation(self, **kwargs) -> _NoOpGeneration:
        return _NoOpGeneration()

    def span(self, **kwargs) -> "_NoOpTrace":
        return self

    def update(self, **kwargs) -> None:
        pass


class _NoOpLangfuse:
    def trace(self, **kwargs) -> _NoOpTrace:
        return _NoOpTrace()

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Langfuse singleton
# ---------------------------------------------------------------------------

_langfuse_instance: _NoOpLangfuse | None = None


def get_langfuse() -> _NoOpLangfuse:
    global _langfuse_instance
    if _langfuse_instance is not None:
        return _langfuse_instance

    if not settings.langfuse_secret_key or not settings.langfuse_public_key:
        logger.debug("langfuse_disabled", reason="keys not configured")
        _langfuse_instance = _NoOpLangfuse()
        return _langfuse_instance

    try:
        from langfuse import Langfuse

        client = Langfuse(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host,
        )
        if not callable(getattr(client, "trace", None)):
            raise AttributeError("Langfuse.trace() not available — SDK version incompatible")
        _langfuse_instance = client  # type: ignore[assignment]
        logger.info("langfuse_ready", host=settings.langfuse_host)
    except Exception as exc:
        logger.warning("langfuse_init_failed", error=str(exc))
        _langfuse_instance = _NoOpLangfuse()

    return _langfuse_instance
