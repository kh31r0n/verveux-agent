import structlog
from prometheus_client import Counter

from .config import settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus counters
# ---------------------------------------------------------------------------

agent_requests_total = Counter(
    "agent_requests_total",
    "Total number of chat requests processed",
)

agent_interrupt_events_total = Counter(
    "agent_interrupt_events_total",
    "Total number of interrupt events emitted",
)

agent_node_invocations_total = Counter(
    "agent_node_invocations_total",
    "Total number of node invocations",
    ["node"],
)

agent_tool_errors_total = Counter(
    "agent_tool_errors_total",
    "Total number of tool/node errors",
    ["tool"],
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


def record_node_invocation(node: str) -> None:
    agent_node_invocations_total.labels(node=node).inc()


def record_tool_error(tool: str) -> None:
    agent_tool_errors_total.labels(tool=tool).inc()


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
