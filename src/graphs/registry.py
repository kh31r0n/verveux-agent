"""Graph Registry — maps permanent agent CODE NAMES to compiled LangGraph graphs.

A code name (e.g. ``"helena"``, ``"sofia"``) is the permanent identifier for one
graph topology. Code names are platform-level and assigned by developers; once
seeded into ``agent_code_names`` they are never reassigned to a different builder.

Compilation is lazy and protected by an asyncio lock — graphs are built on first
request per code name. The FastAPI lifespan warms up only the code names that
are currently assigned to at least one active channel connection.

Phase-1 deprecation: ``get_agent_graph_by_agent_type()`` provides a fallback for
requests that arrive without ``agent_code_name`` (older NestJS deployments).
The fallback increments ``legacy_agent_type_fallback_total`` and will be removed
in phase 2.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Callable, Iterable, Mapping

import structlog

from .appointments_graph import build_appointments_graph
from .camila_graph import build_camila_graph
from .enrichment_graph import build_enrichment_graph
from .leads_graph import build_leads_graph
from .prospecting_graph import build_prospecting_graph
from .restaurant_graph import build_restaurant_graph
from .sales_graph import build_sales_graph
from .school_graph import build_school_graph

logger = structlog.get_logger(__name__)

GraphBuilder = Callable

# ── Code name registry ────────────────────────────────────────────────────────
# Keys are permanent identifiers; values are the builders. New code names get a
# DB seed migration AND an entry here in the same release.

CODE_NAME_REGISTRY: dict[str, GraphBuilder] = {
    "helena": build_sales_graph,
    "sofia": build_school_graph,
    "camila": build_camila_graph,
    "giulia": build_restaurant_graph,
    "marco": build_appointments_graph,
    "veronica": build_leads_graph,
    "aurora": build_prospecting_graph,
    "sherlock": build_enrichment_graph,
}


def _register_extra_appointment_code_names() -> None:
    """Phase 6: tenants that run multiple appointment-style agents (e.g. a
    dental clinic *and* a salon under the same Verveux deployment) can
    register additional code names that reuse the appointments graph
    topology.

    Each extra code name still needs an ``AgentCodeName`` row on the backend
    so the platform recognises it; this function only declares the Python
    side. The names are read from ``APPOINTMENTS_EXTRA_CODE_NAMES`` as a
    comma-separated list, e.g. ``dental,salon``.

    Why not seed at the call site? Keeping the resolution here means a
    single deploy variable maps a name to the appointments builder for
    every request, without duplicating the registry literal.
    """
    raw = os.getenv("APPOINTMENTS_EXTRA_CODE_NAMES") or ""
    for name in (s.strip() for s in raw.split(",")):
        if not name or name in CODE_NAME_REGISTRY:
            continue
        CODE_NAME_REGISTRY[name] = build_appointments_graph


_register_extra_appointment_code_names()

# Phase-1 fallback: legacy agent_type → canonical code name.
# Removed in phase 2.
AGENT_TYPE_FALLBACK: Mapping[str, str] = {
    "sales": "helena",
    "school": "sofia",
    "restaurant": "giulia",
    "appointments": "marco",
    "leads": "veronica",
    "prospecting": "aurora",
    "enrichment": "sherlock",
}


class UnknownCodeNameError(Exception):
    """Raised when a request references a code name not in CODE_NAME_REGISTRY."""

    def __init__(self, code_name: str, known: Iterable[str]) -> None:
        self.code_name = code_name
        self.known = sorted(known)
        super().__init__(
            f"Unknown agent_code_name '{code_name}'. Known: {self.known}"
        )


# Module-level state populated lazily during request handling and the
# startup warm-up. Guarded by ``_compile_lock`` for concurrent compilation.
_compiled_graphs: dict[str, object] = {}
_compile_lock = asyncio.Lock()
_checkpointer: object | None = None
_store: object | None = None


def set_checkpointer(checkpointer: object) -> None:
    """Cache the checkpointer captured at FastAPI lifespan startup so lazy
    compilations during request handling can reuse it without dependency
    injection plumbing."""
    global _checkpointer
    _checkpointer = checkpointer


def set_store(store: object) -> None:
    """Cache the long-term ``BaseStore`` captured at lifespan startup.

    Used ONLY by the prospecting graph (aurora) for cross-run strategy memory.
    Deliberately NOT compiled into any graph via ``graph.compile(store=...)`` —
    the aurora nodes read it directly via :func:`get_store_or_none`, so the other
    graph builders (single-arg ``builder(ckpt)``) stay untouched. ``None`` when
    the store failed to initialise; callers must degrade gracefully."""
    global _store
    _store = store


def get_store_or_none() -> object | None:
    """Return the cached long-term store, or ``None`` if unavailable.

    Prospecting memory is best-effort: a missing store degrades the agent to its
    stateless behaviour rather than failing the run."""
    return _store


def known_code_names() -> set[str]:
    """Return the set of code names this Python deployment can serve."""
    return set(CODE_NAME_REGISTRY)


def is_compiled(code_name: str) -> bool:
    return code_name in _compiled_graphs


async def get_or_compile_graph(code_name: str, checkpointer: object | None = None):
    """Return the compiled graph for ``code_name``, compiling on first use.

    Raises :class:`UnknownCodeNameError` if the code name is not registered.
    """
    cached = _compiled_graphs.get(code_name)
    if cached is not None:
        return cached

    ckpt = checkpointer if checkpointer is not None else _checkpointer
    if ckpt is None:
        raise RuntimeError(
            "Checkpointer is not initialised; cannot compile graphs"
        )

    async with _compile_lock:
        cached = _compiled_graphs.get(code_name)
        if cached is not None:
            return cached

        builder = CODE_NAME_REGISTRY.get(code_name)
        if builder is None:
            raise UnknownCodeNameError(code_name, CODE_NAME_REGISTRY)

        # Imported lazily so unit tests can patch the metric module without
        # forcing prometheus_client at registry import time.
        from ..observability import graph_compile_duration

        start = time.perf_counter()
        graph = builder(ckpt)
        duration = time.perf_counter() - start
        _compiled_graphs[code_name] = graph

        try:
            graph_compile_duration.labels(agent_code_name=code_name).observe(
                duration
            )
        except Exception:  # pragma: no cover — metric failure must not block serving
            pass

        logger.info(
            "graph_compiled",
            agent_code_name=code_name,
            duration_seconds=round(duration, 4),
        )
        return graph


async def warm_up(code_names: Iterable[str], checkpointer: object) -> list[str]:
    """Pre-compile graphs for the supplied code names. Returns the list that
    was actually compiled (excludes unknown names; the caller is responsible
    for surfacing startup validation errors)."""
    set_checkpointer(checkpointer)
    compiled: list[str] = []
    for name in code_names:
        if name not in CODE_NAME_REGISTRY:
            logger.warning("warm_up_unknown_code_name", code_name=name)
            continue
        await get_or_compile_graph(name, checkpointer)
        compiled.append(name)
    if compiled:
        logger.info("warm_up_complete", code_names=compiled)
    return compiled


def resolve_legacy_agent_type(agent_type: str) -> str | None:
    """Phase-1 only: map a legacy agent_type to its canonical code name.

    Returns ``None`` if the agent_type is not recognised, so the caller can
    raise an HTTP 400 instead of silently routing to a default.
    """
    return AGENT_TYPE_FALLBACK.get((agent_type or "").lower())
