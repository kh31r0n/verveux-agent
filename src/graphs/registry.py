"""Graph Registry — maps agent_type strings to compiled LangGraph graphs.

Each agent domain gets its own fully isolated, pre-compiled graph.  The
``get_agent_graph()`` function is the single dispatcher used by ``main.py``
at request time.

Design rationale (RFC V3):
- Complete graph isolation prevents state pollution between domains.
- Each graph is compiled once at startup and reused for all requests.
- Unknown agent types fall back to SALES with a warning log.
"""

import logging
from typing import Callable

import structlog

from .appointments_graph import build_appointments_graph
from .restaurant_graph import build_restaurant_graph
from .sales_graph import build_sales_graph
from .school_graph import build_school_graph

logger = structlog.get_logger(__name__)

# Type alias for graph builder functions
GraphBuilder = Callable

# ── Registry ──────────────────────────────────────────────────────────────────
# Keys MUST match the lowercase AgentType enum values sent by NestJS:
#   SALES → "sales", SCHOOL → "school", etc.

AGENT_REGISTRY: dict[str, GraphBuilder] = {
    "sales": build_sales_graph,
    "school": build_school_graph,
    "restaurant": build_restaurant_graph,
    "appointments": build_appointments_graph,
}

# ── Compiled graph cache (populated at startup) ──────────────────────────────

_compiled_graphs: dict[str, object] = {}


def compile_all_graphs(checkpointer) -> None:
    """Compile every registered graph with the given checkpointer.

    Called once during FastAPI lifespan startup.
    """
    for agent_type, builder in AGENT_REGISTRY.items():
        _compiled_graphs[agent_type] = builder(checkpointer)
        logger.info("graph_compiled", agent_type=agent_type)

    logger.info(
        "all_graphs_compiled",
        agent_types=list(_compiled_graphs.keys()),
    )


def get_agent_graph(agent_type: str):
    """Return the compiled graph for the given agent_type.

    Falls back to SALES if the agent_type is unknown, logging a warning.
    Returns ``None`` only if ``compile_all_graphs()`` has not been called.
    """
    normalized = agent_type.lower() if agent_type else "sales"

    graph = _compiled_graphs.get(normalized)
    if graph is not None:
        return graph

    logger.warning(
        "unknown_agent_type_fallback",
        requested=normalized,
        fallback="sales",
    )
    return _compiled_graphs.get("sales")
