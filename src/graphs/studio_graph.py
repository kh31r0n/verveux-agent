"""LangGraph Studio entrypoint — used by `langgraph dev` only.

Compiles the main graph with an in-memory checkpointer so LangGraph Studio
can visualise and interactively run the graph without requiring Postgres
checkpointing.  The DB pool IS initialised so agent nodes that query
Postgres (RAG, audit log) work normally when DATABASE_URL is provided.
"""

import asyncio

from src.graphs.main_graph import build_graph
from src.db.postgres import init_pool, run_migrations


def _boot_db() -> None:
    """Synchronously initialise the asyncpg pool before the graph is compiled."""

    async def _setup() -> None:
        pool = await init_pool()
        await run_migrations(pool)

    try:
        asyncio.run(_setup())
    except Exception as exc:
        # Studio can still visualise graph structure without a live DB.
        import sys
        print(f"[studio_graph] DB init skipped: {exc}", file=sys.stderr)


_boot_db()

# LangGraph API manages its own persistence — do NOT pass a checkpointer here.
graph = build_graph(None)  # type: ignore[arg-type]
