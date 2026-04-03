import asyncpg
import structlog

from ..config import settings

logger = structlog.get_logger(__name__)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        raise RuntimeError("Database pool has not been initialised. Call init_pool() at startup.")
    return _pool


async def init_pool() -> asyncpg.Pool:
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    logger.info("db_pool_created", dsn=settings.database_url.split("@")[-1])
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("db_pool_closed")


async def run_migrations(pool: asyncpg.Pool) -> None:
    """Apply init.sql migrations for application tables.

    Statements are executed individually so that a failure in one
    (e.g. the pgvector extension or ivfflat index) does not roll back
    the entire batch — checkpoint tables must always be created.
    """
    import pathlib
    import re

    sql_path = pathlib.Path(__file__).parent.parent.parent / "migrations" / "init.sql"
    sql = sql_path.read_text(encoding="utf-8")

    # Split on semicolons that are followed by a newline (skip empty chunks)
    statements = [s.strip() for s in re.split(r";\s*\n", sql) if s.strip()]

    async with pool.acquire() as conn:
        for stmt in statements:
            try:
                await conn.execute(stmt)
            except Exception as exc:
                logger.warning(
                    "migration_statement_failed",
                    error=str(exc),
                    statement=stmt[:120],
                )

    logger.info("migrations_applied", file=str(sql_path))
