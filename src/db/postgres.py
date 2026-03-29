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
    """Apply init.sql migrations for application tables."""
    import pathlib

    sql_path = pathlib.Path(__file__).parent.parent.parent / "migrations" / "init.sql"
    sql = sql_path.read_text(encoding="utf-8")
    async with pool.acquire() as conn:
        await conn.execute(sql)
    logger.info("migrations_applied", file=str(sql_path))
