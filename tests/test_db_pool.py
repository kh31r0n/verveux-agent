"""The asyncpg pool is sized from settings, not hard-coded (2026-10-16: the
shared db-f1-micro ran out of connection slots)."""

from unittest.mock import AsyncMock, patch

from src.config import settings
import pytest

from src.db import postgres


@pytest.fixture(autouse=True)
def _reset_pool():
    yield
    postgres._pool = None  # init_pool sets the module global


async def test_pool_bounds_come_from_settings():
    create = AsyncMock(return_value=object())
    with (
        patch.object(settings, "database_pool_min", 1),
        patch.object(settings, "database_pool_max", 4),
        patch("src.db.postgres.asyncpg.create_pool", create),
    ):
        await postgres.init_pool()
    kwargs = create.await_args.kwargs
    assert kwargs["min_size"] == 1
    assert kwargs["max_size"] == 4


async def test_min_never_exceeds_max():
    create = AsyncMock(return_value=object())
    with (
        patch.object(settings, "database_pool_min", 9),
        patch.object(settings, "database_pool_max", 3),
        patch("src.db.postgres.asyncpg.create_pool", create),
    ):
        await postgres.init_pool()
    kwargs = create.await_args.kwargs
    assert kwargs["min_size"] == 3
    assert kwargs["max_size"] == 3
