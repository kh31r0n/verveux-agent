"""The Campus store against a real Postgres: the schema of migrations/init.sql,
offsets and epochs, the Companion's cooldown claimed atomically by concurrent
instances, designs and retention.

Runs only with ``CAMPUS_TEST_DATABASE_URL`` (a throwaway database), e.g.

    docker run -d --rm -p 5439:5432 -e POSTGRES_PASSWORD=t postgres:16-alpine
    CAMPUS_TEST_DATABASE_URL=postgresql://postgres:t@localhost:5439/postgres \\
      uv run pytest tests/test_campus_store_pg.py
"""

import asyncio
import os
import pathlib
import re
from datetime import timedelta

import asyncpg
import pytest

from src.campus.store import PgStore

URL = os.environ.get("CAMPUS_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="CAMPUS_TEST_DATABASE_URL is not set")


@pytest.fixture
async def store():
    pool = await asyncpg.create_pool(URL, min_size=1, max_size=5)
    sql = (pathlib.Path(__file__).parent.parent / "migrations" / "init.sql").read_text()
    campus = sql[sql.index("-- ── Campus") :]
    for statement in (s.strip() for s in re.split(r";\s*\n", campus) if s.strip()):
        await pool.execute(statement)
    await pool.execute("TRUNCATE campus_streams, campus_designs CASCADE")
    yield PgStore(pool)
    await pool.close()


async def test_events_have_growing_offsets_within_an_epoch(store: PgStore) -> None:
    stream = await store.open_stream("session:A1", timedelta(hours=2))
    assert [await store.append("session:A1", "command", {"n": n}) for n in range(3)] == [0, 1, 2]
    events = await store.events("session:A1", stream.epoch, 1, 10)
    assert [(e.offset, e.data) for e in events] == [(1, {"n": 1}), (2, {"n": 2})]
    # Another epoch reads nothing.
    assert await store.events("session:A1", "otra", 0, 10) == []
    # Still open and fresh: same epoch.
    assert (await store.open_stream("session:A1", timedelta(hours=2))).epoch == stream.epoch


async def test_a_closed_or_idle_stream_reopens_empty_with_a_new_epoch(store: PgStore) -> None:
    first = await store.open_stream("session:B1", timedelta(hours=2))
    await store.set_info("session:B1", {"lang": "es"})
    await store.append("session:B1", "command", {})
    await store.close("session:B1")
    second = await store.open_stream("session:B1", timedelta(hours=2))
    assert second.epoch != first.epoch and not second.closed and second.info == {}
    assert await store.events("session:B1", second.epoch, 0, 10) == []
    assert await store.append("session:B1", "command", {}) == 0
    # Idle for longer than allowed counts as over.
    third = await store.open_stream("session:B1", timedelta(0))
    assert third.epoch != second.epoch


async def test_concurrent_appends_never_share_an_offset(store: PgStore) -> None:
    await store.open_stream("session:C1", timedelta(hours=2))
    offsets = await asyncio.gather(*(store.append("session:C1", "e", {}) for _ in range(20)))
    assert sorted(offsets) == list(range(20))


async def test_the_companion_cooldown_is_claimed_once_across_instances(store: PgStore) -> None:
    await store.open_stream("session:D1", timedelta(hours=2))
    claims = await asyncio.gather(
        *(store.claim_mate("session:D1", "g1", timedelta(seconds=15)) for _ in range(5))
    )
    assert [c for c in claims if c is not None] == [1]
    # Another group has its own cooldown; a zero cooldown lets it speak again.
    assert await store.claim_mate("session:D1", "g2", timedelta(seconds=15)) == 2
    assert await store.claim_mate("session:D1", "g1", timedelta(0)) == 3


async def test_designs_and_retention(store: PgStore) -> None:
    await store.create_design("d" * 32)
    assert (await store.design("d" * 32)).status == "working"
    await store.finish_design("d" * 32, "waiting_approval", {"questions": [{"id": "g1"}]})
    design = await store.design("d" * 32)
    assert (design.status, design.proposal) == ("waiting_approval", {"questions": [{"id": "g1"}]})
    assert await store.approve_design("d" * 32, ["g1"])
    assert not await store.approve_design("e" * 32, [])

    await store.open_stream("session:E1", timedelta(hours=2))
    await store.append("session:E1", "command", {})
    await store.pool.execute(
        "UPDATE campus_streams SET last_feed_at = now() - interval '40 days' WHERE id = 'session:E1'"
    )
    removed = await store.purge(timedelta(days=30))
    assert removed == {"streams": 1, "designs": 0}
    assert await store.pool.fetchval("SELECT count(*) FROM campus_events WHERE stream_id = 'session:E1'") == 0
