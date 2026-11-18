"""What the Campus agents keep between requests, in Postgres.

There is no Temporal here: a session's stream is a table of events, so the
``feed`` that produces an event and the SSE ``stream`` that reads it may land
on different Cloud Run instances. Each stream has an ``epoch`` (a new one when
an expired session is reopened) and its events a growing ``offset``; an SSE
event id is ``{epoch}:{offset}``, which is what the gateway sends back as
``Last-Event-ID`` to resume where it left off.

``MemoryStore`` has the same contract, for tests and a database-less run.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import asyncpg


@dataclass
class Stream:
    id: str
    epoch: str
    info: dict[str, Any]
    closed: bool
    last_feed_at: datetime


@dataclass
class Event:
    offset: int
    event: str
    data: dict[str, Any]


@dataclass
class Design:
    id: str
    status: str
    proposal: dict[str, Any]
    created_at: datetime
    error: str | None = None


class Store(Protocol):
    async def open_stream(self, stream_id: str, idle: timedelta) -> Stream:
        """The stream, created — or reopened with a new epoch and no events — if
        it does not exist, was closed, or has had no news for ``idle``. Marks
        the feed as just received."""

    async def stream(self, stream_id: str) -> Stream | None: ...

    async def set_info(self, stream_id: str, info: dict[str, Any]) -> None: ...

    async def append(self, stream_id: str, event: str, data: dict[str, Any]) -> int:
        """Adds an event and answers its offset."""

    async def events(self, stream_id: str, epoch: str, from_offset: int, limit: int) -> list[Event]:
        """Events of that epoch from ``from_offset`` on; none if the epoch changed."""

    async def close(self, stream_id: str) -> None: ...

    async def claim_mate(self, stream_id: str, team: str, cooldown: timedelta) -> int | None:
        """The next Companion sequence for ``team``, or ``None`` if it spoke
        less than ``cooldown`` ago. Atomic across instances."""

    async def create_design(self, design_id: str) -> None: ...

    async def design(self, design_id: str) -> Design | None: ...

    async def finish_design(
        self, design_id: str, status: str, proposal: dict[str, Any], error: str | None = None
    ) -> None: ...

    async def approve_design(self, design_id: str, approved: list[str]) -> bool: ...

    async def purge(self, older_than: timedelta) -> dict[str, int]: ...


def _now() -> datetime:
    return datetime.now(UTC)


class PgStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def open_stream(self, stream_id: str, idle: timedelta) -> Stream:
        async with self.pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT * FROM campus_streams WHERE id = $1 FOR UPDATE", stream_id
            )
            stale = row is None or row["closed"] or row["last_feed_at"] < _now() - idle
            if stale:
                await conn.execute("DELETE FROM campus_events WHERE stream_id = $1", stream_id)
                row = await conn.fetchrow(
                    """INSERT INTO campus_streams (id, epoch) VALUES ($1, $2)
                       ON CONFLICT (id) DO UPDATE SET epoch = excluded.epoch, info = '{}',
                         closed = false, next_offset = 0, mate_spoke = '{}', mate_seq = 0,
                         created_at = now(), last_feed_at = now()
                       RETURNING *""",
                    stream_id,
                    uuid.uuid4().hex,
                )
            else:
                row = await conn.fetchrow(
                    "UPDATE campus_streams SET last_feed_at = now() WHERE id = $1 RETURNING *",
                    stream_id,
                )
        return _stream(row)

    async def stream(self, stream_id: str) -> Stream | None:
        row = await self.pool.fetchrow("SELECT * FROM campus_streams WHERE id = $1", stream_id)
        return _stream(row) if row else None

    async def set_info(self, stream_id: str, info: dict[str, Any]) -> None:
        await self.pool.execute(
            "UPDATE campus_streams SET info = $2::jsonb WHERE id = $1", stream_id, json.dumps(info)
        )

    async def append(self, stream_id: str, event: str, data: dict[str, Any]) -> int:
        async with self.pool.acquire() as conn, conn.transaction():
            offset = await conn.fetchval(
                """UPDATE campus_streams SET next_offset = next_offset + 1 WHERE id = $1
                   RETURNING next_offset - 1""",
                stream_id,
            )
            if offset is None:
                raise LookupError(f"no stream {stream_id}")
            await conn.execute(
                """INSERT INTO campus_events (stream_id, "offset", event, data)
                   VALUES ($1, $2, $3, $4::jsonb)""",
                stream_id,
                offset,
                event,
                json.dumps(data, ensure_ascii=False),
            )
        return offset

    async def events(self, stream_id: str, epoch: str, from_offset: int, limit: int) -> list[Event]:
        rows = await self.pool.fetch(
            """SELECT e."offset", e.event, e.data FROM campus_events e
               JOIN campus_streams s ON s.id = e.stream_id
               WHERE e.stream_id = $1 AND s.epoch = $2 AND e."offset" >= $3
               ORDER BY e."offset" LIMIT $4""",
            stream_id,
            epoch,
            from_offset,
            limit,
        )
        return [Event(r["offset"], r["event"], json.loads(r["data"])) for r in rows]

    async def close(self, stream_id: str) -> None:
        await self.pool.execute("UPDATE campus_streams SET closed = true WHERE id = $1", stream_id)

    async def claim_mate(self, stream_id: str, team: str, cooldown: timedelta) -> int | None:
        return await self.pool.fetchval(
            """UPDATE campus_streams
               SET mate_spoke = jsonb_set(mate_spoke, ARRAY[$2::text], to_jsonb(now()::text)),
                   mate_seq = mate_seq + 1
               WHERE id = $1 AND (mate_spoke ->> $2::text IS NULL
                 OR (mate_spoke ->> $2::text)::timestamptz < now() - $3::interval)
               RETURNING mate_seq""",
            stream_id,
            team,
            cooldown,
        )

    async def create_design(self, design_id: str) -> None:
        await self.pool.execute("INSERT INTO campus_designs (id) VALUES ($1)", design_id)

    async def design(self, design_id: str) -> Design | None:
        row = await self.pool.fetchrow("SELECT * FROM campus_designs WHERE id = $1", design_id)
        if row is None:
            return None
        return Design(
            row["id"], row["status"], json.loads(row["proposal"]), row["created_at"], row["error"]
        )

    async def finish_design(
        self, design_id: str, status: str, proposal: dict[str, Any], error: str | None = None
    ) -> None:
        await self.pool.execute(
            """UPDATE campus_designs SET status = $2, proposal = $3::jsonb, error = $4
               WHERE id = $1""",
            design_id,
            status,
            json.dumps(proposal, ensure_ascii=False),
            error,
        )

    async def approve_design(self, design_id: str, approved: list[str]) -> bool:
        result = await self.pool.execute(
            "UPDATE campus_designs SET approved = $2::jsonb, status = 'done' WHERE id = $1",
            design_id,
            json.dumps(approved),
        )
        return result.endswith(" 1")

    async def purge(self, older_than: timedelta) -> dict[str, int]:
        before = _now() - older_than
        streams = await self.pool.execute(
            "DELETE FROM campus_streams WHERE last_feed_at < $1", before
        )
        designs = await self.pool.execute("DELETE FROM campus_designs WHERE created_at < $1", before)
        return {"streams": int(streams.split()[-1]), "designs": int(designs.split()[-1])}


def _stream(row: asyncpg.Record) -> Stream:
    return Stream(
        row["id"], row["epoch"], json.loads(row["info"]), row["closed"], row["last_feed_at"]
    )


@dataclass
class _MemStream:
    stream: Stream
    events: list[Event] = field(default_factory=list)
    mate_spoke: dict[str, datetime] = field(default_factory=dict)
    mate_seq: int = 0


class MemoryStore:
    """The same contract in one process. Tests; never Cloud Run."""

    def __init__(self) -> None:
        self.streams: dict[str, _MemStream] = {}
        self.designs: dict[str, Design] = {}
        self.approved: dict[str, list[str]] = {}

    async def open_stream(self, stream_id: str, idle: timedelta) -> Stream:
        current = self.streams.get(stream_id)
        if current is None or current.stream.closed or current.stream.last_feed_at < _now() - idle:
            current = _MemStream(Stream(stream_id, uuid.uuid4().hex, {}, False, _now()))
            self.streams[stream_id] = current
        current.stream.last_feed_at = _now()
        return current.stream

    async def stream(self, stream_id: str) -> Stream | None:
        current = self.streams.get(stream_id)
        return current.stream if current else None

    async def set_info(self, stream_id: str, info: dict[str, Any]) -> None:
        self.streams[stream_id].stream.info = info

    async def append(self, stream_id: str, event: str, data: dict[str, Any]) -> int:
        current = self.streams[stream_id]
        offset = len(current.events)
        current.events.append(Event(offset, event, json.loads(json.dumps(data))))
        return offset

    async def events(self, stream_id: str, epoch: str, from_offset: int, limit: int) -> list[Event]:
        current = self.streams.get(stream_id)
        if current is None or current.stream.epoch != epoch:
            return []
        return current.events[from_offset : from_offset + limit]

    async def close(self, stream_id: str) -> None:
        self.streams[stream_id].stream.closed = True

    async def claim_mate(self, stream_id: str, team: str, cooldown: timedelta) -> int | None:
        current = self.streams[stream_id]
        last = current.mate_spoke.get(team)
        if last is not None and _now() - last < cooldown:
            return None
        current.mate_spoke[team] = _now()
        current.mate_seq += 1
        return current.mate_seq

    async def create_design(self, design_id: str) -> None:
        self.designs[design_id] = Design(design_id, "working", {}, _now())

    async def design(self, design_id: str) -> Design | None:
        return self.designs.get(design_id)

    async def finish_design(
        self, design_id: str, status: str, proposal: dict[str, Any], error: str | None = None
    ) -> None:
        design = self.designs[design_id]
        design.status, design.proposal, design.error = status, proposal, error

    async def approve_design(self, design_id: str, approved: list[str]) -> bool:
        design = self.designs.get(design_id)
        if design is None:
            return False
        design.status = "done"
        self.approved[design_id] = approved
        return True

    async def purge(self, older_than: timedelta) -> dict[str, int]:
        before = _now() - older_than
        streams = [k for k, s in self.streams.items() if s.stream.last_feed_at < before]
        designs = [k for k, d in self.designs.items() if d.created_at < before]
        for key in streams:
            del self.streams[key]
        for key in designs:
            del self.designs[key]
        return {"streams": len(streams), "designs": len(designs)}
