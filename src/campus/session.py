"""The Campus runtime: what Temporal did in vervux-campus, on this service's
own means.

- A session's ``feed`` is handled in the request that brings it: ``started``
  is stored synchronously, everything else starts agents as background tasks
  and the request answers 202 at once. This service runs with CPU always
  allocated (``cpu_idle = false``), so the tasks keep running.
- What the agents propose is appended to the session's stream in Postgres
  (``store.py``) and any instance serves it over SSE.
- Every agent has a deadline per attempt and a number of attempts. One that
  fails never stops the others nor the game: the game goes on with its script.
- ``finished`` runs the Grader; when the report is out, the stream ends.

What is lost compared to Temporal: work in flight when an instance dies is not
resumed. A live agent's late answer is useless anyway; the report would be
missing, and the host still has the game's fixed grades in the Game Core.
"""

import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from .config import campus_settings
from .design import designer
from .files import Files
from .live import AGENTS, EVALUATOR, GAME_MASTER, GRADER, TEAMMATE, LiveState
from .store import Store

logger = structlog.get_logger(__name__)

# Words a player calls the Companion with.
MATE_CALLS = ("@ia", "ia,", " ia ", "ia?", "compañer", "companion", "companheir")

NARRATION_DEMO = [
    "¡Bienvenidos ",
    "a la partida! ",
    "Soy su Game Master. ",
    "Preparen sus dados: ",
    "la primera ronda ",
    "está por comenzar.",
]


def session_stream(session_id: str) -> str:
    return f"session:{session_id}"


def demo_stream(session_id: str) -> str:
    return f"demo:{session_id}"


def calls_mate(text: str) -> bool:
    """The Companion speaks when called or when someone asks a question."""
    low = f" {text.lower()} "
    return "?" in text or any(call in low for call in MATE_CALLS)


class Campus:
    def __init__(self, store: Store, files: Files) -> None:
        self.store = store
        self.files = files
        self._tasks: set[asyncio.Task[Any]] = set()
        # The agents in flight of each stream, on this instance.
        self._working: dict[str, set[asyncio.Task[Any]]] = {}
        # Wakes the SSE readers of this instance at once; readers of other
        # instances find the event on their next poll.
        self._news: dict[str, asyncio.Event] = {}

    # --- streams -------------------------------------------------------------

    def news(self, stream_id: str) -> asyncio.Event:
        return self._news.setdefault(stream_id, asyncio.Event())

    async def emit(self, stream_id: str, event: dict[str, Any]) -> None:
        data = dict(event)
        name = data.pop("event")
        await self.store.append(stream_id, name, data)
        signal = self._news.pop(stream_id, None)
        if signal:
            signal.set()

    async def end(self, stream_id: str) -> None:
        await self.store.close(stream_id)
        signal = self._news.pop(stream_id, None)
        if signal:
            signal.set()

    def spawn(self, work: Awaitable[Any], name: str, stream_id: str | None = None) -> None:
        task = asyncio.ensure_future(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if stream_id is not None:
            working = self._working.setdefault(stream_id, set())
            working.add(task)
            task.add_done_callback(working.discard)

    async def drain(self) -> None:
        """Waits for every task in flight. Tests, and an orderly shutdown."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # --- sessions ------------------------------------------------------------

    async def feed(self, session_id: str, feed: dict[str, Any]) -> None:
        stream_id = session_stream(session_id)
        stream = await self.store.open_stream(
            stream_id, timedelta(hours=campus_settings.idle_hours)
        )
        if feed.get("feed") == "started":
            await self.store.set_info(stream_id, feed)
            return
        work = await self._work(session_id, stream.info, feed)
        if feed.get("feed") == "finished":
            self.spawn(self._finish(stream_id, work), "campus_grader")
            return
        # Each one on its own: a slow evaluation does not delay the next narration.
        for name, state in work:
            self.spawn(self._run(name, state), f"campus_{name}", stream_id)

    async def _work(
        self, session_id: str, session: dict[str, Any], feed: dict[str, Any]
    ) -> list[tuple[str, LiveState]]:
        """Which agents handle this news."""
        ai = session.get("ai", {})
        groups = session.get("mode") == "groups"
        companion = groups and ai.get("companion", "off") != "off"
        stream_id = session_stream(session_id)

        def state(item: dict[str, Any]) -> LiveState:
            return LiveState(
                session_id=session_id,
                session=session,
                feed=item,
                emit=lambda event: self.emit(stream_id, event),
                files=self.files,
            )

        kind = feed.get("feed")
        if kind == "moment":
            work = [(GAME_MASTER, state(feed))]
            line = feed.get("line", {})
            # When an activity starts, the Companion greets each group.
            if companion and line.get("line") == "activity_intro":
                nudge = {
                    "feed": "nudge",
                    "moment": feed["moment"],
                    "activity": line.get("kind"),
                    "teams": [team["team_id"] for team in feed.get("teams", [])],
                }
                work.append((TEAMMATE, state(nudge)))
            return work
        if kind in ("concepts", "guess", "open_answers"):
            return [(EVALUATOR, state(feed))]
        if kind == "chat":
            if not companion or not calls_mate(feed.get("text", "")):
                return []
            seq = await self.store.claim_mate(
                stream_id,
                feed.get("team_id", ""),
                timedelta(seconds=campus_settings.mate_cooldown_seconds),
            )
            if seq is None:
                return []
            return [(TEAMMATE, state({**feed, "seq": seq}))]
        if kind == "finished":
            return [(GRADER, state(feed))]
        return []

    async def _run(self, name: str, state: LiveState) -> None:
        agent = AGENTS[name]
        for attempt in range(1, agent.attempts + 1):
            try:
                await asyncio.wait_for(agent.run(state), timeout=agent.seconds)
                return
            except Exception as error:  # noqa: BLE001 — a failed agent never stops the game
                logger.warning(
                    "campus_agent_failed",
                    agent=name,
                    session_id=state.session_id,
                    attempt=attempt,
                    error=f"{type(error).__name__}: {error}"[:300],
                )

    async def _finish(self, stream_id: str, work: list[tuple[str, LiveState]]) -> None:
        # The report is the last thing: what is still running on this instance
        # is awaited, then the report, then the stream ends. An agent running
        # on another instance may finish after the end; nobody needs it then.
        try:
            await asyncio.gather(*self._working.pop(stream_id, set()), return_exceptions=True)
            for name, state in work:
                await self._run(name, state)
        finally:
            await self.end(stream_id)

    # --- the Designer --------------------------------------------------------

    async def start_design(self, design_id: str, params: dict[str, Any]) -> None:
        await self.store.create_design(design_id)
        self.spawn(self._design(design_id, params), "campus_designer")

    async def _design(self, design_id: str, params: dict[str, Any]) -> None:
        try:
            result = await designer.ainvoke(
                {**params, "design_id": design_id},
                config={"configurable": {"files": self.files}},
            )
        except Exception as error:  # noqa: BLE001 — the panel must know it failed
            logger.warning("campus_design_failed", design_id=design_id, error=str(error)[:300])
            await self.store.finish_design(
                design_id,
                "failed",
                {"files": [], "questions": [], "discarded": 0},
                f"{type(error).__name__}: {error}"[:300],
            )
            return
        proposal = {
            "files": result.get("files", []),
            "questions": result.get("questions", []),
            "discarded": result.get("discarded", 0),
        }
        # No generated question enters without the host's approval.
        await self.store.finish_design(design_id, "waiting_approval", proposal)

    async def design_state(self, design_id: str) -> dict[str, Any] | None:
        design = await self.store.design(design_id)
        if design is None:
            return None
        status = design.status
        late = timedelta(minutes=campus_settings.design_timeout_minutes)
        if status == "working" and design.created_at < _now() - late:
            # Its instance died mid-work: nobody will finish it.
            status = "failed"
        proposal = design.proposal or {"files": [], "questions": [], "discarded": 0}
        state = {"status": status, **proposal}
        if design.error:
            state["error"] = design.error
        return state

    # --- phase 0 demo --------------------------------------------------------

    async def narration_demo(self, session_id: str, pause: float = 0.12) -> bool:
        """A fixed narration in fragments, to test the transport without a
        model. ``False`` if one is already running for that session."""
        stream_id = demo_stream(session_id)
        current = await self.store.stream(stream_id)
        if current is not None and not current.closed:
            return False
        await self.store.open_stream(stream_id, timedelta(0))
        self.spawn(self._narrate_demo(stream_id, pause), "campus_narration_demo")
        return True

    async def _narrate_demo(self, stream_id: str, pause: float) -> None:
        text = ""
        try:
            for fragment in NARRATION_DEMO:
                await self.emit(stream_id, {"event": "gm.narration.delta", "text": fragment})
                text += fragment
                await asyncio.sleep(pause)
            await self.emit(stream_id, {"event": "gm.narration.done", "text": text})
        finally:
            await self.end(stream_id)

    # --- retention -----------------------------------------------------------

    async def purge_forever(self, every: timedelta = timedelta(hours=6)) -> None:
        while True:
            try:
                removed = await self.store.purge(timedelta(days=campus_settings.retention_days))
                if any(removed.values()):
                    logger.info("campus_retention", **removed)
            except Exception as error:  # noqa: BLE001 — retention never stops the service
                logger.warning("campus_retention_failed", error=str(error))
            await asyncio.sleep(every.total_seconds())


def _now() -> datetime:
    return datetime.now(UTC)


_campus: Campus | None = None


def configure(campus: Campus) -> None:
    global _campus
    _campus = campus


def get_campus() -> Campus:
    if _campus is None:
        raise RuntimeError("Campus is not configured")
    return _campus

