"""The Campus routes, under ``/campus``: what the board game's gateway (in
rocky-nest-backend, ``src/campus/realtime/agents.bridge.ts``) calls.

The contract is the one vervux-campus's agents API had, path for path, so the
gateway only points ``CAMPUS_AGENTS_URL`` at ``…/campus``:

- ``POST /sessions/:id/feed``: something the Game Core told about a session.
- ``GET /sessions/:id/stream``: what the agents propose, as SSE. The event id
  is ``{epoch}:{offset}``; ``Last-Event-ID`` resumes; ``end`` closes it.
- ``GET /sessions/:id/audio/:file``: a line of the Game Master's voice.
- ``POST /designs``, ``GET /designs/:id``, ``POST /designs/:id/approval``: the
  Designer.
- ``POST /sessions/:id/narration-demo`` and its ``/stream``: phase 0's test.

Every route needs the service credential (``verify_service_caller``) like the
rest of this service; the gateway sends a Google ID token.
"""

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Header, HTTPException, Path, UploadFile
from fastapi.responses import Response
from sse_starlette import EventSourceResponse, ServerSentEvent

from ..auth.service_auth import verify_service_caller
from . import design
from .config import campus_settings
from .session import demo_stream, get_campus, session_stream

SessionId = Annotated[str, Path(pattern=r"^[A-Z0-9]{4,12}$")]
DesignId = Annotated[str, Path(pattern=r"^[a-f0-9]{32}$")]
LastEventId = Annotated[str | None, Header()]

# How often a reader looks for events another instance wrote.
POLL_SECONDS = 0.5
BATCH = 200

router = APIRouter(prefix="/campus", tags=["campus"], dependencies=[Depends(verify_service_caller)])


@router.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok", "llm": campus_settings.llm}


@router.post("/sessions/{session_id}/feed", status_code=202)
async def feed(session_id: SessionId, body: dict[str, Any]) -> dict[str, str]:
    if not isinstance(body.get("feed"), str):
        raise HTTPException(422, "Falta el tipo de la novedad")
    await get_campus().feed(session_id, body)
    return {"status": "accepted"}


@router.get("/sessions/{session_id}/stream")
async def stream(session_id: SessionId, last_event_id: LastEventId = None) -> EventSourceResponse:
    return await _stream(session_stream(session_id), last_event_id)


@router.get("/sessions/{session_id}/audio/{name}")
async def audio(
    session_id: SessionId, name: Annotated[str, Path(pattern=r"^[0-9]{1,6}\.wav$")]
) -> Response:
    data = await get_campus().files.get(f"audio/{session_id}/{name}")
    if data is None:
        raise HTTPException(404, "No existe ese audio")
    return Response(data, media_type="audio/wav")


@router.post("/designs", status_code=202)
async def start_design(
    files: list[UploadFile],
    topic: Annotated[str, Form(max_length=80)] = "",
    lang: Annotated[str, Form(pattern="^(es|en|pt)$")] = "es",
    max_questions: Annotated[int, Form(ge=1, le=30)] = 10,
) -> dict[str, str]:
    if not 1 <= len(files) <= design.MAX_FILES:
        raise HTTPException(422, f"Se admiten de 1 a {design.MAX_FILES} archivos")
    campus = get_campus()
    design_id = uuid.uuid4().hex
    prefix = design.design_prefix(design_id)
    for index, upload in enumerate(files):
        # The name comes from the host's browser: only what is harmless is kept.
        name = re.sub(r"[^\w.\- ]", "_", upload.filename or "archivo")[-80:]
        # Past the maximum size the Designer rejects it, file by file.
        data = await upload.read(design.MAX_FILE_BYTES + 1)
        await campus.files.put(
            f"{prefix}/{index + 1:02d}-{name}", data, upload.content_type or "application/octet-stream"
        )
    params = {"lang": lang, "topic": topic.strip(), "max_questions": max_questions}
    await campus.start_design(design_id, params)
    return {"design_id": design_id}


@router.get("/designs/{design_id}")
async def design_state(design_id: DesignId) -> dict[str, Any]:
    state = await get_campus().design_state(design_id)
    if state is None:
        raise HTTPException(404, "No existe ese diseño")
    return state


@router.post("/designs/{design_id}/approval")
async def approve_design(design_id: DesignId, body: dict[str, Any]) -> dict[str, str]:
    approved = body.get("approved")
    if not isinstance(approved, list) or not all(isinstance(q, str) for q in approved):
        raise HTTPException(422, "Falta la lista de preguntas aprobadas")
    if not await get_campus().store.approve_design(design_id, approved):
        raise HTTPException(404, "No existe ese diseño")
    return {"status": "approved"}


@router.post("/sessions/{session_id}/narration-demo", status_code=202)
async def start_narration_demo(session_id: SessionId) -> dict[str, str]:
    if not await get_campus().narration_demo(session_id):
        raise HTTPException(409, "Ya hay una narración en curso para esta partida")
    return {"workflow_id": demo_stream(session_id)}


@router.get("/sessions/{session_id}/narration-demo/stream")
async def narration_demo_stream(
    session_id: SessionId, last_event_id: LastEventId = None
) -> EventSourceResponse:
    return await _stream(demo_stream(session_id), last_event_id)


async def _stream(stream_id: str, last_event_id: str | None) -> EventSourceResponse:
    """A stream over SSE. Whoever reconnects sends ``Last-Event-ID`` and gets
    only what it missed; if the epoch changed, everything from the start."""
    campus = get_campus()
    current = await campus.store.stream(stream_id)
    if current is None:
        raise HTTPException(404, "No hay un flujo para esta partida")
    epoch = current.epoch
    seen_epoch, _, offset = (last_event_id or "").partition(":")
    next_offset = int(offset) + 1 if seen_epoch == epoch and offset.isdigit() else 0

    async def events() -> AsyncIterator[ServerSentEvent]:
        nonlocal next_offset
        while True:
            # Taken before reading, so an event written meanwhile is not missed.
            news = campus.news(stream_id)
            batch = await campus.store.events(stream_id, epoch, next_offset, BATCH)
            for item in batch:
                next_offset = item.offset + 1
                yield ServerSentEvent(
                    id=f"{epoch}:{item.offset}",
                    event=item.event,
                    data=json.dumps(item.data, ensure_ascii=False),
                )
            if len(batch) == BATCH:
                continue
            now = await campus.store.stream(stream_id)
            if now is None or now.epoch != epoch:
                break
            idle = timedelta(hours=campus_settings.idle_hours)
            if now.last_feed_at < datetime.now(UTC) - idle:
                # Abandoned: nobody told anything about it for too long.
                break
            if now.closed:
                # Closed: whatever was written before closing is delivered first.
                if not await campus.store.events(stream_id, epoch, next_offset, 1):
                    break
                continue
            try:
                await asyncio.wait_for(news.wait(), timeout=POLL_SECONDS)
            except TimeoutError:
                pass
        yield ServerSentEvent(event="end", data="{}")

    return EventSourceResponse(events(), ping=15)
