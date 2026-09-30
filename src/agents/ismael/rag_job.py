"""The background half of an ismael answer.

A chat turn has 60 s before the backend gives up on it, and Brain can need
minutes: up to ~2 to boot from stopped, and a real question has taken more than
180 s to answer. So the turn only replies with a holding message and spawns
this job, which waits for Brain, asks, and delivers the answer out of band
through ``POST /internal/conversations/:id/agent-messages``. That endpoint
stores the message, pushes it to the Moodle bubble over the widget's SSE, and
bills the job's LLM usage — idempotently, keyed on the job id.

One job per conversation at a time. Whether one is running is recorded in the
shared LangGraph store (Postgres), so it holds across the agent's Cloud Run
instances; the in-process registry is only the fallback for when the store is
unavailable (tests, local runs without it).

If the instance dies mid-job the answer is lost; the store entry expires after
`JOB_TTL_SECONDS` and the student can ask again. Acceptable at this volume.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog

from ...config import settings
from ...providers.registry import get_provider, resolve_model
from ...services import brain
from ...usage import make_usage_record
from .. import backend_client
from .prompts import GENERAL_PROMPT
from .texts import text

logger = structlog.get_logger(__name__)

# Longest a job can legitimately run, plus slack. Past it a "running" entry is
# treated as dead (the instance that owned it was recycled).
JOB_TTL_SECONDS = (
    settings.brain_boot_timeout_seconds + settings.brain_answer_timeout_seconds + 120
)
MAX_REFERENCES = 5

# Strong refs: asyncio keeps only weak references to tasks.
_tasks: set[asyncio.Task] = set()
# conversation_id → (job_id, question, started monotonic) — fallback registry.
_local_jobs: dict[str, tuple[str, str, float]] = {}


@dataclass
class RagJob:
    tenant_id: str
    conversation_id: str
    question: str
    language: str
    persona: str
    # The turn's `configurable` minus everything the job does not need: LLM
    # provider, model, the provider's credentials and the prompt payloads. Kept
    # in memory only — never in graph state, never logged.
    llm_config: dict
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)


def get_store_or_none():
    # Imported at call time: the registry imports this module's graph.
    from ...graphs.registry import get_store_or_none as _get

    return _get()


def _namespace(tenant_id: str) -> tuple[str, ...]:
    return ("ismael_jobs", tenant_id or "_")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def running_job(tenant_id: str, conversation_id: str) -> dict | None:
    """The conversation's live job as ``{job_id, question}``, or None."""
    store = get_store_or_none()
    if store is not None:
        try:
            item = await store.aget(_namespace(tenant_id), conversation_id)
        except Exception as exc:  # noqa: BLE001 — fall back to the local registry
            logger.info("ismael_job_store_read_failed", error=str(exc))
        else:
            value = getattr(item, "value", None) if item else None
            if not value or value.get("status") != "running":
                return None
            started = datetime.fromisoformat(value["started_at"])
            age = (datetime.now(timezone.utc) - started).total_seconds()
            if age > JOB_TTL_SECONDS:
                return None
            return {"job_id": value.get("job_id"), "question": value.get("question", "")}

    local = _local_jobs.get(conversation_id)
    if local and time.monotonic() - local[2] <= JOB_TTL_SECONDS:
        return {"job_id": local[0], "question": local[1]}
    return None


async def _record(job: RagJob, status: str) -> None:
    if status == "running":
        _local_jobs[job.conversation_id] = (job.job_id, job.question, time.monotonic())
    else:
        current = _local_jobs.get(job.conversation_id)
        if current and current[0] == job.job_id:
            _local_jobs.pop(job.conversation_id, None)

    store = get_store_or_none()
    if store is None:
        return
    value = {"job_id": job.job_id, "status": status, "question": job.question}
    value["started_at" if status == "running" else "finished_at"] = _now_iso()
    try:
        if status != "running":
            # Keep started_at so a stale read still parses.
            item = await store.aget(_namespace(job.tenant_id), job.conversation_id)
            previous = getattr(item, "value", None) if item else None
            if previous and previous.get("job_id") != job.job_id:
                return  # a newer job owns the slot
            value["started_at"] = (previous or {}).get("started_at") or _now_iso()
        await store.aput(_namespace(job.tenant_id), job.conversation_id, value)
    except Exception as exc:  # noqa: BLE001 — the local registry still holds
        logger.info("ismael_job_store_write_failed", error=str(exc), status=status)


def _reference_items(answer: dict, lang: str) -> list[tuple[str, str | None]]:
    """(title, detail) per cited source: once per (title, page), in citation order."""
    evidence = {e.get("chunk_id"): e for e in answer.get("evidence") or [] if isinstance(e, dict)}
    items: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str]] = set()
    for citation in answer.get("citations") or []:
        if not isinstance(citation, dict):
            continue
        ev = evidence.get(citation.get("chunk_id")) or {}
        title = (ev.get("title") or citation.get("section_title") or "").strip()
        if not title:
            continue
        page = citation.get("page") or ev.get("page")
        where = (citation.get("section_title") or "").strip()
        detail = f"{text('page', lang)} {page}" if page else where
        key = (title, str(detail))
        if key in seen:
            continue
        seen.add(key)
        items.append((title, detail if detail and detail != title else None))
        if len(items) >= MAX_REFERENCES:
            break
    return items


def format_answer(answer: dict, lang: str) -> str:
    """Brain's answer text plus a References block built from its citations.

    Brain verifies every citation server-side, so the text is sent as written —
    an LLM rewrite could detach a claim from the source that supports it.
    """
    body = (answer.get("text") or "").strip()
    items = _reference_items(answer, lang)
    if not items:
        return body
    lines = [f"- {title}" + (f", {detail}" if detail else "") for title, detail in items]
    return f"{body}\n\n{text('references', lang)}\n" + "\n".join(lines)


def answer_references(answer: dict, lang: str) -> dict | None:
    """The same References block as data, for the widget's citation list.

    The backend keeps it only while ``format_answer``'s text ends with exactly
    this block (``referencesSuffix`` in src/messages/references.ts) — both are
    built from ``_reference_items``, so they cannot drift apart.
    """
    items = _reference_items(answer, lang)
    if not items or not (answer.get("text") or "").strip():
        return None
    return {
        "heading": text("references", lang),
        "items": [{"title": title, "detail": detail} for title, detail in items],
    }


async def _general_answer(job: RagJob) -> tuple[str, list]:
    """Concise general-knowledge answer for a question the library does not cover."""
    config = {"configurable": job.llm_config}
    prompts = job.llm_config.get("prompts") or {}
    payload = prompts.get("THEOLOGY_GENERAL") or {}
    template = (payload.get("content") if isinstance(payload, dict) else "") or GENERAL_PROMPT
    language_rule = "Always respond in English." if job.language == "en" else "Responde siempre en español."
    system = template.replace("{persona}", job.persona).replace("{language_rule}", language_rule)

    provider = get_provider(config)
    model = resolve_model(config)
    reply = await provider.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": job.question}],
        model,
    )
    usage = [make_usage_record(node="ismael_general_answer", provider=provider, model=model)]
    reply = (reply or "").strip()
    if not reply:
        raise brain.BrainError("general answer came back empty")
    return f"{reply}\n\n{text('general_notice', job.language)}", usage


async def _compose(job: RagJob) -> tuple[str, list, str, dict | None]:
    """(message, usage, outcome, references) for the job — never raises."""
    try:
        await brain.ensure_started()
        boot_deadline = time.monotonic() + settings.brain_boot_timeout_seconds
        if not await brain.wait_healthy(boot_deadline):
            return text("failed", job.language), [], "boot_timeout", None

        question_id = await brain.ask(job.question)
        answer_deadline = time.monotonic() + settings.brain_answer_timeout_seconds
        outcome = await brain.wait_answer(question_id, answer_deadline)
        if outcome is None:
            return text("failed", job.language), [], "answer_timeout", None
        if outcome.get("state") != "done" or not isinstance(outcome.get("answer"), dict):
            logger.warning("ismael_brain_failed", error=outcome.get("error"))
            return text("failed", job.language), [], "brain_failed", None

        answer = outcome["answer"]
        if answer.get("state") == "answered" and (answer.get("text") or "").strip():
            return (
                format_answer(answer, job.language),
                [],
                "answered",
                answer_references(answer, job.language),
            )

        # insufficient_evidence / off_corpus: a short general answer, labelled.
        message, usage = await _general_answer(job)
        return message, usage, f"general:{answer.get('state')}", None
    except Exception as exc:  # noqa: BLE001 — the student still gets a reply
        logger.error("ismael_job_error", job_id=job.job_id, error=str(exc))
        return text("failed", job.language), [], "error", None


async def run(job: RagJob) -> None:
    started = time.monotonic()
    try:
        message, usage, outcome, references = await _compose(job)
        try:
            await backend_client.post_agent_message(
                job.conversation_id,
                job_id=job.job_id,
                text=message,
                turn_usage=usage,
                references=references,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("ismael_delivery_failed", job_id=job.job_id, error=str(exc))
            outcome = f"{outcome}:undelivered"
        logger.info(
            "ismael_job_finished",
            job_id=job.job_id,
            conversation_id=job.conversation_id,
            outcome=outcome,
            seconds=round(time.monotonic() - started, 1),
        )
    finally:
        await _record(job, "finished")


async def spawn(job: RagJob) -> None:
    """Mark the conversation busy and start the job in the background."""
    await _record(job, "running")
    task = asyncio.create_task(run(job))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def start_host_in_background() -> None:
    """Fire the EC2 start without waiting — the survey buys the boot time."""
    task = asyncio.create_task(brain.ensure_started())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def job_llm_config(config: dict) -> dict:
    """The slice of a turn's configurable the job needs (see `RagJob.llm_config`)."""
    cfg = config.get("configurable") or {}
    keep = (
        "llm_provider",
        "llm_model",
        "openai_api_key",
        "anthropic_api_key",
        "gemini_credentials",
        "gemini_project_id",
        "gemini_location",
    )
    sliced = {k: cfg[k] for k in keep if k in cfg}
    prompts = cfg.get("prompts") or {}
    if "THEOLOGY_GENERAL" in prompts:
        sliced["prompts"] = {"THEOLOGY_GENERAL": prompts["THEOLOGY_GENERAL"]}
    return sliced
