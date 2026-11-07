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

A theology question that arrives while a job runs is queued (at most
`MAX_QUEUED`) instead of dropped: on 2026-10-29 «¿Quién es Oscar Archila?» and
«¿Qué significa ser católico?» got «después puedes hacerme otra» and were never
answered. The queue holds only the question — never credentials — in the same
store, under its own key, so any instance can add to it; the job that finishes
takes the next one and runs it with its own (same-tenant) LLM config.

When Brain retrieved fragments but judged them insufficient — they covered
part of the question, or did not define it in so many words — the general
answer is grounded on them (`_grounded_answer`): on 2026-10-29 the course's own
book «TEOLOGÍA SOCIAL» and a definition of «vitalismo espiritual» were found
and thrown away, and the director got an ungrounded comparison of traditions
instead. Only the fragments the answer says it used become References.

A question that names a book (triage's ``work``) is asked of that document
alone when it matches one in the library's catalog: Brain never searches
titles, so «¿y el libro Las llaves del poder?» came back insufficient_evidence
on 2026-10-15 although the book is in the library.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import structlog

from ...config import settings
from ...providers.registry import get_provider, resolve_model
from ...schemas.ismael import GroundedAnswer
from ...services import brain
from ...usage import make_usage_record
from .. import backend_client
from .common import fold
from .prompts import GENERAL_PROMPT
from .texts import text

logger = structlog.get_logger(__name__)

# Longest a job can legitimately run, plus slack. Past it a "running" entry is
# treated as dead (the instance that owned it was recycled).
JOB_TTL_SECONDS = (
    settings.brain_boot_timeout_seconds + settings.brain_answer_timeout_seconds + 120
)
MAX_REFERENCES = 5
# Questions waiting behind the running job, per conversation.
MAX_QUEUED = 2
# A Spanish answer this long with no accented letter lost its accents
# (gemini-3.5-flash, 2026-10-29: «teologicos», «Catolica», «salvacion»).
_ACCENT_CHECK_MIN_WORDS = 40
_ACCENTED = re.compile(r"[áéíóúñü¿¡ÁÉÍÓÚÑÜ]")
_ACCENT_RETRY_NOTE = (
    "\n- IMPORTANTE: escribe en español correcto, con TODAS las tildes (á, é, í, ó, ú), "
    "la ñ y los signos de apertura (¿ ¡). Una respuesta sin tildes no sirve."
)
# Fragments handed to a grounded general answer, and how much of each.
MAX_GROUNDING_FRAGMENTS = 5
MAX_FRAGMENT_CHARS = 1500
GROUNDED_TIMEOUT_SECONDS = 45.0
GROUNDING_RULE = """

La búsqueda en la biblioteca de la institución encontró los fragmentos de abajo, pero no bastan para responder toda la pregunta{reason}. Úsalos primero:
- Lo que respondas a partir de ellos va primero y se lo atribuyes a su obra (y a su autor, si aparece): «Darío Silva-Silva, en Las llaves del poder, sostiene…».
- Completa con conocimiento general solo lo que los fragmentos no cubren, siguiendo las reglas de arriba, y no se lo atribuyas a ellos.
- No escribas los identificadores de los fragmentos en el texto.
- En used, los id (F1, F2…) de los fragmentos de los que sacaste algo; una lista vacía si no usaste ninguno.

Responde SOLO con un objeto JSON con las claves answer (el texto para el estudiante) y used (la lista de ids)."""
# A refusal Brain explains as its own model's failure says nothing about the
# question; it is left out of the grounding hint.
_MODEL_FAILURE_REASON = re.compile(r"^el modelo\b", re.IGNORECASE)

# Appended in code to the general prompt for a question copied from an
# evaluation: the concept is explained, the item is never answered.
GUIDE_RULE = (
    "\n- La pregunta viene de una evaluación del estudiante (cuestionario o tarea). "
    "Explica el concepto de fondo y termina con una o dos preguntas que lo ayuden a "
    "razonar. NO elijas ni descartes opciones (a, b, c…), NO digas cuál es la respuesta "
    "correcta y NO redactes la respuesta por él."
)

# Strong refs: asyncio keeps only weak references to tasks.
_tasks: set[asyncio.Task] = set()
# conversation_id → (job_id, question, started monotonic) — fallback registry.
_local_jobs: dict[str, tuple[str, str, float]] = {}
# conversation_id → queued questions — fallback for the store-backed queue.
_local_queues: dict[str, list[dict]] = {}


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
    # The book, document or author the question names (triage), or "".
    work: str = ""
    # The question was copied from an evaluation: explain, never answer the item.
    guide: bool = False
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


# ── The queue behind the running job ────────────────────────────────────────


def _queue_key(conversation_id: str) -> str:
    return f"{conversation_id}:queue"


def _queue_item(job: RagJob) -> dict:
    """What a queued question keeps: never the LLM config or its credentials."""
    return {"question": job.question, "work": job.work, "guide": job.guide, "queued_at": _now_iso()}


def _fresh(items: list) -> list[dict]:
    now = datetime.now(timezone.utc)
    out = []
    for item in items or []:
        if not isinstance(item, dict) or not item.get("question"):
            continue
        try:
            age = (now - datetime.fromisoformat(item.get("queued_at") or "")).total_seconds()
        except ValueError:
            continue
        if age <= JOB_TTL_SECONDS * (MAX_QUEUED + 1):
            out.append(item)
    return out


async def _read_queue(tenant_id: str, conversation_id: str) -> list[dict] | None:
    """The queue, or None when the store cannot be read (use the local one)."""
    store = get_store_or_none()
    if store is None:
        return None
    try:
        item = await store.aget(_namespace(tenant_id), _queue_key(conversation_id))
    except Exception as exc:  # noqa: BLE001
        logger.info("ismael_queue_read_failed", error=str(exc))
        return None
    value = getattr(item, "value", None) if item else None
    return _fresh((value or {}).get("items") or [])


async def _write_queue(tenant_id: str, conversation_id: str, items: list[dict]) -> bool:
    store = get_store_or_none()
    if store is None:
        return False
    try:
        await store.aput(_namespace(tenant_id), _queue_key(conversation_id), {"items": items})
    except Exception as exc:  # noqa: BLE001
        logger.info("ismael_queue_write_failed", error=str(exc))
        return False
    return True


async def enqueue(job: RagJob) -> int | None:
    """Queue ``job``'s question behind the running one: its 1-based position,
    or None when the queue is full."""
    items = await _read_queue(job.tenant_id, job.conversation_id)
    local = items is None
    if local:
        items = _fresh(_local_queues.get(job.conversation_id, []))
    if len(items) >= MAX_QUEUED:
        return None
    items.append(_queue_item(job))
    if local or not await _write_queue(job.tenant_id, job.conversation_id, items):
        _local_queues[job.conversation_id] = items
    logger.info("ismael_question_queued", conversation_id=job.conversation_id, position=len(items))
    return len(items)


async def take_next(tenant_id: str, conversation_id: str) -> dict | None:
    """Pop the oldest queued question, or None."""
    items = await _read_queue(tenant_id, conversation_id)
    if items is not None and items:
        head, rest = items[0], items[1:]
        if await _write_queue(tenant_id, conversation_id, rest):
            return head
    local = _fresh(_local_queues.get(conversation_id, []))
    if not local:
        _local_queues.pop(conversation_id, None)
        return None
    _local_queues[conversation_id] = local[1:]
    return local[0]


async def _spawn_next(job: RagJob) -> bool:
    """Start the next queued question with this job's LLM config."""
    item = await take_next(job.tenant_id, job.conversation_id)
    if not item:
        return False
    await spawn(
        replace(
            job,
            question=str(item.get("question") or ""),
            work=str(item.get("work") or ""),
            guide=bool(item.get("guide")),
            job_id=uuid.uuid4().hex,
        )
    )
    return True


# A title that is really a file name: "07-LlavesDelPoder-INT.pdf.corrected".
_FILE_EXTENSIONS = re.compile(r"(?:\.(?:pdf|epub|docx?|txt|corrected))+$", re.IGNORECASE)
_FILE_MARKERS = re.compile(r"(?:[-_ ]+(?:int(?:-s)?|s|corregido|corrected))+$", re.IGNORECASE)
_LEADING_NUMBER = re.compile(r"^\d+\s*[.\-_)]+\s*")
_CAMEL = re.compile(r"(?<=[a-záéíóúñ])(?=[A-ZÁÉÍÓÚÑ])")


def clean_title(title: str) -> str:
    """A library title fit to show: file names lose their extension, numbering
    and markers ("Llaves Del Poder"); real titles keep their punctuation."""
    title = " ".join((title or "").split())
    filename_like = bool(_FILE_EXTENSIONS.search(title)) or " " not in title
    cleaned = _FILE_EXTENSIONS.sub("", title)
    cleaned = _FILE_MARKERS.sub("", cleaned)
    cleaned = _LEADING_NUMBER.sub("", cleaned)
    if filename_like:
        cleaned = _CAMEL.sub(" ", cleaned)
        cleaned = " ".join(re.sub(r"[-_]+", " ", cleaned).split())
    return cleaned.strip() or title


# A recording's locator: "00:12:34 · https://www.youtube.com/watch?v=…&t=754s"
# (or an s3:// URL a student cannot open). Brain builds it from facts that never
# move, so it is parsed rather than rebuilt.
_TIMED_LOCATOR = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})\s*·\s*(\S+)")
MAX_REFERENCE_DETAIL = 200  # REFERENCE_LIMITS.maxDetailLength in the backend
MAX_REFERENCE_TITLE = 300


def _source_title(ev: dict, fallback: str) -> str:
    """«Las llaves del poder — Darío Silva-Silva (2004)» from the evidence's
    record facts (Brain's ficha); just the clean title when it has none."""
    title = clean_title(ev.get("title") or fallback)
    if not title:
        return ""
    who = (ev.get("author") or ev.get("source_name") or "").strip()
    year = ev.get("published_year")
    if who and who.lower() not in title.lower():
        title = f"{title} — {who}"
    if isinstance(year, int) and str(year) not in title:
        title = f"{title} ({year})"
    return title[:MAX_REFERENCE_TITLE]


def _timed_detail(locator: str) -> str | None:
    """«min 12:34 · <link that opens it there>» for a recording, else None."""
    match = _TIMED_LOCATOR.match(locator or "")
    if not match:
        return None
    hours, minutes, seconds, where = match.groups()
    at = f"{minutes}:{seconds}" if hours.strip("0") == "" else f"{int(hours)}:{minutes}:{seconds}"
    detail = f"min {at}"
    if where.startswith(("http://", "https://")) and len(detail) + 3 + len(where) <= MAX_REFERENCE_DETAIL:
        detail = f"{detail} · {where}"
    return detail


def _reference_items(answer: dict, lang: str) -> list[tuple[str, str | None]]:
    """(title, detail) per cited source: once per (title, page), in citation order.

    The title carries the author and year when Brain's ficha knows them; a
    recording's detail is the minute it was said, with its link.
    """
    evidence = {e.get("chunk_id"): e for e in answer.get("evidence") or [] if isinstance(e, dict)}
    items: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str]] = set()
    for citation in answer.get("citations") or []:
        if not isinstance(citation, dict):
            continue
        ev = evidence.get(citation.get("chunk_id")) or {}
        title = _source_title(ev, citation.get("section_title") or "")
        if not title:
            continue
        page = citation.get("page") or ev.get("page")
        where = (citation.get("section_title") or "").strip()
        if ev.get("kind") == "ficha":
            detail = None  # the record itself: nothing to point into
        elif timed := _timed_detail(citation.get("locator") or ev.get("locator") or ""):
            detail = timed
        else:
            detail = f"{text('page', lang)} {page}" if page else where
        key = (title, str(detail))
        if key in seen:
            continue
        seen.add(key)
        items.append((title, detail if detail and detail != title else None))
        if len(items) >= MAX_REFERENCES:
            break
    return items


# Brain's model sometimes opens with "Según los textos proporcionados, …": the
# student never sees those texts, so the lead-in reads as noise. Only this
# opening clause is removed; the claims that follow are untouched.
_SOURCE_PREAMBLE = re.compile(
    r"^(?:"
    r"(?:según|de acuerdo con|con base en|basándome en|a partir de)\s+(?:los|las)\s+"
    r"(?:textos|fragmentos|fuentes|documentos|pasajes)"
    # One adjective may sit in between: "los textos bíblicos citados".
    r"(?:\s+(?:bíblic|biblic|teológic|teologic)[oa]s)?"
    r"(?:\s+(?:proporcionad|disponibl|consultad|dad|citad|recibid)[oae]s)?"
    r"|"
    r"(?:según|de acuerdo con|con base en)\s+la\s+documentación"
    r"(?:\s+(?:proporcionad|disponibl|consultad|dad|citad|recibid)a)?"
    r"|"
    r"(?:according to|based on)\s+the\s+(?:provided\s+)?"
    r"(?:texts|fragments|sources|documents|passages|excerpts)(?:\s+provided)?"
    r")\s*,\s*",
    re.IGNORECASE,
)


def strip_source_preamble(body: str) -> str:
    """``body`` without a leading "according to the provided texts," clause."""
    match = _SOURCE_PREAMBLE.match(body)
    rest = body[match.end():] if match else ""
    if not rest.strip():
        return body
    return rest[0].upper() + rest[1:]


def format_answer(answer: dict, lang: str) -> str:
    """Brain's answer text plus a References block built from its citations.

    Brain verifies every citation server-side, so the text is sent as written —
    an LLM rewrite could detach a claim from the source that supports it. The
    one exception is a "según los textos proporcionados," lead-in, which carries
    no claim (``strip_source_preamble``).
    """
    body = strip_source_preamble((answer.get("text") or "").strip())
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


def _general_system(job: RagJob) -> str:
    """The tenant's THEOLOGY_GENERAL (or the default), filled in, plus the guide rule."""
    prompts = job.llm_config.get("prompts") or {}
    payload = prompts.get("THEOLOGY_GENERAL") or {}
    template = (payload.get("content") if isinstance(payload, dict) else "") or GENERAL_PROMPT
    language_rule = "Always respond in English." if job.language == "en" else "Responde siempre en español."
    system = template.replace("{persona}", job.persona).replace("{language_rule}", language_rule)
    return system + GUIDE_RULE if job.guide else system


async def _general_answer(job: RagJob) -> tuple[str, list]:
    """Concise general-knowledge answer for a question the library does not cover."""
    config = {"configurable": job.llm_config}
    system = _general_system(job)

    provider = get_provider(config)
    model = resolve_model(config)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": job.question}]
    reply = (await provider.chat(messages, model) or "").strip()
    usage = [make_usage_record(node="ismael_general_answer", provider=provider, model=model)]
    if lacks_accents(reply, job.language):
        retried = await provider.chat(
            [{"role": "system", "content": system + _ACCENT_RETRY_NOTE}, messages[1]], model
        )
        usage.append(make_usage_record(node="ismael_general_answer", provider=provider, model=model))
        retried = (retried or "").strip()
        logger.info("ismael_general_accents_retry", fixed=bool(retried) and not lacks_accents(retried, job.language))
        if retried and not lacks_accents(retried, job.language):
            reply = retried
    if not reply:
        raise brain.BrainError("general answer came back empty")
    return reply, usage


def _grounding_fragments(answer: dict) -> list[dict]:
    """The usable fragments of a refusal, best first, with an id each."""
    out = []
    evidence = [e for e in answer.get("evidence") or [] if isinstance(e, dict)]
    evidence.sort(key=lambda e: -float(e.get("score") or 0))
    for ev in evidence:
        body = (ev.get("text") or "").strip()
        if not body or ev.get("kind") == "ficha":
            continue
        out.append({**ev, "_id": f"F{len(out) + 1}", "_text": body[:MAX_FRAGMENT_CHARS]})
        if len(out) >= MAX_GROUNDING_FRAGMENTS:
            break
    return out


def _fragments_block(fragments: list[dict]) -> str:
    parts = []
    for fragment in fragments:
        work = _source_title(fragment, "") or "Sin título"
        where = (fragment.get("breadcrumb") or "").strip()
        head = f"<<<FUENTE id={fragment['_id']} | obra: {work}" + (f" | sección: {where}" if where else "")
        body = fragment["_text"].replace("FUENTE>>>", "")
        parts.append(f"{head}\n{body}\nFUENTE>>>")
    return "\n\n" + "\n\n".join(parts)


async def _grounded_answer(job: RagJob, answer: dict) -> tuple[dict, list] | None:
    """A general answer built on the fragments a refusal still carries.

    Returns ``(answer_dict, usage)`` shaped like Brain's — text, citations for
    the fragments used, their evidence — so `format_answer` and
    `answer_references` render it; None when there is nothing to ground on or
    the call fails (the plain general answer follows).
    """
    fragments = _grounding_fragments(answer)
    if not fragments:
        return None
    reason = (answer.get("reason") or "").strip()
    hint = "" if not reason or _MODEL_FAILURE_REASON.match(reason) else f" (lo que notó la búsqueda: {reason[:300]})"
    config = {"configurable": job.llm_config}
    system = (
        _general_system(job)
        + GROUNDING_RULE.replace("{reason}", hint)
        + _fragments_block(fragments)
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": job.question}]
    usage: list = []
    try:
        provider = get_provider(config)
        model = resolve_model(config)
        result = await _structured(provider, messages, model)
        usage.append(make_usage_record(node="ismael_grounded_answer", provider=provider, model=model))
        if lacks_accents(result.answer, job.language):
            messages[0] = {"role": "system", "content": system + _ACCENT_RETRY_NOTE}
            retried = await _structured(provider, messages, model)
            usage.append(make_usage_record(node="ismael_grounded_answer", provider=provider, model=model))
            if retried.answer.strip() and not lacks_accents(retried.answer, job.language):
                result = retried
    except Exception as exc:  # noqa: BLE001 — the plain general answer still follows
        logger.warning("ismael_grounded_answer_failed", error=str(exc) or type(exc).__name__)
        return None
    body = result.answer.strip()
    if not body:
        return None
    by_id = {f["_id"]: f for f in fragments}
    used = [by_id[i] for i in dict.fromkeys(i.strip().upper() for i in result.used) if i in by_id]
    logger.info("ismael_grounded_answer", shown=len(fragments), used=len(used))
    citations = [
        {
            "chunk_id": f.get("chunk_id"),
            "page": f.get("page"),
            "section_title": (f.get("breadcrumb") or "").strip() or None,
            "locator": f.get("locator") or "",
        }
        for f in used
    ]
    evidence = [{k: v for k, v in f.items() if not k.startswith("_")} for f in used]
    return {"text": body, "citations": citations, "evidence": evidence}, usage


async def _structured(provider, messages: list[dict], model: str) -> GroundedAnswer:
    return await asyncio.wait_for(
        provider.generate_structured(
            messages, model, GroundedAnswer, thinking_budget=settings.ismael_thinking_answer
        ),
        timeout=GROUNDED_TIMEOUT_SECONDS,
    )


def lacks_accents(reply: str, language: str) -> bool:
    """A long Spanish answer without a single accented letter lost its accents."""
    if language == "en" or len(reply.split()) < _ACCENT_CHECK_MIN_WORDS:
        return False
    return not _ACCENTED.search(reply)


_ARTICLES = {"el", "la", "los", "las", "lo", "un", "una", "del", "de", "the", "a", "an", "libro", "book"}
# Honorifics in an author field ("Rev. Darío Silva Silva") name no one.
_HONORIFICS = {"rev", "dr", "pastor", "pr", "mons", "padre", "fr"}


def _words(value: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", fold(value or "")).split()


def _core(value: str) -> str:
    """Folded words without leading articles: «el libro Las llaves» → «llaves …»."""
    words = _words(value)
    while words and words[0] in _ARTICLES:
        words.pop(0)
    return " ".join(words)


def _surnames(author: str) -> set[str]:
    return {w for w in _words(author) if len(w) >= 4 and w not in _HONORIFICS}


def _long_enough(core: str) -> bool:
    return len(core.split()) >= 2 or len(core) >= 8


def match_work(work: str, documents: list[dict]) -> dict | None:
    """The one catalog document ``work`` names, or None when none (or several) do.

    A title matches when one title contains the other as whole words (at least
    two words or eight letters), the title plus its subtitle included; an
    author (or any contributor in Brain's ficha) alone matches only when they
    have a single document. A title shared by several documents is settled by
    the author.
    """
    wanted = _core(work)
    if not wanted:
        return None
    wanted_words = set(wanted.split())
    padded = f" {wanted} "
    by_title, by_author = [], []
    for doc in documents:
        base = clean_title(doc.get("title") or "")
        # The full title too: «Las llaves del poder: autoridad espiritual» is
        # also how a student may name it, once Brain's ficha has a subtitle.
        titles = {_core(base)}
        if doc.get("subtitle"):
            titles.add(_core(f"{base} {doc['subtitle']}"))
        if any(
            title and (
                (_long_enough(title) and f" {title} " in padded)
                or (_long_enough(wanted) and padded in f" {title} ")
            )
            for title in titles
        ):
            by_title.append(doc)
        people = " ".join([doc.get("author") or "", *(doc.get("contributors") or [])])
        if _surnames(people) & wanted_words:
            by_author.append(doc)
    if len(by_title) == 1:
        return by_title[0]
    if len(by_title) > 1:
        both = [d for d in by_title if d in by_author]
        return both[0] if len(both) == 1 else None
    return by_author[0] if len(by_author) == 1 else None


async def _work_filter(job: RagJob) -> dict[str, str] | None:
    """``{"document_id": …}`` when the question names one catalog document."""
    if not job.work.strip():
        return None
    try:
        documents = await brain.list_documents()
    except Exception as exc:  # noqa: BLE001 — an unscoped question still gets answered
        logger.info("ismael_catalog_unavailable", error=str(exc))
        return None
    doc = match_work(job.work, documents)
    return {"document_id": doc["id"]} if doc else None


def _log_answer(job: RagJob, outcome: dict, filters: dict | None) -> None:
    """Why Brain answered as it did — an insufficient_evidence used to leave no trace."""
    answer = outcome.get("answer") if isinstance(outcome.get("answer"), dict) else {}
    titles = []
    for ev in answer.get("evidence") or []:
        title = clean_title(ev.get("title") or "") if isinstance(ev, dict) else ""
        if title and title not in titles:
            titles.append(title)
    logger.info(
        "ismael_brain_answer",
        job_id=job.job_id,
        state=answer.get("state") or outcome.get("state"),
        reason=(answer.get("reason") or "")[:300] or None,
        citations=len(answer.get("citations") or []),
        evidence_titles=titles[:5],
        document_id=(filters or {}).get("document_id"),
    )


async def _compose(job: RagJob) -> tuple[str, list, str, dict | None]:
    """(message, usage, outcome, references) for the job — never raises."""
    try:
        await brain.ensure_started()
        boot_deadline = time.monotonic() + settings.brain_boot_timeout_seconds
        if not await brain.wait_healthy(boot_deadline):
            return text("failed", job.language), [], "boot_timeout", None

        filters = await _work_filter(job)
        question_id = await brain.ask(job.question, filters=filters)
        answer_deadline = time.monotonic() + settings.brain_answer_timeout_seconds
        outcome = await brain.wait_answer(question_id, answer_deadline)
        if outcome is None:
            return text("failed", job.language), [], "answer_timeout", None
        _log_answer(job, outcome, filters)
        if outcome.get("state") != "done" or not isinstance(outcome.get("answer"), dict):
            logger.warning("ismael_brain_failed", error=outcome.get("error"))
            return text("failed", job.language), [], "brain_failed", None

        answer = outcome["answer"]
        if answer.get("state") == "answered" and (answer.get("text") or "").strip():
            return (
                _guided(job, format_answer(answer, job.language)),
                [],
                "answered",
                answer_references(answer, job.language),
            )

        if answer.get("state") == "insufficient_evidence":
            grounded = await _grounded_answer(job, answer)
            if grounded is not None:
                built, usage = grounded
                return (
                    _guided(job, format_answer(built, job.language)),
                    usage,
                    "grounded:insufficient_evidence",
                    answer_references(built, job.language),
                )

        # insufficient_evidence / off_corpus: a short general answer, unlabelled
        # (the outcome below still tells them apart in the logs).
        message, usage = await _general_answer(job)
        return _guided(job, message), usage, f"general:{answer.get('state')}", None
    except Exception as exc:  # noqa: BLE001 — the student still gets a reply
        # str() of a timeout is "": on 2026-10-26 two failures logged nothing.
        logger.error(
            "ismael_job_error",
            job_id=job.job_id,
            error=str(exc) or type(exc).__name__,
            error_type=type(exc).__name__,
        )
        return text("failed", job.language), [], "error", None


def _guided(job: RagJob, message: str) -> str:
    """An answer to a question copied from an evaluation opens with the note
    that the item is the student's to answer."""
    return f"{text('guide_note', job.language)}\n\n{message}" if job.guide else message


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
        try:
            # Before "finished" is recorded: the next job takes the slot, so a
            # question arriving now queues behind it instead of racing it.
            await _spawn_next(job)
        except Exception as exc:  # noqa: BLE001 — the student can ask again
            logger.error("ismael_queue_next_failed", job_id=job.job_id, error=str(exc))
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
