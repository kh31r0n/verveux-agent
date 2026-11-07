"""Running pablo's background jobs (scoring, class report).

They are started by the backend (202) and report back through ``/internal``
endpoints, so a long attempt never hits the 60 s chat turn. Credentials are
fetched here — there is no chat turn to inherit them from — and the tenant's
own model is used: a student and a teacher read what these jobs write.
"""

from __future__ import annotations

import asyncio

import structlog

from ...config import settings
from .. import backend_client

logger = structlog.get_logger(__name__)

_tasks: set[asyncio.Task] = set()
_running: set[str] = set()
_semaphore: asyncio.Semaphore | None = None


def _gate() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(max(1, settings.pablo_job_concurrency))
    return _semaphore


async def llm_config(tenant_id: str, prompts: dict) -> dict:
    """A RunnableConfig for the tenant's chat model (never logged)."""
    provider = "openai"
    model = ""
    keys: dict = {}
    try:
        creds = await backend_client.fetch_agent_credentials(tenant_id)
        provider = creds.get("provider", "OPENAI").lower()
        model = creds.get("model") or ""
        if provider == "openai":
            keys["openai_api_key"] = creds.get("apiKey", "")
        elif provider == "anthropic":
            keys["anthropic_api_key"] = creds.get("apiKey", "")
        elif provider == "gemini":
            keys["gemini_credentials"] = creds.get("geminiCredentials", {})
            keys["gemini_project_id"] = creds.get("geminiProjectId", "")
            keys["gemini_location"] = creds.get("geminiLocation", "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("pablo_credentials_fetch_failed", tenant_id=tenant_id, error=str(exc))
        keys["openai_api_key"] = settings.openai_api_key
    return {
        "configurable": {
            "llm_provider": provider,
            "llm_model": model,
            "prompts": prompts or {},
            **keys,
        }
    }


def spawn(key: str, coro) -> bool:
    """Start ``coro`` unless a job with the same key is already running in this
    process. Returns False for the duplicate (the backend retried)."""
    if key in _running:
        coro.close()
        return False
    _running.add(key)

    async def _run():
        try:
            async with _gate():
                await coro
        except Exception as exc:  # noqa: BLE001 — a job must never crash the service
            logger.error("pablo_job_crashed", key=key, error=str(exc)[:500])
        finally:
            _running.discard(key)

    task = asyncio.create_task(_run())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return True
