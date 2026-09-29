"""Company Brain client for the theology agent (ismael).

Brain is the cited RAG over the theology library. It runs on a single EC2 host
in AWS that is stopped whenever nobody uses it, so this module does two things:

* **Start the host** (`ensure_started`). The agent runs on Cloud Run, which has
  no AWS credentials. It asks the metadata server for a Google ID token with a
  fixed audience and trades it for short-lived AWS credentials with
  ``AssumeRoleWithWebIdentity`` — the role (yorch-aws-platform, brain-host
  module) trusts only this service account and may only start that instance.
  No AWS key is stored anywhere.
* **Ask** (`ask` / `collect`). Brain answers asynchronously: ``POST /ask`` hands
  the question over and returns an id, ``GET /ask/:id`` reports ``running``,
  ``done`` (with the answer) or ``failed``. Auth is a Brain service key sent as
  ``X-Api-Key``; the key is bound to one Brain tenant, which is therefore never
  sent.

boto3 is synchronous, so the two AWS calls run in a worker thread.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import structlog

from ..config import settings

logger = structlog.get_logger(__name__)

_METADATA_IDENTITY_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/"
    "service-accounts/default/identity"
)

# EC2 states in which a StartInstances call is pointless or refused: the host is
# already coming up (pending/running) or cannot be started until it finishes
# stopping — the caller's health loop retries the start a little later.
_NO_START_STATES = {"pending", "running", "stopping", "shutting-down"}


class BrainError(Exception):
    """Brain refused or failed a call; the message is safe to log."""


async def _google_id_token(audience: str) -> str:
    """ID token for this Cloud Run service account, minted by the metadata server."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(
            _METADATA_IDENTITY_URL,
            params={"audience": audience, "format": "full"},
            headers={"Metadata-Flavor": "Google"},
        )
        resp.raise_for_status()
        return resp.text.strip()


def _start_instance_sync(web_identity_token: str) -> str:
    """Assume the starter role and start the Brain host. Returns the prior state."""
    import boto3  # imported lazily: only this path needs it

    sts = boto3.client("sts", region_name=settings.brain_aws_region)
    creds = sts.assume_role_with_web_identity(
        RoleArn=settings.brain_aws_role_arn,
        RoleSessionName="ismael-brain-start",
        WebIdentityToken=web_identity_token,
        DurationSeconds=900,
    )["Credentials"]
    ec2 = boto3.client(
        "ec2",
        region_name=settings.brain_aws_region,
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
    )
    reservations = ec2.describe_instances(InstanceIds=[settings.brain_instance_id])[
        "Reservations"
    ]
    state = reservations[0]["Instances"][0]["State"]["Name"]
    if state not in _NO_START_STATES:
        ec2.start_instances(InstanceIds=[settings.brain_instance_id])
    return state


async def ensure_started() -> str:
    """Start the Brain host if it is stopped. Never raises.

    Returns the state the host was in (``stopped`` means this call started it),
    ``skipped`` when starting is disabled, or ``error``.
    """
    if settings.brain_start_mode != "aws":
        return "skipped"
    if not (settings.brain_aws_role_arn and settings.brain_instance_id):
        logger.warning("brain_start_not_configured")
        return "error"
    try:
        token = await _google_id_token(settings.brain_oidc_audience)
        state = await asyncio.to_thread(_start_instance_sync, token)
    except Exception as exc:  # noqa: BLE001 — a failed start degrades to a timeout
        logger.error("brain_start_failed", error=str(exc))
        return "error"
    logger.info("brain_start_requested", prior_state=state)
    return state


def _headers() -> dict[str, str]:
    return {"X-Api-Key": settings.brain_api_key}


async def is_healthy() -> bool:
    """True when the Brain API answers its liveness probe (502 while stopped)."""
    try:
        async with httpx.AsyncClient(timeout=settings.brain_request_timeout_seconds) as c:
            resp = await c.get(f"{settings.brain_api_url}/health/live")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def wait_healthy(deadline: float, restart_every: float = 30.0) -> bool:
    """Poll the liveness probe until `deadline` (a ``time.monotonic()`` value).

    Re-requests the start every `restart_every` seconds: a host caught mid-stop
    by the idle watcher refuses StartInstances until it has fully stopped, and a
    later retry is what brings it back.
    """
    last_start = time.monotonic()
    while time.monotonic() < deadline:
        if await is_healthy():
            return True
        if time.monotonic() - last_start >= restart_every:
            await ensure_started()
            last_start = time.monotonic()
        await asyncio.sleep(settings.brain_poll_interval_seconds)
    return False


async def ask(text: str) -> str:
    """Hand a question to Brain; returns its question id."""
    body: dict[str, Any] = {
        "library_id": settings.brain_library_id,
        "text": text,
        "effort": settings.brain_ask_effort,
    }
    async with httpx.AsyncClient(timeout=settings.brain_request_timeout_seconds) as c:
        resp = await c.post(f"{settings.brain_api_url}/ask", json=body, headers=_headers())
    if resp.status_code != 200:
        raise BrainError(f"POST /ask → {resp.status_code}: {resp.text[:300]}")
    question_id = resp.json().get("question_id")
    if not question_id:
        raise BrainError("POST /ask returned no question_id")
    return str(question_id)


async def collect(question_id: str) -> dict:
    """One ``GET /ask/:id``: ``{state: running|done|failed, answer, error}``."""
    async with httpx.AsyncClient(timeout=settings.brain_request_timeout_seconds) as c:
        resp = await c.get(f"{settings.brain_api_url}/ask/{question_id}", headers=_headers())
    if resp.status_code != 200:
        raise BrainError(f"GET /ask/{question_id} → {resp.status_code}: {resp.text[:300]}")
    return resp.json()


async def wait_answer(question_id: str, deadline: float) -> dict | None:
    """Poll a question until it leaves ``running``; None when `deadline` passes."""
    while time.monotonic() < deadline:
        outcome = await collect(question_id)
        if outcome.get("state") != "running":
            return outcome
        await asyncio.sleep(settings.brain_poll_interval_seconds)
    return None
