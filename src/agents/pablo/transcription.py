"""Google Cloud Speech-to-Text v2 for audio answers.

The audio is a temporary blob the backend uploaded (pre-signed URL); pablo
downloads it, transcribes it with synchronous ``recognize`` (≤ 60 s of audio,
the activity caps recordings to that) and never stores it. The backend deletes
the blob once the response is reported, and its TTL cleanup is the safety net.

Auth is Application Default Credentials (Cloud Run's service account), the
same as the Gemini provider: no key in configuration. The REST API is called
directly so the service needs no extra SDK.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass

import structlog
from httpx import AsyncClient, HTTPError

from ...config import settings

logger = structlog.get_logger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


@dataclass
class Transcript:
    text: str
    # "ok" | "failed" | "empty"
    status: str
    error: str = ""


def _language_codes(lang: str) -> list[str]:
    mapping = dict(
        pair.split(":", 1) for pair in settings.pablo_stt_languages.split(",") if ":" in pair
    )
    return [mapping.get(lang) or mapping.get("es") or "es-US"]


def _access_token() -> tuple[str, str]:
    """(token, project) from ADC. Blocking — run in a thread."""
    import google.auth
    from google.auth.transport.requests import Request

    credentials, project = google.auth.default(scopes=_SCOPES)
    credentials.refresh(Request())
    return credentials.token, settings.pablo_stt_project_id or project or ""


async def _download(url: str) -> bytes:
    if not url.startswith("https://") and not settings.pablo_allow_http_audio:
        raise ValueError("audio_url must be https")
    async with AsyncClient(timeout=settings.pablo_stt_timeout_seconds) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        if len(resp.content) > settings.pablo_max_audio_bytes:
            raise ValueError("audio too large")
        return resp.content


async def _recognize(audio: bytes, lang: str) -> str:
    token, project = await asyncio.to_thread(_access_token)
    if not project:
        raise ValueError("no project for Speech-to-Text")
    location = settings.pablo_stt_location
    host = "speech.googleapis.com" if location == "global" else f"{location}-speech.googleapis.com"
    url = f"https://{host}/v2/projects/{project}/locations/{location}/recognizers/_:recognize"
    body = {
        "config": {
            "autoDecodingConfig": {},
            "languageCodes": _language_codes(lang),
            "model": settings.pablo_stt_model,
            "features": {"enableAutomaticPunctuation": True},
        },
        "content": base64.b64encode(audio).decode("ascii"),
    }
    async with AsyncClient(timeout=settings.pablo_stt_timeout_seconds) as client:
        resp = await client.post(url, json=body, headers={"Authorization": f"Bearer {token}"})
        resp.raise_for_status()
        data = resp.json()
    pieces = [
        (result.get("alternatives") or [{}])[0].get("transcript", "")
        for result in data.get("results") or []
    ]
    return " ".join(p.strip() for p in pieces if p.strip())


async def transcribe(audio_url: str, lang: str) -> Transcript:
    """The transcript, retried ``pablo_stt_attempts`` times with backoff.

    Never raises: a failure is a status the backend stores
    (``transcription_failed``) and the teacher reviews.
    """
    last_error = ""
    try:
        audio = await _download(audio_url)
    except (HTTPError, ValueError) as exc:
        logger.warning("pablo_audio_download_failed", error=str(exc))
        return Transcript(text="", status="failed", error=f"download: {exc}")
    for attempt in range(settings.pablo_stt_attempts):
        try:
            transcript = await _recognize(audio, lang)
            return Transcript(text=transcript, status="ok" if transcript else "empty")
        except Exception as exc:  # noqa: BLE001 — every failure is retried, then reported
            last_error = str(exc)[:300]
            logger.warning("pablo_stt_failed", attempt=attempt + 1, error=last_error)
            if attempt + 1 < settings.pablo_stt_attempts:
                await asyncio.sleep(0.5 * 2**attempt)
    return Transcript(text="", status="failed", error=last_error)
