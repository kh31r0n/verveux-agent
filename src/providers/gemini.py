from typing import AsyncIterator
import asyncio
import json
import random

import structlog
from google.genai import Client, types
from google.oauth2 import service_account

from .base import ChatProvider, UsageInfo
from langgraph.types import RunnableConfig
from ..config import settings

logger = structlog.get_logger(__name__)


def _is_gemini_rate_limit(exc: Exception) -> bool:
    """True for a Gemini quota error (HTTP 429 / RESOURCE_EXHAUSTED).

    Detection is deliberately broad — the google-genai SDK surfaces the quota
    error as a ClientError whose code is 429 and whose message contains
    RESOURCE_EXHAUSTED — so we match either the numeric code or the status text
    without importing the SDK's error type. Scoped to this module: only the
    Gemini provider retries; openai/anthropic are untouched.
    """
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


# Sentinel credential type meaning "use Application Default Credentials".
# On Cloud Run that is the attached service account, which already holds
# roles/aiplatform.user — so the platform-default Gemini path needs no secret
# at all. Per-tenant credential blobs still take precedence.
#
# ADC is the ONLY practical auth for this provider. Gemini Enterprise Agent
# Platform is served from aiplatform.googleapis.com, and that service rejects
# API keys outright in a standard billed project — verified 2026-08-11 with an
# unrestricted key on the exact request its docs print:
#
#   POST …/v1/publishers/google/models/gemini-2.5-flash:generateContent?key=…
#   -> 401 UNAUTHENTICATED / CREDENTIALS_MISSING
#      "API keys are not supported by this API."
#
# The refusal comes from the service, not from a key restriction, so there is
# nothing to configure around it: the API-key path belongs to express mode,
# which is a separate onboarding with its own project. Do not reintroduce one.
ADC_CREDENTIALS = {"type": "adc"}

# ADC resolution is a blocking metadata-server call and GeminiProvider is
# constructed per node invocation, so the result is cached process-wide.
# (credentials, project_id) — project_id is whatever ADC reports, which may be
# None for some credential kinds.
_adc_cache: tuple[object, str | None] | None = None

# "global", not a region. Gemini 3 publishes ONLY to the global endpoint: on
# us-central1 every 3.x id (gemini-3.5-flash, 3.6-flash, 3.1-flash-lite,
# 3.1-pro-preview) answers 404 NOT_FOUND while the 2.5 family answers 200.
# Regionalising this silently pins the platform to Gemini 2.5.
_DEFAULT_ADC_LOCATION = "global"


def _load_adc() -> tuple[object, str | None]:
    global _adc_cache
    if _adc_cache is None:
        import google.auth

        _adc_cache = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        logger.info("gemini_adc_resolved", project_id=_adc_cache[1])
    return _adc_cache


def resolve_gemini_credentials(config: RunnableConfig) -> dict:
    creds = (config.get("configurable") or {}).get("gemini_credentials")
    if creds:
        return creds if isinstance(creds, dict) else json.loads(creds)
    json_str = settings.gemini_service_account_json
    if json_str:
        return json.loads(json_str)
    # No explicit credentials anywhere: fall back to ADC rather than failing.
    # A misconfiguration still surfaces — google.auth.default() raises
    # DefaultCredentialsError when there is nothing to fall back to.
    return dict(ADC_CREDENTIALS)


def resolve_gemini_project_id(
    config: RunnableConfig, credentials: dict | None = None
) -> str:
    project_id = (config.get("configurable") or {}).get("gemini_project_id") or settings.gemini_project_id
    # A tenant's own service-account blob names its project. Prefer it over ADC,
    # which would report the platform's project and silently bill Gemini calls
    # to the wrong place.
    if not project_id and credentials:
        project_id = credentials.get("project_id")
    if not project_id:
        # ADC knows its own project on Cloud Run; GCP_PROJECT_ID is not a
        # settings field, so this is the only place the runtime project surfaces.
        try:
            project_id = _load_adc()[1]
        except Exception as exc:  # noqa: BLE001 — fall through to the clear error
            logger.warning("gemini_adc_project_lookup_failed", error=str(exc))
    if not project_id:
        raise ValueError("Gemini project id not found.")
    return project_id


def resolve_gemini_location(config: RunnableConfig) -> str:
    location = (config.get("configurable") or {}).get("gemini_location") or settings.gemini_location
    if not location:
        # Unlike the project id, a location cannot be discovered — ADC carries
        # no region. Default to the global endpoint, which is the only one that
        # serves Gemini 3, rather than refusing to serve.
        location = _DEFAULT_ADC_LOCATION
    return location


class GeminiProvider(ChatProvider):
    """Gemini Enterprise Agent Platform, via aiplatform.googleapis.com.

    `vertexai=True` on the google-genai client selects that endpoint rather
    than the Gemini Developer API on generativelanguage.googleapis.com. The
    flag keeps the SDK's own name; the product it reaches is Agent Platform.
    """

    def __init__(self, config: RunnableConfig):
        super().__init__()
        self.credentials_dict = resolve_gemini_credentials(config)
        self.project_id = resolve_gemini_project_id(config, self.credentials_dict)
        self.location = resolve_gemini_location(config)

        cred_type = self.credentials_dict.get("type")
        if cred_type == "adc":
            # Cloud Run's attached service account (roles/aiplatform.user).
            credentials, _ = _load_adc()
        elif cred_type == "service_account":
            credentials = service_account.Credentials.from_service_account_info(
                self.credentials_dict,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        elif cred_type == "authorized_user":
            from google.oauth2.credentials import Credentials as OAuth2Credentials
            credentials = OAuth2Credentials.from_authorized_user_info(
                self.credentials_dict,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            raise ValueError(f"Unsupported credentials type: {cred_type}")

        self._client = Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
            credentials=credentials,
        )

    def _build_contents(self, messages: list[dict]) -> tuple[list, str | None]:
        """Convert OpenAI-style messages to Gemini content format."""
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=content)],
                ))
            else:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=content)],
                ))

        return contents, system_instruction

    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        self.last_usage = UsageInfo()
        contents, system_instruction = self._build_contents(messages)

        config = types.GenerateContentConfig(
            systemInstruction=system_instruction,
        ) if system_instruction else None

        # Retry only the Gemini 429/RESOURCE_EXHAUSTED quota error, and only
        # while no chunk has been yielded yet (a mid-stream failure can't be
        # replayed without duplicating text). Exponential backoff + full jitter.
        attempt = 0
        while True:
            yielded = False
            try:
                stream = await self._client.aio.models.generate_content_stream(
                    model=model or "gemini-3.5-flash",
                    contents=contents,
                    config=config,
                )
                async for response in stream:
                    yielded = True
                    if response.text:
                        yield response.text
                    # Gemini emits usage_metadata on the final chunk(s); the
                    # last non-empty value is cumulative for the completion.
                    usage_meta = getattr(response, "usage_metadata", None)
                    if usage_meta is not None:
                        self.last_usage = UsageInfo(
                            input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
                            output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
                            cached_input_tokens=getattr(usage_meta, "cached_content_token_count", 0) or 0,
                        )
                return
            except Exception as exc:
                if (
                    yielded
                    or attempt >= settings.gemini_max_retries
                    or not _is_gemini_rate_limit(exc)
                ):
                    raise
                delay = min(
                    settings.gemini_retry_base_seconds * (2 ** attempt),
                    settings.gemini_retry_max_seconds,
                )
                delay = random.uniform(0, delay)  # full jitter
                logger.warning(
                    "gemini_rate_limited_retry",
                    attempt=attempt + 1,
                    max_retries=settings.gemini_max_retries,
                    delay_seconds=round(delay, 2),
                )
                await asyncio.sleep(delay)
                attempt += 1
