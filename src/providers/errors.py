"""Telling a broken configuration apart from a bad page.

The autonomous graphs fan out one LLM call per search result and swallow the
failures individually, because one unparseable page must not sink a run of 130.
That is right for a bad page and catastrophic for a bad credential: every call
fails identically, the reduce step sees an empty candidate list, and the run
reports the one outcome indistinguishable from an honest empty search — a
COMPLETED run that found nothing.

It happened twice in one day on 2026-09-14, for two different reasons:

* the tenant's provider was not enabled platform-side, the agent fell back to
  the platform OpenAI key, and 72 extractions returned 429
  `credit_balance_exhausted`;
* after that was fixed, the tenant carried location `us-central1` against a
  `gemini-3.5-flash` model — which is published only to the `global` endpoint —
  and 132 extractions returned 404 NOT_FOUND.

Both runs reported COMPLETED with `created=0, errors=0`. Worse, COMPLETED also
took the day's `(tenantId, runDate)` slot while `retryFailedRun` accepts only a
FAILED run, so each false success locked the tenant out of retrying until the
next day.

`services.serper.SerperAuthError` already draws exactly this line for the search
credential. This is the same judgement for the LLM credential, and the two
should stay recognisably alike.
"""

from __future__ import annotations


class ProviderConfigError(RuntimeError):
    """The LLM provider rejected the request on configuration, not on content.

    Fatal to the whole run rather than a per-item miss: a wrong key, a disabled
    project or a model that is not served from the configured location fails
    every single call the same way, so continuing only spends time and Serper
    credits to arrive at a wrong answer. Raising here reaches the `/…/run`
    background task, which reports the run FAILED with the provider's own
    message — which is both the accurate status and the one that re-opens the
    retry path.
    """


# HTTP statuses that are a verdict on the configuration rather than on this
# particular request. 429 is deliberately absent: a plain rate limit IS
# transient and the Gemini provider already retries it with backoff.
_CONFIG_STATUSES = frozenset({401, 403, 404})

# ...except when a 429 is really "this account is out of money", which no amount
# of backoff fixes. OpenAI reports exactly this pair for an exhausted balance.
_FATAL_QUOTA_MARKERS = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "billing_hard_limit_reached",
)

# Status text for the same verdicts on the Google/Vertex side, which surfaces
# them as gRPC-style names rather than bare numbers.
_CONFIG_STATUS_TEXT = (
    "NOT_FOUND",
    "PERMISSION_DENIED",
    "UNAUTHENTICATED",
    "API_KEY_INVALID",
)


class StructuredOutputError(RuntimeError):
    """The model answered, but not with a value the requested schema accepts.

    ``kind`` says why: ``invalid_json`` (not parseable), ``schema_mismatch``
    (parseable, fails validation), ``truncated`` (the output limit was reached —
    possibly all on thinking — before the JSON closed) or ``empty`` (no text).
    Unlike ``ProviderConfigError`` the credential and model are fine, so the same
    item can succeed on a later attempt.
    """

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


def is_provider_config_error(exc: BaseException) -> bool:
    """True when `exc` is a verdict on the credential/model configuration.

    Matched without importing any provider SDK's exception types — there are
    three SDKs and they disagree about where the status lives — following
    `_is_gemini_rate_limit` in providers/gemini.py, which takes the same
    duck-typed approach for the same reason.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and status in _CONFIG_STATUSES:
        return True

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int) and response_status in _CONFIG_STATUSES:
        return True

    text = str(exc)

    # An exhausted balance reports 429, so it is checked on the text rather than
    # on the status — see _FATAL_QUOTA_MARKERS.
    if any(marker in text for marker in _FATAL_QUOTA_MARKERS):
        return True

    return any(marker in text for marker in _CONFIG_STATUS_TEXT)
