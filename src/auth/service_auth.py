"""Service-to-service authentication.

Every caller of this service is another service, never an end user. The NestJS
backend drives ``/chat/stream`` and ``/chat/resume``; its schedulers drive
``/prospecting/run`` and ``/enrichment/run``. There has never been a browser or
mobile client on the other end, which is why this module verifies *workload*
identity (Google Cloud Run OIDC) rather than user identity. It replaces the AWS
Cognito verifier that used to live at ``src/auth/cognito.py``.

Two accepted credentials:

1. **Google-signed OIDC ID token** on ``Authorization: Bearer``. Cloud Run's IAM
   layer already validates this at the edge (the service runs with
   ``allow_unauthenticated = false``) and forwards it intact, so re-verifying
   here is defence in depth rather than the only gate — worth having because the
   same image runs under ``docker compose`` with no Cloud Run in front, and
   because ``INGRESS_TRAFFIC_INTERNAL_ONLY`` is a network boundary, not an
   identity one.
2. **The shared secret** (``WEBHOOK_API_KEY``) on ``X-System-Key`` or
   ``x-agent-key``. Historically the only live path, and still what local dev and
   the test suite use. Gated by ``ALLOW_SHARED_SECRET_AUTH`` so production can
   turn it off once every caller sends an ID token.

Both header spellings are accepted everywhere. They used to be split by endpoint
(``X-System-Key`` on the chat routes, ``x-agent-key`` on the batch routes) for no
reason beyond history, and keeping the split would have forced a lockstep deploy
across three backend call sites.
"""

import asyncio
import time
from typing import Optional

import httpx
import structlog
from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from ..config import settings
from ..observability import service_auth_rejects_total

logger = structlog.get_logger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)

# Google's OIDC signing keys and issuer. Both are fleet-wide constants, not
# per-project values — unlike Cognito, whose JWKS URL had to be assembled from a
# region and a user-pool id (the two settings this module deleted).
GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")

# The principal every authenticated service caller resolves to.
#
# LOAD-BEARING PERSISTED VALUE — do not change it. `scoped_thread_id` puts this
# string in the second segment of every LangGraph thread id, so every checkpoint
# row in Postgres reads `{tenantId}:system:{conversationId}:{codeName}:v{n}`.
# Returning anything else (a service-account email, a numeric `sub`) orphans the
# conversation history of every live conversation at once.
SERVICE_PRINCIPAL = "system"

# In-memory JWKS cache: (keys_dict, fetched_at_monotonic_seconds)
_jwks_cache: tuple[dict, float] = ({}, 0.0)
_JWKS_TTL_SECONDS = 300  # 5 minutes
_jwks_lock = asyncio.Lock()


def _allowed_service_accounts() -> frozenset[str]:
    """Service-account emails permitted to call this service, lowercased."""
    raw = settings.service_auth_allowed_service_accounts or ""
    return frozenset(
        part.strip().lower() for part in raw.split(",") if part.strip()
    )


async def get_jwks() -> dict:
    global _jwks_cache

    now = time.monotonic()
    cached_keys, fetched_at = _jwks_cache
    if cached_keys and (now - fetched_at) < _JWKS_TTL_SECONDS:
        return cached_keys

    async with _jwks_lock:
        # Re-check after acquiring lock in case another coroutine just refreshed
        cached_keys, fetched_at = _jwks_cache
        if cached_keys and (time.monotonic() - fetched_at) < _JWKS_TTL_SECONDS:
            return cached_keys

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(GOOGLE_JWKS_URL)
            response.raise_for_status()
            keys = response.json()

        _jwks_cache = (keys, time.monotonic())
        logger.info("jwks_refreshed", url=GOOGLE_JWKS_URL)
        return keys


def _reject(
    reason: str,
    detail: str,
    status_code: int = status.HTTP_401_UNAUTHORIZED,
) -> HTTPException:
    """Count and log a rejection, then build the exception to raise.

    Detail sent to the caller stays coarse; the specific reason lives in the log
    line and the metric label, mirroring the `[WidgetReject]` / `[FlutterReject]`
    convention on the backend.
    """
    service_auth_rejects_total.labels(reason=reason).inc()
    logger.warning("service_auth_reject", reason=reason)
    return HTTPException(status_code=status_code, detail=detail)


async def verify_service_caller(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    x_system_key: Optional[str] = Header(None, alias="X-System-Key"),
    x_agent_key: Optional[str] = Header(None, alias="x-agent-key"),
) -> dict:
    """Authenticate the calling service. Returns a principal dict on success."""

    # ── 1. Shared secret ─────────────────────────────────────────────────────
    # Checked first because it costs nothing. A *present but wrong* key does not
    # short-circuit: during the transition the backend sends both an ID token
    # and the legacy header, and a stale secret must not veto a valid token.
    presented_key = x_system_key or x_agent_key
    if presented_key is not None:
        if settings.allow_shared_secret_auth and presented_key == settings.webhook_api_key:
            return {
                "sub": SERVICE_PRINCIPAL,
                "is_system": True,
                "auth_method": "shared_secret",
            }
        logger.warning(
            "service_auth_shared_secret_unusable",
            enabled=settings.allow_shared_secret_auth,
        )

    # ── 2. Google OIDC ID token ──────────────────────────────────────────────
    audience = (settings.service_auth_audience or "").strip()
    if not audience:
        # OIDC not configured (local dev, tests). The shared secret was the only
        # option and it did not satisfy us.
        raise _reject("no_credentials", "Not authenticated")

    if credentials is None:
        raise _reject("no_credentials", "Not authenticated")

    allowed = _allowed_service_accounts()
    if not allowed:
        # Fail closed: an audience without an allowlist would accept an ID token
        # from any Google identity that can reach us.
        raise _reject(
            "allowlist_unconfigured",
            "Service authentication is misconfigured",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        jwks = await get_jwks()
    except Exception as exc:
        logger.error("jwks_fetch_failed", error=str(exc))
        raise _reject(
            "jwks_unavailable",
            "Unable to fetch Google signing keys",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        payload: dict = jwt.decode(
            credentials.credentials,
            jwks,
            algorithms=["RS256"],
            audience=audience,
            issuer=GOOGLE_ISSUERS,
            options={"leeway": 60},
        )
    except JWTError as exc:
        raise _reject("invalid_token", f"Invalid token: {exc}")
    except Exception as exc:
        logger.error(
            "token_decode_unexpected_error",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise _reject(
            "token_decode_error", f"Token validation error: {type(exc).__name__}"
        )

    # Google sets email/email_verified on service-account ID tokens. Identity is
    # the SA email, not `sub` (a numeric id nobody configures by hand).
    if payload.get("email_verified") is not True:
        raise _reject("email_unverified", "Caller identity is not verified")

    email = (payload.get("email") or "").lower()
    if email not in allowed:
        logger.warning("service_auth_caller_not_allowed", caller=email)
        raise _reject("caller_not_allowed", "Caller is not authorized")

    return {
        "sub": SERVICE_PRINCIPAL,
        "is_system": True,
        "auth_method": "oidc",
        "caller_email": email,
    }


def get_current_user(principal: dict = Depends(verify_service_caller)) -> str:
    """The thread-scoping principal for an authenticated caller.

    Always ``SERVICE_PRINCIPAL``. See the constant's note: this value is baked
    into every persisted thread id.
    """
    return principal["sub"]


def scoped_thread_id(
    tenant_id: str,
    user_sub: str,
    conversation_id: str,
    agent_code_name: str,
    agent_version: int,
) -> str:
    """Construct a thread_id scoped to tenant, user, conversation, agent code
    name, and agent version.

    Format: ``{tenantId}:{userSub}:{conversationId}:{agentCodeName}:v{agentVersion}``

    Every segment is independently queryable in logs. The graph that owns a
    checkpoint is explicit in the key, so cleanup for a decommissioned agent
    is a prefix scan on ``*:helena:*``.
    """
    return (
        f"{tenant_id}:{user_sub}:{conversation_id}:"
        f"{agent_code_name}:v{agent_version}"
    )
