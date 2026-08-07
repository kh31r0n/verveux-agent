"""
Machine-to-machine auth for service callers of the non-chat endpoints (e.g. the
spend-file repair the FinOps import calls). Distinct from the Cognito JWT used by
the chat endpoints — there is no user context here, no SSE, and no interrupt.

The shared secret is the same `WEBHOOK_API_KEY` the backend already uses in both
directions: `X-System-Key` inbound (see `get_current_user` in auth/cognito.py,
which accepts it as a system caller) and `X-Agent-Key` outbound (backend_client).
This dependency is the *required* form of that check — it never falls back to a JWT.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Optional

import structlog
from fastapi import Header, HTTPException, Request, status

from ..config import settings

logger = structlog.get_logger(__name__)


def verify_hmac(secret: str, signature: Optional[str], body: bytes) -> bool:
    """Constant-time check of `sha256=<hex>` over the raw body. Pure — unit-tested."""
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")


async def require_system_key(
    request: Request = None,  # injected by FastAPI; defaulted so unit tests can omit it
    x_system_key: Optional[str] = Header(default=None, alias="X-System-Key"),
    x_hmac_signature: Optional[str] = Header(default=None, alias="x-hmac-signature"),
) -> None:
    """FastAPI dependency: validate ``X-System-Key`` (constant-time) and, when an HMAC
    secret is configured, the ``x-hmac-signature`` over the raw body."""
    expected = settings.webhook_api_key
    if not expected:
        # Misconfiguration — fail closed rather than accept any caller.
        logger.error("system_key_not_configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System key not configured on the agent",
        )
    if not x_system_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-System-Key header",
        )
    if not hmac.compare_digest(x_system_key, expected):
        logger.warning("system_key_invalid")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid system key",
        )

    secret = settings.agent_machine_hmac_secret
    if secret:
        if not x_hmac_signature:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing x-hmac-signature header",
            )
        body = await request.body() if request is not None else b""
        if not verify_hmac(secret, x_hmac_signature, body):
            logger.warning("machine_hmac_invalid")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid HMAC signature",
            )
