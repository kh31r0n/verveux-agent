import asyncio
import time
from typing import Optional

import httpx
import structlog
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from ..config import settings

logger = structlog.get_logger(__name__)

bearer_scheme = HTTPBearer()

# In-memory JWKS cache: (keys_dict, fetched_at_unix_seconds)
_jwks_cache: tuple[dict, float] = ({}, 0.0)
_JWKS_TTL_SECONDS = 300  # 5 minutes
_jwks_lock = asyncio.Lock()


def _cognito_jwks_url() -> str:
    return f"https://cognito-idp.{settings.cognito_region}.amazonaws.com/{settings.cognito_user_pool_id}/.well-known/jwks.json"


def _cognito_issuer() -> str:
    return f"https://cognito-idp.{settings.cognito_region}.amazonaws.com/{settings.cognito_user_pool_id}"


async def get_jwks() -> dict:
    global _jwks_cache

    now = time.monotonic()
    cached_keys, fetched_at = _jwks_cache
    if cached_keys and (now - fetched_at) < _JWKS_TTL_SECONDS:
        return cached_keys

    async with _jwks_lock:
        # Re-check after acquiring lock in case another coroutine just refreshed
        cached_keys, fetched_at = _jwks_cache
        if cached_keys and (now - fetched_at) < _JWKS_TTL_SECONDS:
            return cached_keys

        jwks_url = _cognito_jwks_url()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(jwks_url)
            response.raise_for_status()
            keys = response.json()

        _jwks_cache = (keys, time.monotonic())
        logger.info("jwks_refreshed", url=jwks_url)
        return keys


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> dict:
    token = credentials.credentials
    try:
        jwks = await get_jwks()
    except Exception as exc:
        logger.error("jwks_fetch_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to fetch JWKS from Cognito: {exc}",
        )

    try:
        decode_options = {"verify_aud": False, "leeway": 60}

        payload: dict = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            issuer=_cognito_issuer(),
            options=decode_options,
        )

        # Cognito-specific: validate token_use claim
        token_use = payload.get("token_use")
        if token_use not in ("access", "id"):
            raise JWTError(f"Invalid token_use claim: {token_use}")

        # Optionally validate audience/client_id
        if settings.cognito_app_client_id:
            # Access tokens use "client_id", ID tokens use "aud"
            client_id = payload.get("client_id") or payload.get("aud")
            if client_id != settings.cognito_app_client_id:
                raise JWTError(f"Token client_id/aud mismatch: {client_id}")

        return payload
    except JWTError as exc:
        logger.warning("token_validation_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        )
    except Exception as exc:
        logger.error("token_decode_unexpected_error", error=str(exc), error_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation error: {type(exc).__name__}",
        )


def get_current_user(payload: dict = Depends(verify_token)) -> str:
    sub: Optional[str] = payload.get("sub")
    if not sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing sub claim",
        )
    return sub


def scoped_thread_id(user_sub: str, client_thread_id: str) -> str:
    """Construct a thread_id scoped to the authenticated user."""
    return f"{user_sub}:{client_thread_id}"
