"""Service-to-service authentication (src/auth/service_auth.py).

The module this replaces (the AWS Cognito verifier) had no tests at all, and its
JWT branch was dead in production — every real request short-circuited on the
shared secret. So these are the first tests of the token path in this service.

Tokens are signed with a throwaway RSA key and `get_jwks` is patched to return
the matching JWKS, so nothing here touches the network.
"""

import base64
import time
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt

from src.auth import service_auth
from src.auth.service_auth import (
    SERVICE_PRINCIPAL,
    get_current_user,
    scoped_thread_id,
    verify_service_caller,
)
from src.config import settings

AUDIENCE = "https://yorchio-agent-123.us-central1.run.app"
CALLER = "yorchio-backend@yorch-platform-prod.iam.gserviceaccount.com"
KID = "test-signing-key"


# ── Signing helpers ──────────────────────────────────────────────────────────


def _b64u(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


@pytest.fixture(scope="module")
def signing_key() -> tuple[str, dict]:
    """(private PEM, JWKS) for a throwaway RSA key."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    numbers = key.public_key().public_numbers()
    jwks = {
        "keys": [
            {
                "kty": "RSA",
                "kid": KID,
                "use": "sig",
                "alg": "RS256",
                "n": _b64u(numbers.n),
                "e": _b64u(numbers.e),
            }
        ]
    }
    return pem, jwks


def _token(signing_key: tuple[str, dict], **overrides) -> str:
    """A Google service-account ID token, shaped as Cloud Run mints them."""
    pem, _ = signing_key
    now = int(time.time())
    claims = {
        "iss": "https://accounts.google.com",
        "aud": AUDIENCE,
        "azp": "112233445566778899",
        "sub": "112233445566778899",
        "email": CALLER,
        "email_verified": True,
        "iat": now,
        "exp": now + 3600,
        **overrides,
    }
    return jwt.encode(claims, pem, algorithm="RS256", headers={"kid": KID})


def _bearer(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


async def _verify(credentials=None, x_system_key=None, x_agent_key=None) -> dict:
    """Call the dependency directly with every parameter supplied.

    Omitting one would leave FastAPI's `Header(None)` / `Security(...)` sentinel
    in place — a truthy object, not None — and the credential checks would read
    it as a presented value.
    """
    return await verify_service_caller(
        credentials=credentials,
        x_system_key=x_system_key,
        x_agent_key=x_agent_key,
    )


@pytest.fixture
def oidc_enabled(monkeypatch, signing_key):
    """Configure OIDC verification and serve the local JWKS."""
    _, jwks = signing_key
    monkeypatch.setattr(settings, "service_auth_audience", AUDIENCE)
    monkeypatch.setattr(
        settings, "service_auth_allowed_service_accounts", f" {CALLER.upper()} "
    )
    monkeypatch.setattr(settings, "allow_shared_secret_auth", False)
    with patch.object(service_auth, "get_jwks", AsyncMock(return_value=jwks)):
        yield


# ── The checkpoint invariant ─────────────────────────────────────────────────


class TestServicePrincipalIsStable:
    """`SERVICE_PRINCIPAL` is persisted data, not an auth detail.

    It is the second segment of every LangGraph thread id already in Postgres
    (`{tenantId}:system:{conversationId}:{codeName}:v{n}`). If authenticating by
    a different mechanism produced a different principal — an SA email, a numeric
    `sub` — every live conversation would silently lose its checkpoint history.
    """

    def test_the_literal_has_not_drifted(self) -> None:
        assert SERVICE_PRINCIPAL == "system"

    def test_thread_ids_keep_their_shape(self) -> None:
        assert (
            scoped_thread_id("t1", SERVICE_PRINCIPAL, "c1", "helena", 2)
            == "t1:system:c1:helena:v2"
        )

    async def test_oidc_callers_resolve_to_the_same_principal(
        self, oidc_enabled, signing_key
    ) -> None:
        principal = await _verify(credentials=_bearer(_token(signing_key)))
        assert principal["sub"] == SERVICE_PRINCIPAL
        assert get_current_user(principal) == "system"

    async def test_shared_secret_callers_resolve_to_the_same_principal(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(settings, "allow_shared_secret_auth", True)
        principal = await _verify(x_system_key=settings.webhook_api_key)
        assert get_current_user(principal) == "system"


# ── OIDC verification ────────────────────────────────────────────────────────


class TestOidcVerification:
    async def test_accepts_an_allowlisted_caller(
        self, oidc_enabled, signing_key
    ) -> None:
        principal = await _verify(credentials=_bearer(_token(signing_key)))
        assert principal["auth_method"] == "oidc"
        assert principal["caller_email"] == CALLER.lower()

    async def test_rejects_a_token_minted_for_another_audience(
        self, oidc_enabled, signing_key
    ) -> None:
        token = _token(signing_key, aud="https://some-other-service.run.app")
        with pytest.raises(HTTPException) as exc:
            await _verify(credentials=_bearer(token))
        assert exc.value.status_code == 401

    async def test_rejects_a_foreign_issuer(self, oidc_enabled, signing_key) -> None:
        token = _token(signing_key, iss="https://evil.example.com")
        with pytest.raises(HTTPException) as exc:
            await _verify(credentials=_bearer(token))
        assert exc.value.status_code == 401

    async def test_rejects_a_caller_outside_the_allowlist(
        self, oidc_enabled, signing_key
    ) -> None:
        token = _token(signing_key, email="attacker@some-other-project.iam.gserviceaccount.com")
        with pytest.raises(HTTPException) as exc:
            await _verify(credentials=_bearer(token))
        assert exc.value.status_code == 401

    async def test_rejects_an_unverified_identity(
        self, oidc_enabled, signing_key
    ) -> None:
        token = _token(signing_key, email_verified=False)
        with pytest.raises(HTTPException) as exc:
            await _verify(credentials=_bearer(token))
        assert exc.value.status_code == 401

    async def test_rejects_an_expired_token(self, oidc_enabled, signing_key) -> None:
        now = int(time.time())
        token = _token(signing_key, iat=now - 7200, exp=now - 3600)
        with pytest.raises(HTTPException) as exc:
            await _verify(credentials=_bearer(token))
        assert exc.value.status_code == 401

    async def test_rejects_a_token_signed_by_someone_else(
        self, oidc_enabled, signing_key
    ) -> None:
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = other.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode()
        forged = jwt.encode(
            {
                "iss": "https://accounts.google.com",
                "aud": AUDIENCE,
                "email": CALLER,
                "email_verified": True,
                "exp": int(time.time()) + 3600,
            },
            pem,
            algorithm="RS256",
            headers={"kid": KID},
        )
        with pytest.raises(HTTPException) as exc:
            await _verify(credentials=_bearer(forged))
        assert exc.value.status_code == 401

    async def test_fails_closed_when_the_allowlist_is_empty(
        self, oidc_enabled, monkeypatch, signing_key
    ) -> None:
        """An audience with no allowlist would accept any Google identity."""
        monkeypatch.setattr(settings, "service_auth_allowed_service_accounts", "")
        with pytest.raises(HTTPException) as exc:
            await _verify(credentials=_bearer(_token(signing_key)))
        assert exc.value.status_code == 503

    async def test_reports_jwks_unavailability_as_503(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "service_auth_audience", AUDIENCE)
        monkeypatch.setattr(settings, "service_auth_allowed_service_accounts", CALLER)
        monkeypatch.setattr(settings, "allow_shared_secret_auth", False)
        with patch.object(
            service_auth, "get_jwks", AsyncMock(side_effect=RuntimeError("no net"))
        ):
            with pytest.raises(HTTPException) as exc:
                await _verify(credentials=_bearer("whatever"))
        assert exc.value.status_code == 503


# ── Shared secret ────────────────────────────────────────────────────────────


class TestSharedSecret:
    async def test_accepts_either_header_spelling(self, monkeypatch) -> None:
        """The two names were split by endpoint before; both work everywhere now."""
        monkeypatch.setattr(settings, "allow_shared_secret_auth", True)
        for kwargs in (
            {"x_system_key": settings.webhook_api_key},
            {"x_agent_key": settings.webhook_api_key},
        ):
            principal = await _verify(**kwargs)
            assert principal["auth_method"] == "shared_secret"

    async def test_rejects_a_wrong_secret(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "allow_shared_secret_auth", True)
        with pytest.raises(HTTPException) as exc:
            await _verify(x_agent_key="nope")
        assert exc.value.status_code == 401

    async def test_rejects_the_secret_once_disabled(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "allow_shared_secret_auth", False)
        with pytest.raises(HTTPException) as exc:
            await _verify(x_system_key=settings.webhook_api_key)
        assert exc.value.status_code == 401

    async def test_a_stale_secret_does_not_veto_a_valid_token(
        self, oidc_enabled, signing_key
    ) -> None:
        """Dual-send during the migration: the backend sends both credentials.

        A rotated-but-not-yet-redeployed secret must not turn a perfectly good
        ID token into a 401.
        """
        principal = await _verify(
            credentials=_bearer(_token(signing_key)),
            x_system_key="stale-secret-from-a-previous-rotation",
        )
        assert principal["auth_method"] == "oidc"

    async def test_rejects_a_caller_with_no_credentials_at_all(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(settings, "service_auth_audience", "")
        with pytest.raises(HTTPException) as exc:
            await _verify()
        assert exc.value.status_code == 401


# ── Endpoint wiring ──────────────────────────────────────────────────────────


class TestEndpointsShareOneDependency:
    """All four authenticated routes accept the same credentials.

    `/chat/stream` used to read only `X-System-Key` while the batch routes read
    only `x-agent-key`.
    """

    BODIES = {
        "/prospecting/run": {"tenant_id": "t1", "run_id": "r1"},
        "/enrichment/run": {
            "tenant_id": "t1",
            "attempt_id": "a1",
            "contact_id": "c1",
            "website_url": "https://acme.com/",
        },
    }

    def _client(self):
        from fastapi.testclient import TestClient

        from src.main import app

        return TestClient(app)

    @pytest.mark.parametrize("path", list(BODIES))
    @pytest.mark.parametrize("header", ["x-agent-key", "X-System-Key"])
    def test_batch_routes_accept_both_header_spellings(
        self, monkeypatch, path: str, header: str
    ) -> None:
        from src.graphs.registry import UnknownCodeNameError

        monkeypatch.setattr(settings, "allow_shared_secret_auth", True)
        with patch("src.main.get_or_compile_graph", AsyncMock()) as compile_mock:
            compile_mock.side_effect = UnknownCodeNameError("nobody", [])
            response = self._client().post(
                path,
                json={**self.BODIES[path], "agent_code_name": "nobody"},
                headers={header: settings.webhook_api_key},
            )
        # Authenticated, then rejected on the unknown code name — a 401 here
        # would mean the credential never got past the dependency.
        assert response.status_code == 400

    @pytest.mark.parametrize("path", list(BODIES))
    def test_batch_routes_reject_a_missing_credential(self, path: str) -> None:
        assert self._client().post(path, json=self.BODIES[path]).status_code == 401

    def test_chat_stream_rejects_a_missing_credential(self) -> None:
        response = self._client().post(
            "/chat/stream",
            json={"tenant_id": "t1", "conversation_id": "c1", "message": "hola"},
        )
        assert response.status_code == 401
