import os

# Set required env vars before any app modules are imported (config.py validates at import time)
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/test")
# Service auth: no audience means OIDC verification is off, so the suite runs
# against the shared-secret path without touching Google's JWKS endpoint.
# tests/test_service_auth.py sets these per-test where it needs OIDC.
os.environ.setdefault("SERVICE_AUTH_AUDIENCE", "")
os.environ.setdefault("ALLOW_SHARED_SECRET_AUTH", "true")
# Required at boot (config.py validates it) — prospecting web search key.
os.environ.setdefault("SERPER_API_KEY", "test-serper-key")
# Google Places is opt-in per test: forced empty (an env var beats `.env`) so
# business listings fall back to the patched Serper path unless a test sets it.
os.environ["GOOGLE_PLACES_API_KEY"] = ""

import pytest  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402


@pytest.fixture(autouse=True)
def _no_serper_balance_lookup():
    """report_node reads the Serper balance (a real GET /account). Never from
    tests; the tests that care patch it themselves."""
    with patch(
        "src.agents.prospecting_nodes.serper_account_balance",
        AsyncMock(return_value=None),
    ):
        yield
