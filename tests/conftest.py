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
