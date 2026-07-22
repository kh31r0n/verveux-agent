import os

# Set required env vars before any app modules are imported (config.py validates at import time)
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/test")
os.environ.setdefault("COGNITO_USER_POOL_ID", "us-east-1_TestPool")
os.environ.setdefault("COGNITO_REGION", "us-east-1")
# Required at boot (config.py validates it) — prospecting web search key.
os.environ.setdefault("SERPER_API_KEY", "test-serper-key")
