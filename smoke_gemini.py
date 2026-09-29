"""Manual smoke test for the Gemini Enterprise provider, over ADC.

    gcloud auth application-default login     # once, on a laptop
    venv/bin/python smoke_gemini.py

Replaces the old test_vertex.py, which drove the provider with an API key.
That path is gone: Agent Platform (aiplatform.googleapis.com) rejects API keys
in a standard billed project with 401 CREDENTIALS_MISSING regardless of how the
key is restricted, so ADC and per-tenant service accounts are the only auth.

On Cloud Run this same code path runs against the attached service account,
which holds roles/aiplatform.user.
"""

import asyncio
import os

# Minimal env vars so config.py doesn't fail on import
os.environ.setdefault("DATABASE_URL", "postgres://x")
os.environ.setdefault("SERPER_API_KEY", "test-serper-key")

from src.providers.gemini import GeminiProvider

# Empty configurable -> resolve_gemini_credentials falls back to ADC and
# resolve_gemini_location falls back to "global", which is the only endpoint
# that serves Gemini 3 (every 3.x id is 404 on us-central1).
CONFIG = {"configurable": {}}

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")


async def main():
    provider = GeminiProvider(CONFIG)
    print(f"project={provider.project_id} location={provider.location} model={MODEL}\n")

    messages = [
        {"role": "system", "content": "Eres un asistente amable. Responde en español."},
        {"role": "user", "content": "Hola, ¿cómo estás?"},
    ]

    async for chunk in provider.stream_chat(messages=messages, model=MODEL):
        print(chunk, end="", flush=True)

    u = provider.last_usage
    print(
        f"\n\nusage: in={u.input_tokens} out={u.output_tokens} "
        f"cached={u.cached_input_tokens}"
    )
    assert u.input_tokens > 0, "usage_metadata never arrived — billing would read 0"


if __name__ == "__main__":
    asyncio.run(main())
