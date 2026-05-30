"""
Standalone test for the Vertex AI provider configured in this project.
Run: cd verveux-agent && .venv/bin/python test_vertex.py
"""

import asyncio
import os

# Minimal env vars so config.py doesn't fail on import
os.environ.setdefault("DATABASE_URL", "postgres://x")
os.environ.setdefault("COGNITO_USER_POOL_ID", "test")
os.environ.setdefault("COGNITO_REGION", "us-east-1")

from src.providers.vertex import VertexProvider


async def main():
    # Same credentials the backend provides to the agent
    config = {
        "configurable": {
            "vertex_credentials": {
                "apiKey": os.environ["VERTEX_API_KEY"],
            },
            "vertex_project_id": "verveux",
            "vertex_location": "us-central1",
        }
    }

    provider = VertexProvider(config)

    messages = [
        {"role": "system", "content": "Eres un asistente amable. Responde en español."},
        {"role": "user", "content": "Hola, ¿cómo estás?"},
    ]

    print("--- Streaming response from Vertex AI (gemini-1.5-flash) ---\n")

    async for chunk in provider.stream_chat(
        messages=messages,
        model="gemini-2.5-flash",
    ):
        print(chunk, end="", flush=True)

    print("\n\n--- Done ---")


if __name__ == "__main__":
    asyncio.run(main())
