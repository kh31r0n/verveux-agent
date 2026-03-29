from langgraph.types import RunnableConfig
from openai import AsyncOpenAI

from .config import settings


def resolve_api_key(config: RunnableConfig) -> str:
    """Return the OpenAI API key.

    Priority:
    1. config["configurable"]["openai_api_key"]  — set by NestJS in production
    2. settings.openai_api_key (OPENAI_API_KEY env var) — fallback for LangGraph Studio / dev
    """
    key: str = (config.get("configurable") or {}).get("openai_api_key") or settings.openai_api_key
    if not key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY in .env or pass it via configurable."
        )
    return key


def get_openai_client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key)
