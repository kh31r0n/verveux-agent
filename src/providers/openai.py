from typing import AsyncIterator
from openai import AsyncOpenAI
from .base import ChatProvider
from langgraph.types import RunnableConfig
from ..config import settings

def resolve_api_key(config: RunnableConfig) -> str:
    """Return the OpenAI API key."""
    key: str = (config.get("configurable") or {}).get("openai_api_key") or settings.openai_api_key
    if not key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY in .env or pass it via configurable."
        )
    return key

class OpenAIProvider(ChatProvider):
    def __init__(self, config: RunnableConfig):
        self.api_key = resolve_api_key(config)
        self.client = AsyncOpenAI(api_key=self.api_key)


    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def chat(
        self,
        messages: list[dict],
        model: str,
        **kwargs,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        response = await self.client.embeddings.create(
            input=texts,
            model=model,
        )
        return [embedding.embedding for embedding in response.data]
