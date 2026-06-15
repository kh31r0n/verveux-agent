from typing import AsyncIterator
from openai import AsyncOpenAI
from .base import ChatProvider, UsageInfo
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
        super().__init__()
        self.api_key = resolve_api_key(config)
        self.client = AsyncOpenAI(api_key=self.api_key)


    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        # Reset usage in case a prior call populated it.
        self.last_usage = UsageInfo()
        # include_usage tells the API to emit a final chunk (with empty choices)
        # whose `.usage` carries token counts for the whole completion.
        kwargs_with_usage = {
            **kwargs,
            "stream_options": {"include_usage": True},
        }
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs_with_usage,
        )
        async for chunk in stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
            if chunk.usage is not None:
                self.last_usage = _usage_from_openai(chunk.usage)

    async def chat(
        self,
        messages: list[dict],
        model: str,
        **kwargs,
    ) -> str:
        self.last_usage = UsageInfo()
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        if response.usage is not None:
            self.last_usage = _usage_from_openai(response.usage)
        return response.choices[0].message.content or ""

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        response = await self.client.embeddings.create(
            input=texts,
            model=model,
        )
        return [embedding.embedding for embedding in response.data]


def _usage_from_openai(usage) -> UsageInfo:
    # OpenAI's usage object exposes prompt_tokens / completion_tokens at the top
    # level, with optional detail breakdowns for cached and reasoning tokens.
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return UsageInfo(
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cached_input_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
        reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
    )
