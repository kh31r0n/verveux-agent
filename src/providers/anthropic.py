from typing import AsyncIterator
from anthropic import AsyncAnthropic
from .base import ChatProvider
from langgraph.types import RunnableConfig
from ..config import settings

def resolve_anthropic_api_key(config: RunnableConfig) -> str:
    """Return the Anthropic API key."""
    key: str = (config.get("configurable") or {}).get("anthropic_api_key") or settings.anthropic_api_key
    if not key:
        raise ValueError(
            "Anthropic API key not found. Set ANTHROPIC_API_KEY in .env or pass it via configurable."
        )
    return key

class AnthropicProvider(ChatProvider):
    def __init__(self, config: RunnableConfig):
        self.api_key = resolve_anthropic_api_key(config)
        self.client = AsyncAnthropic(api_key=self.api_key)


    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        # Anthropic uses a different format for system prompts
        system_prompt = ""
        if messages and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]

        async with self.client.messages.stream(
            model=model,
            system=system_prompt,
            messages=messages,
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield text
