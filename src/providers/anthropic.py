from typing import AsyncIterator
from anthropic import AsyncAnthropic
from .base import ChatProvider, UsageInfo
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
        super().__init__()
        self.api_key = resolve_anthropic_api_key(config)
        self.client = AsyncAnthropic(api_key=self.api_key)


    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        self.last_usage = UsageInfo()
        # Anthropic uses a different format for system prompts
        system_prompt = ""
        if messages and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]

        # max_tokens is required by the Anthropic SDK; default to a generous
        # cap when the caller doesn't supply one.
        kwargs_with_defaults = {"max_tokens": 4096, **kwargs}

        async with self.client.messages.stream(
            model=model,
            system=system_prompt,
            messages=messages,
            **kwargs_with_defaults,
        ) as stream:
            async for text in stream.text_stream:
                yield text

            # Available once the stream completes — has input_tokens,
            # output_tokens, and (when present) cache_read_input_tokens.
            final_message = await stream.get_final_message()
            usage = getattr(final_message, "usage", None)
            if usage is not None:
                self.last_usage = UsageInfo(
                    input_tokens=getattr(usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(usage, "output_tokens", 0) or 0,
                    cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                )
