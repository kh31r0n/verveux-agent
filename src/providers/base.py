from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class UsageInfo:
    """Token-usage snapshot from a single provider call.

    Populated by each provider after `stream_chat` / `chat` returns. Captured
    in the agent state by the calling node and forwarded to NestJS for billing.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0


class ChatProvider(ABC):
    # Populated after each stream_chat / chat call. Nodes read it immediately
    # after draining the stream; subsequent calls overwrite it, so concurrent
    # calls per provider instance are unsupported (matches today's per-node
    # provider instantiation pattern).
    last_usage: UsageInfo

    def __init__(self) -> None:
        self.last_usage = UsageInfo()

    @property
    def name(self) -> str:
        """Short provider identifier, e.g. "openai". Used to label usage rows."""
        # Subclasses can override; default derives from the class name.
        cls = type(self).__name__
        return cls.removesuffix("Provider").lower()

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Yield text tokens from a streaming chat completion."""
        yield

    async def chat(
        self,
        messages: list[dict],
        model: str,
        **kwargs,
    ) -> str:
        """Return a single chat completion."""
        response = ""
        async for chunk in self.stream_chat(messages, model, **kwargs):
            response += chunk
        return response

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Generate embeddings. Optional; raises NotImplementedError by default."""
        raise NotImplementedError("Embedding not supported by this provider")
