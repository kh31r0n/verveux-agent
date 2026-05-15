from abc import ABC, abstractmethod
from typing import AsyncIterator

class ChatProvider(ABC):
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
