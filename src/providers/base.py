from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, TypeVar

from pydantic import BaseModel, ValidationError

from ..json_utils import strip_json_fences
from .errors import StructuredOutputError

Schema = TypeVar("Schema", bound=BaseModel)


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

    async def generate_structured(
        self,
        messages: list[dict],
        model: str,
        schema: type[Schema],
        *,
        thinking_budget: int | None = None,
        temperature: float | None = 0.0,
    ) -> Schema:
        """One completion validated against ``schema``; ``last_usage`` holds its tokens.

        This default asks for JSON through the prompt and validates the text, so
        every provider supports it. Providers with a native schema mode (Gemini)
        override it; ``thinking_budget`` is ignored by providers without one.
        """
        text = await self.chat(messages, model)
        return parse_structured(text, schema)

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Generate embeddings. Optional; raises NotImplementedError by default."""
        raise NotImplementedError("Embedding not supported by this provider")


def parse_structured(text: str, schema: type[Schema]) -> Schema:
    """Validate model text as ``schema``, classifying every failure."""
    body = strip_json_fences(text or "")
    if not body:
        raise StructuredOutputError(
            f"the model returned no text for {schema.__name__}", kind="empty"
        )
    try:
        return schema.model_validate_json(body)
    except ValidationError as exc:
        kind = (
            "invalid_json"
            if any(err.get("type") == "json_invalid" for err in exc.errors())
            else "schema_mismatch"
        )
        raise StructuredOutputError(
            f"the model's JSON does not match {schema.__name__}: {exc}", kind=kind
        ) from exc
