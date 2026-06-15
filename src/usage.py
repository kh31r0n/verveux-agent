"""Per-invocation usage helper.

Each LangGraph node appends one dict per provider call to `state["turn_usage"]`.
NestJS consumes the list on the SSE `done` event and persists it as
`AiInvocationUsage` rows (with pricing snapshots and idempotency).
"""
from __future__ import annotations

from typing import TypedDict

from .providers.base import ChatProvider, UsageInfo


class InvocationUsage(TypedDict, total=False):
    """One row per LLM call, accumulated in AgentState.turn_usage."""

    node: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int


def make_usage_record(
    *,
    node: str,
    provider: ChatProvider,
    model: str,
    usage: UsageInfo | None = None,
) -> InvocationUsage:
    """Build an InvocationUsage dict from a provider's most recent call.

    Pass `usage` explicitly when the provider instance is shared (rare today —
    nodes instantiate per-call). Otherwise reads `provider.last_usage`.
    """
    u = usage or provider.last_usage
    return InvocationUsage(
        node=node,
        provider=provider.name,
        model=model,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cached_input_tokens=u.cached_input_tokens,
        reasoning_tokens=u.reasoning_tokens,
    )
