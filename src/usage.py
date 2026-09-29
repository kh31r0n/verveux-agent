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
    # Optional provenance, set only by callers that pass it (the email agent).
    # Existing rows keep their exact shape: the backend's DTOs reject keys they
    # do not declare.
    latency_ms: int
    prompt_key: str
    prompt_id: str
    prompt_version: str
    prompt_sha: str


def make_usage_record(
    *,
    node: str,
    provider: ChatProvider,
    model: str,
    usage: UsageInfo | None = None,
    latency_ms: int | None = None,
    provenance: dict | None = None,
) -> InvocationUsage:
    """Build an InvocationUsage dict from a provider's most recent call.

    Pass `usage` explicitly when the provider instance is shared (rare today —
    nodes instantiate per-call). Otherwise reads `provider.last_usage`.
    `provenance` is the dict from `resolve_prompt_provenance`.
    """
    u = usage or provider.last_usage
    record = InvocationUsage(
        node=node,
        provider=provider.name,
        model=model,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cached_input_tokens=u.cached_input_tokens,
        reasoning_tokens=u.reasoning_tokens,
    )
    if latency_ms is not None:
        record["latency_ms"] = latency_ms
    if provenance:
        record.update(provenance)  # type: ignore[typeddict-item]
    return record
