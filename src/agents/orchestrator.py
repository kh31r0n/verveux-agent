import json
from typing import Literal

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are the orchestrator agent of a security operations AI assistant (Helena).

Your job is to analyse the user's request and decide which specialist agent to invoke next.

Available agents:
- **rag**: Retrieves relevant information from the internal knowledge base (documents, runbooks, past incidents).
  Use this when the user asks a knowledge question, needs context, or you need background before acting.
- **workflow**: Triggers n8n or Apache Airflow workflows on behalf of the user.
  Use this when the user explicitly asks to run a workflow, create a ticket, escalate to an external system, or automate a task.
- **escalate**: Escalates the issue to a human operator (e.g. on-call engineer, security analyst).
  Use this when the issue requires human judgement, is critical, or all automated remediation options are exhausted.
- **end**: End the conversation turn.
  Use this when the user's request has been fully answered and no further action is needed.

Rules:
- Respond in the same language the user writes in.
- Your response must be a single JSON object on one line — no markdown code fences, no extra text.
- JSON schema: {"response": "<natural language message to user>", "next": "<rag|workflow|escalate|end>"}
- The "response" field is shown to the user. Be helpful and concise.
- If you need more information to decide, set next=end and ask the user in "response".
"""


async def orchestrator_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("orchestrator")

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)

    langfuse = get_langfuse()
    thread_id: str = state.get("thread_id", "unknown")

    trace = langfuse.trace(
        name="orchestrator",
        metadata={"thread_id": thread_id, "node": "orchestrator"},
    )
    generation = trace.generation(
        name="orchestrator_llm",
        model="gpt-4o",
        input={"messages": [m.content if hasattr(m, "content") else str(m) for m in state["messages"]]},
    )

    messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in state["messages"]:
        if hasattr(msg, "type"):
            role = "assistant" if msg.type == "ai" else "user"
        else:
            role = "user"
        content = msg.content if hasattr(msg, "content") else str(msg)
        messages_payload.append({"role": role, "content": content})

    write = get_stream_writer()

    stream = await client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=messages_payload,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_response = ""
    prompt_tokens = 0
    completion_tokens = 0

    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else ""
        if delta:
            full_response += delta
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    generation.end(
        output=full_response,
        usage={"input": prompt_tokens, "output": completion_tokens},
    )

    # Parse JSON response — extract display text and routing decision separately
    try:
        parsed = json.loads(full_response.strip())
        response_text: str = parsed.get("response", full_response)
        next_node: str = parsed.get("next", "end")
    except (json.JSONDecodeError, AttributeError):
        logger.warning("orchestrator_json_parse_failed", raw=full_response[:200])
        response_text = full_response
        next_node = "end"

    valid_next: set[str] = {"rag", "workflow", "escalate", "end"}
    if next_node not in valid_next:
        next_node = "end"

    # Stream response_text token-by-token only when ending the turn (visible to user).
    # When routing to another node, the downstream node handles streaming.
    if next_node == "end":
        for char in response_text:
            write({"type": "token", "content": char})

    # Store natural language response for the user; routing decision goes into state["next"]
    return {
        "messages": [AIMessage(content=response_text)],
        "next": next_node,
    }


def route_decision(
    state: AgentState,
) -> Literal["rag", "workflow", "escalation", "end"]:
    next_node = state.get("next", "end")
    mapping: dict[str, Literal["rag", "workflow", "escalation", "end"]] = {
        "rag": "rag",
        "workflow": "workflow",
        "escalate": "escalation",
        "end": "end",
    }
    return mapping.get(next_node, "end")
