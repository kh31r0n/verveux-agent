import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation

logger = structlog.get_logger(__name__)

_ESCALATION_SYSTEM_PROMPT = """You are a security operations assistant handling an escalation.
Summarise the conversation and confirm to the user that the issue is being escalated to a human operator.
Be professional and reassuring. Include:
- A one-sentence summary of the issue
- Confirmation that a human operator will be notified
- What the user can expect next

Respond in the same language the user writes in.
"""


async def escalation_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("escalation")

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="escalation_node",
        metadata={"thread_id": thread_id, "node": "escalation"},
    )

    # Build conversation summary for the escalation message
    conversation_summary: list[str] = []
    for msg in state["messages"]:
        role = "User" if getattr(msg, "type", "") == "human" else "Assistant"
        content = msg.content if hasattr(msg, "content") else str(msg)
        conversation_summary.append(f"{role}: {content}")

    messages_payload = [
        {"role": "system", "content": _ESCALATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Conversation so far:\n\n" + "\n".join(conversation_summary),
        },
    ]

    generation = trace.generation(
        name="escalation_llm",
        model="gpt-4o",
        input={"messages": messages_payload},
    )

    write = get_stream_writer()

    stream = await client.chat.completions.create(
        model="gpt-5",
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
            write({"type": "token", "content": delta})
            full_response += delta
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    generation.end(
        output=full_response,
        usage={"input": prompt_tokens, "output": completion_tokens},
    )

    # Build structured escalation payload for NestJS to forward
    escalation_payload: dict = {
        "target": "escalation",
        "thread_id": thread_id,
        "summary": "\n".join(conversation_summary[-6:]),  # last 3 turns
        "messages": [
            {
                "role": "user" if getattr(m, "type", "") == "human" else "assistant",
                "content": m.content if hasattr(m, "content") else str(m),
            }
            for m in state["messages"]
        ],
    }

    logger.info("escalation_triggered", thread_id=thread_id)

    return {
        "messages": [
            AIMessage(
                content=full_response,
                additional_kwargs={"escalation_payload": escalation_payload},
            )
        ],
    }
