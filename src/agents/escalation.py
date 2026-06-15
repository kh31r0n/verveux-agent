import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from ..usage import make_usage_record
from .utils import language_instruction, resolve_prompt

logger = structlog.get_logger(__name__)

_ESCALATION_SYSTEM_PROMPT = """You are a security operations assistant handling an escalation.
Summarise the conversation and confirm to the user that the issue is being escalated to a human operator.
Be professional and reassuring. Include:
- A one-sentence summary of the issue
- Confirmation that a human operator will be notified
- What the user can expect next

{language_rule}
"""


async def escalation_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("escalation")

    provider = get_provider(config)
    model = resolve_model(config)
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

    lang_rule = language_instruction(state.get("language", "en"))
    agent_type = (state.get("agent_type") or "school").upper()
    escalation_key = f"{agent_type}_ESCALATION"
    escalation_prompt = resolve_prompt(config, escalation_key, _ESCALATION_SYSTEM_PROMPT)
    messages_payload = [
        {"role": "system", "content": escalation_prompt.format(language_rule=lang_rule)},
        {
            "role": "user",
            "content": "Conversation so far:\n\n" + "\n".join(conversation_summary),
        },
    ]

    generation = trace.generation(
        name="escalation_llm",
        model=model,
        input={"messages": messages_payload},
    )

    write = get_stream_writer()

    stream = provider.stream_chat(
        model=model,
        messages=messages_payload,
    )

    full_response = ""
    async for chunk in stream:
        write({"type": "token", "content": chunk})
        full_response += chunk

    generation.end(
        output=full_response,
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
        "turn_usage": [
            make_usage_record(node="escalation", provider=provider, model=model)
        ],
    }
