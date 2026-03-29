import json
from typing import Literal

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation

logger = structlog.get_logger(__name__)

_WORKFLOW_DECISION_PROMPT = """You are a workflow dispatcher agent for a security operations platform.
Analyse the user's request and determine:
1. Which workflow system to use: "n8n" or "airflow"
2. The workflow identifier (webhook_id for n8n, dag_id for Airflow)
3. The input parameters to pass

Respond with a single JSON object only (no markdown, no extra text):
{
  "target": "n8n" | "airflow",
  "webhook_id": "<string — only if target is n8n, else omit>",
  "dag_id": "<string — only if target is airflow, else omit>",
  "parameters": { ... } | "conf": { ... },
  "description": "<one sentence describing what this workflow will do>"
}

If you cannot determine the appropriate workflow, respond:
{"error": "<explanation of why you cannot proceed>"}
"""

_REJECTION_PROMPT = """You are a security operations assistant.
The user declined to trigger a workflow action. Acknowledge this politely and let them know the action was cancelled.
Respond in the same language the user writes in. Keep your response concise.
"""


async def workflow_node(
    state: AgentState,
    config: RunnableConfig,
) -> Command[Literal["orchestrator"]]:
    record_node_invocation("workflow")

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="workflow_node",
        metadata={"thread_id": thread_id, "node": "workflow"},
    )

    # Build messages for workflow decision
    messages_payload = [{"role": "system", "content": _WORKFLOW_DECISION_PROMPT}]
    for msg in state["messages"]:
        if hasattr(msg, "type"):
            role = "assistant" if msg.type == "ai" else "user"
        else:
            role = "user"
        content = msg.content if hasattr(msg, "content") else str(msg)
        messages_payload.append({"role": role, "content": content})

    generation = trace.generation(
        name="workflow_decision_llm",
        model="gpt-4o",
        input={"messages": messages_payload},
    )

    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages_payload,
        stream=True,
    )

    full_response = ""
    prompt_tokens = 0
    completion_tokens = 0

    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_response += delta
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    generation.end(
        output=full_response,
        usage={"input": prompt_tokens, "output": completion_tokens},
    )

    try:
        decision = json.loads(full_response.strip())
    except json.JSONDecodeError:
        logger.warning("workflow_json_parse_failed", thread_id=thread_id)
        error_msg = AIMessage(content="I was unable to determine the appropriate workflow for your request.")
        return Command(goto="orchestrator", update={"messages": [error_msg]})

    if "error" in decision:
        error_msg = AIMessage(content=decision["error"])
        return Command(goto="orchestrator", update={"messages": [error_msg]})

    target: str = decision.get("target", "n8n")
    description: str = decision.get("description", "trigger a workflow")

    if target == "n8n":
        action_payload: dict = {
            "question": f"Do you approve triggering this n8n workflow?\n\n{description}",
            "workflow_type": "n8n",
            "webhook_id": decision.get("webhook_id", ""),
            "parameters": decision.get("parameters", {}),
        }
    else:
        action_payload = {
            "question": f"Do you approve triggering this Airflow DAG?\n\n{description}",
            "workflow_type": "airflow",
            "dag_id": decision.get("dag_id", ""),
            "conf": decision.get("conf", decision.get("parameters", {})),
        }

    # Pause and surface approval request — no side effects before this line
    approved: bool = interrupt(action_payload)

    if approved:
        # Build the structured payload for NestJS to forward
        if target == "n8n":
            trigger_payload: dict = {
                "target": "n8n",
                "webhook_id": action_payload["webhook_id"],
                "parameters": action_payload["parameters"],
            }
        else:
            trigger_payload = {
                "target": "airflow",
                "dag_id": action_payload["dag_id"],
                "conf": action_payload["conf"],
            }

        confirmation_msg = AIMessage(
            content=f"Workflow approved. Triggering: {description}",
            additional_kwargs={"trigger_payload": trigger_payload},
        )
        return Command(goto="orchestrator", update={"messages": [confirmation_msg]})
    else:
        # Inform user of cancellation via LLM
        rejection_stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _REJECTION_PROMPT},
                {
                    "role": "user",
                    "content": f"The user declined to trigger: {description}",
                },
            ],
            stream=True,
        )

        rejection_text = ""
        async for chunk in rejection_stream:
            rejection_text += chunk.choices[0].delta.content or ""

        rejection_msg = AIMessage(content=rejection_text)
        return Command(goto="orchestrator", update={"messages": [rejection_msg]})
