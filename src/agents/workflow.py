import json
from typing import Literal

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from ..graphs.state import AgentState
from ..json_utils import strip_json_fences
from ..providers.registry import get_provider, resolve_model
from ..observability import get_langfuse, record_node_invocation
from .utils import language_instruction

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
{language_rule} Keep your response concise.
"""


async def workflow_node(
    state: AgentState,
    config: RunnableConfig,
) -> Command[Literal["orchestrator"]]:
    record_node_invocation("workflow")

    provider = get_provider(config)
    model = resolve_model(config)
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
        model=model,
        input={"messages": messages_payload},
    )

    stream = provider.stream_chat(
        model=model,
        messages=messages_payload,
    )

    full_response = ""
    async for chunk in stream:
        full_response += chunk

    generation.end(
        output=full_response,
    )

    try:
        decision = json.loads(strip_json_fences(full_response))
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
        rejection_text = await provider.chat(
            model=model,
            messages=[
                {"role": "system", "content": _REJECTION_PROMPT.format(language_rule=language_instruction(state.get("language", "en")))},
                {
                    "role": "user",
                    "content": f"The user declined to trigger: {description}",
                },
            ],
        )

        rejection_msg = AIMessage(content=rejection_text)
        return Command(goto="orchestrator", update={"messages": [rejection_msg]})
