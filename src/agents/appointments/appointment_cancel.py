"""appointment_cancel — list active bookings, pick one, then call the backend.

Two phases collapse into one node:
  1. If `cancellation_target_id` is unset, fetch active appointments for the
     contact, present them, and either pick automatically (single match) or
     ask the user to choose.
  2. If `cancellation_target_id` is set after a user reply, hit
     /internal/appointments/:id/cancel.
"""

import re

import structlog
from httpx import HTTPStatusError
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from ...graphs.state import AgentState
from ...observability import record_node_invocation
from ..backend_client import (
    cancel_appointment,
    list_active_appointments_for_contact,
)

logger = structlog.get_logger(__name__)


def _format_appointment_line(idx: int, appt: dict) -> str:
    starts = appt.get("startsAt", "")
    type_name = ""
    if isinstance(appt.get("appointmentType"), dict):
        type_name = appt["appointmentType"].get("name", "")
    return f"{idx + 1}. {type_name or 'Cita'} — {starts}"


def _last_user_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages") or []):
        if getattr(msg, "type", "") == "human":
            return msg.content
    return ""


def _parse_choice(text: str, appts: list[dict]) -> str | None:
    digits = re.match(r"\s*(\d{1,2})\b", text)
    if digits:
        idx = int(digits.group(1)) - 1
        if 0 <= idx < len(appts):
            return appts[idx].get("id")
    return None


async def appointment_cancel_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("appointment_cancel")

    tenant_id = state.get("tenant_id", "")
    conversation_id = state.get("conversation_id", "")
    contact_id = state.get("contact_id", "")

    try:
        appts = await list_active_appointments_for_contact(contact_id, tenant_id)
    except Exception as exc:
        logger.error("appointment_cancel_lookup_failed", error=str(exc))
        return {
            "messages": [
                AIMessage(
                    content="No pude consultar tus citas en este momento. Inténtalo en unos minutos."
                )
            ],
        }

    if not appts:
        return {
            "booking_intent": None,
            "messages": [
                AIMessage(content="No encuentro citas activas a tu nombre.")
            ],
        }

    target_id = state.get("cancellation_target_id")
    if not target_id:
        if len(appts) == 1:
            target_id = appts[0].get("id")
        else:
            user_reply = interrupt(
                {
                    "type": "appointment_cancel_choice",
                    "appointments": [
                        {"id": a.get("id"), "startsAt": a.get("startsAt")}
                        for a in appts
                    ],
                    "prompt": "Tienes varias citas activas. ¿Cuál quieres cancelar?\n"
                    + "\n".join(_format_appointment_line(i, a) for i, a in enumerate(appts)),
                }
            )
            target_id = _parse_choice(user_reply or "", appts)
            if not target_id:
                return {
                    "messages": [
                        AIMessage(
                            content="No identifiqué la cita. Responde con el número (1, 2, ...) de la que quieres cancelar."
                        )
                    ],
                }

    try:
        await cancel_appointment(
            target_id,
            tenant_id,
            reason="cancelled-by-customer",
            conversation_id=conversation_id,
        )
    except HTTPStatusError as exc:
        logger.error("appointment_cancel_failed", status=exc.response.status_code)
        return {
            "messages": [
                AIMessage(
                    content="No pude cancelar esa cita; ya podría estar cerrada. Verifícalo o contáctanos."
                )
            ],
        }
    except Exception as exc:
        logger.error("appointment_cancel_unexpected", error=str(exc))
        return {
            "messages": [
                AIMessage(
                    content="Tuve un problema cancelando esa cita. Inténtalo en unos minutos."
                )
            ],
        }

    return {
        "cancellation_target_id": target_id,
        "booking_intent": None,
        "messages": [
            AIMessage(content="Tu cita quedó cancelada. ¿Algo más en lo que te ayude?")
        ],
    }
