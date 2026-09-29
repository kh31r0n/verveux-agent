"""appointment_reschedule — pick the appointment to move and seed the booking flow.

Selecting the source booking sets ``reschedule_source_id`` and reuses the
existing ``availability_lookup → reservation_propose → confirmation`` path
for the new slot. The final ``confirmation`` node detects ``reschedule_source_id``
and dispatches to /internal/appointments/:id/reschedule instead of creating a
new hold.

To keep this node small, the actual reschedule HTTP call lives in
``reschedule_execute_node`` below.
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
    list_active_appointments_for_contact,
    reschedule_appointment,
)

logger = structlog.get_logger(__name__)


def _parse_choice(text: str, appts: list[dict]) -> str | None:
    digits = re.match(r"\s*(\d{1,2})\b", text)
    if digits:
        idx = int(digits.group(1)) - 1
        if 0 <= idx < len(appts):
            return appts[idx].get("id")
    return None


async def appointment_reschedule_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("appointment_reschedule")

    tenant_id = state.get("tenant_id", "")
    contact_id = state.get("contact_id", "")

    source_id = state.get("reschedule_source_id")
    if source_id:
        # Already chose the source; let the downstream availability /
        # propose nodes run.
        return {"booking_intent": "reschedule"}

    try:
        appts = await list_active_appointments_for_contact(contact_id, tenant_id)
    except Exception as exc:
        logger.error("appointment_reschedule_lookup_failed", error=str(exc))
        return {
            "messages": [
                AIMessage(
                    content="No pude consultar tus citas. Inténtalo en unos minutos."
                )
            ],
        }

    if not appts:
        return {
            "booking_intent": None,
            "messages": [
                AIMessage(content="No encuentro citas activas para reagendar.")
            ],
        }

    if len(appts) == 1:
        appt = appts[0]
        return {
            "reschedule_source_id": appt.get("id"),
            "appointment_type_id": appt.get("appointmentTypeId"),
            "appointment_type_name": (
                appt.get("appointmentType", {}).get("name")
                if isinstance(appt.get("appointmentType"), dict)
                else None
            ),
            "booking_intent": "reschedule",
            "messages": [
                AIMessage(
                    content="Voy a buscar nuevos horarios para tu cita. Un momento..."
                )
            ],
        }

    user_reply = interrupt(
        {
            "type": "appointment_reschedule_choice",
            "appointments": [
                {"id": a.get("id"), "startsAt": a.get("startsAt")} for a in appts
            ],
            "prompt": "¿Cuál cita quieres reagendar?\n"
            + "\n".join(
                f"{i + 1}. {a.get('startsAt', '')}" for i, a in enumerate(appts)
            ),
        }
    )
    target_id = _parse_choice(user_reply or "", appts)
    if not target_id:
        return {
            "messages": [
                AIMessage(
                    content="No identifiqué la cita. Responde con el número (1, 2, ...) de la que quieres reagendar."
                )
            ],
        }
    chosen = next((a for a in appts if a.get("id") == target_id), None)
    return {
        "reschedule_source_id": target_id,
        "appointment_type_id": (
            chosen.get("appointmentTypeId") if chosen else None
        ),
        "booking_intent": "reschedule",
        "messages": [
            AIMessage(content="Voy a buscar nuevos horarios. Un momento...")
        ],
    }


async def reschedule_execute_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """Apply the chosen slot to the source appointment.

    Routed after ``reservation_propose`` when ``reschedule_source_id`` is set.
    Mirrors ``confirmation_node``'s error handling for SLOT_TAKEN.
    """
    record_node_invocation("reschedule_execute")

    tenant_id = state.get("tenant_id", "")
    conversation_id = state.get("conversation_id", "")
    source_id = state.get("reschedule_source_id")
    chosen = state.get("chosen_slot") or {}

    if not (tenant_id and source_id and chosen):
        return {
            "messages": [
                AIMessage(
                    content="Me faltan datos para reagendar. ¿Volvemos a buscar disponibilidad?"
                )
            ],
        }

    resources = [
        {"kind": r.get("kind"), "resourceId": r.get("resourceId")}
        for r in chosen.get("resources", [])
        if r.get("kind") and r.get("resourceId")
    ]

    try:
        await reschedule_appointment(
            appointment_id=source_id,
            tenant_id=tenant_id,
            starts_at_iso=chosen.get("startsAt", ""),
            ends_at_iso=chosen.get("endsAt", ""),
            resources=resources,
            conversation_id=conversation_id,
        )
    except HTTPStatusError as exc:
        body: dict = {}
        try:
            body = exc.response.json()
        except Exception:
            pass
        if exc.response.status_code == 409 and body.get("code") == "SLOT_TAKEN":
            return {
                "slot_conflict": True,
                "chosen_slot": None,
                "messages": [
                    AIMessage(
                        content="Justo acaban de tomar ese horario. Déjame buscar otras opciones."
                    )
                ],
            }
        logger.error(
            "appointment_reschedule_failed",
            status=exc.response.status_code,
            code=body.get("code"),
        )
        return {
            "messages": [
                AIMessage(
                    content="No pude reagendar esa cita. Inténtalo en unos minutos."
                )
            ],
        }
    except Exception as exc:
        logger.error("appointment_reschedule_unexpected", error=str(exc))
        return {
            "messages": [
                AIMessage(
                    content="No pude reagendar esa cita. Inténtalo en unos minutos."
                )
            ],
        }

    return {
        "booking_confirmed": True,
        "reschedule_source_id": None,
        "booking_intent": None,
        "messages": [
            AIMessage(content="Tu cita quedó reagendada. Te confirmaremos por aquí.")
        ],
    }
