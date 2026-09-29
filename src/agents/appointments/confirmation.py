"""confirmation — create the PENDING_HOLD and immediately confirm it.

Two backend calls in sequence: POST /internal/appointments/holds then POST
/internal/appointments/:id/confirm. On 409 with code SLOT_TAKEN the node sets
``slot_conflict = True`` so the graph routes back to availability_lookup.
"""

import structlog
from httpx import HTTPStatusError
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ...graphs.state import AgentState
from ...observability import record_node_invocation
from ..backend_client import confirm_appointment, create_appointment_hold

logger = structlog.get_logger(__name__)


def _resources_payload(slot: dict) -> list[dict]:
    out: list[dict] = []
    for r in slot.get("resources", []) or []:
        kind = r.get("kind")
        resource_id = r.get("resourceId")
        if not (kind and resource_id):
            continue
        out.append({"kind": kind, "resourceId": resource_id})
    return out


async def confirmation_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("confirmation")

    tenant_id = state.get("tenant_id", "")
    contact_id = state.get("contact_id", "")
    appointment_type_id = state.get("appointment_type_id")
    chosen = state.get("chosen_slot") or {}
    customer_data = state.get("collected_customer_data") or {}
    thread_id = state.get("thread_id", "")
    conversation_id = state.get("conversation_id", "")

    if not (tenant_id and contact_id and appointment_type_id and chosen):
        return {
            "messages": [
                AIMessage(
                    content="No tengo todos los datos para confirmar la cita. ¿Volvemos al inicio?"
                )
            ],
        }

    starts_at = chosen.get("startsAt", "")
    ends_at = chosen.get("endsAt", "")
    resources = _resources_payload(chosen)

    try:
        hold = await create_appointment_hold(
            tenant_id=tenant_id,
            contact_id=contact_id,
            appointment_type_id=appointment_type_id,
            starts_at_iso=starts_at,
            ends_at_iso=ends_at,
            resources=resources,
            customer_data=customer_data,
            thread_id=thread_id,
            conversation_id=conversation_id,
        )
    except HTTPStatusError as exc:
        body: dict = {}
        try:
            body = exc.response.json()
        except Exception:
            pass
        code = body.get("code", "")
        if exc.response.status_code == 409 and code == "SLOT_TAKEN":
            return {
                "slot_conflict": True,
                "chosen_slot": None,
                "messages": [
                    AIMessage(
                        content="Justo acaban de reservar ese horario. Déjame buscar otras opciones."
                    )
                ],
            }
        logger.error(
            "appointment_hold_failed",
            status=exc.response.status_code,
            code=code,
        )
        return {
            "messages": [
                AIMessage(
                    content="Tuve un problema reservando ese horario. Inténtalo en unos minutos."
                )
            ],
        }
    except Exception as exc:
        logger.error("appointment_hold_unexpected", error=str(exc))
        return {
            "messages": [
                AIMessage(
                    content="Tuve un problema reservando ese horario. Inténtalo en unos minutos."
                )
            ],
        }

    appointment_id = hold.get("id", "")
    hold_expires_at = hold.get("holdExpiresAt", "")

    confirmed_msg = "Tu cita quedó CONFIRMADA. Te enviaré el recordatorio por aquí."
    try:
        await confirm_appointment(
            appointment_id, tenant_id, conversation_id=conversation_id
        )
    except Exception as exc:
        # The hold succeeded but confirm failed. The hold sweeper will clear
        # the reservation in a few minutes; ask the user to retry confirming.
        logger.warning("appointment_confirm_failed", error=str(exc))
        confirmed_msg = (
            "Reservé el horario, pero no pude confirmarlo aún. Responde 'confirmar' "
            "para intentarlo de nuevo en unos segundos."
        )

    return {
        "reservation_appointment_id": appointment_id,
        "hold_expires_at": hold_expires_at,
        "booking_confirmed": True,
        "slot_conflict": False,
        "messages": [AIMessage(content=confirmed_msg)],
    }
