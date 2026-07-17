"""Single policy gate for the business-hours restriction.

Mirrors the `capability_gate` pattern: the backend resolves the flag fresh
every turn (WorkingHoursService + the tenant's WorkingHours/Holiday/
BlockedPeriod rows in Tenant.defaultTimezone) and forwards it as
`state["within_business_hours"]`, so an admin editing the schedule takes
effect on the conversation's next turn — no deploy, no cache.

Outside hours every graph routes non-urgent turns to `faq_response`
(immediate cutoff, including in-progress flows: the checkpointed cart/booking
state survives and the flow resumes once hours reopen). `faq_response`
splices the tenant-editable `{AGENT_TYPE}_OUTSIDE_HOURS` prompt (default
below) so the model tells the customer the business is closed and stays
FAQ-only. Urgent intents (URGENT_INTENTS in shared_routing) are exempt.
"""

from __future__ import annotations

from ..graphs.state import AgentState

# Fallback instruction when the prompt catalog doesn't carry the per-agent
# `{AGENT_TYPE}_OUTSIDE_HOURS` type (older backend builds). Mirrors the
# seeded defaults in verveux-backend/src/ai-prompts/ai-prompts.constants.ts.
DEFAULT_OUTSIDE_HOURS_INSTRUCTION = (
    "IMPORTANTE — FUERA DE HORARIO DE ATENCIÓN:\n"
    "En este momento el negocio está fuera de su horario de atención.\n"
    "- Informa amablemente al cliente que estamos fuera de horario y que "
    "será atendido en el próximo horario de atención.\n"
    "- Solo puedes responder preguntas frecuentes e información general.\n"
    "- NO inicies pedidos, reservas, citas ni ningún otro flujo operativo.\n"
    "- Si el cliente pide algo operativo, indícale con amabilidad que lo "
    "retomaremos dentro del horario de atención."
)


def within_business_hours(state: AgentState) -> bool:
    """True when the tenant is within business hours (or doesn't enforce them).

    Absent (old checkpoints, or an older backend that doesn't send the flag)
    is treated as within-hours so behavior is unchanged unless the backend
    explicitly says otherwise.
    """
    value = state.get("within_business_hours")
    if value is None:
        return True
    return bool(value)
