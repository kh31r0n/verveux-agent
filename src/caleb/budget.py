"""Budget reservation, settlement, and ordered degradation hierarchy for Caleb."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CalebDegradationLevel = Literal[
    "NORMAL",
    "REDUCE_TOP_K_AND_QUERIES",
    "DETERMINISTIC_SCORING_ONLY",
    "HARD_STOP_EXTERNAL_TOOLS",
]


@dataclass
class CalebBudgetEvaluation:
    level: CalebDegradationLevel
    utilization_ratio: float
    remaining_usd: float
    allowed: bool
    reason: str | None = None


def evaluate_budget_state(
    *,
    internal_cost_cap_usd: float,
    reserved_spend_usd: float,
    settled_spend_usd: float,
    proposed_spend_usd: float,
) -> CalebBudgetEvaluation:
    cap = max(internal_cost_cap_usd, 0.0001)
    committed = max(reserved_spend_usd, settled_spend_usd)
    projected = committed + max(0.0, proposed_spend_usd)
    utilization_ratio = round(projected / cap, 4)
    remaining_usd = round(max(0.0, cap - committed), 6)

    if projected > cap or utilization_ratio >= 0.95:
        return CalebBudgetEvaluation(
            level="HARD_STOP_EXTERNAL_TOOLS",
            utilization_ratio=utilization_ratio,
            remaining_usd=remaining_usd,
            allowed=False,
            reason=(
                f"Presupuesto interno alcanzado ({utilization_ratio * 100:.1f}% de "
                f"${cap:.2f} USD). Se detienen llamadas externas pagas y se sintetiza "
                "el reporte con la evidencia acumulada."
            ),
        )

    if utilization_ratio >= 0.85:
        return CalebBudgetEvaluation(
            level="DETERMINISTIC_SCORING_ONLY",
            utilization_ratio=utilization_ratio,
            remaining_usd=remaining_usd,
            allowed=True,
            reason=(
                f"Presupuesto interno al {utilization_ratio * 100:.1f}%: se usa "
                "scoring determinístico por entidad y se reserva saldo para el reporte final."
            ),
        )

    if utilization_ratio >= 0.70:
        return CalebBudgetEvaluation(
            level="REDUCE_TOP_K_AND_QUERIES",
            utilization_ratio=utilization_ratio,
            remaining_usd=remaining_usd,
            allowed=True,
            reason=(
                f"Presupuesto interno al {utilization_ratio * 100:.1f}%: se reduce "
                "top-K de investigación profunda al 50% y se recortan consultas secundarias."
            ),
        )

    return CalebBudgetEvaluation(
        level="NORMAL",
        utilization_ratio=utilization_ratio,
        remaining_usd=remaining_usd,
        allowed=True,
    )


@dataclass
class CalebBudgetTracker:
    """In-memory / graph-state budget tracker mirroring NestJS StudySpendLedger."""

    internal_cost_cap_usd: float
    reserved_spend_usd: float = 0.0
    settled_spend_usd: float = 0.0
    reservations: dict[str, float] = field(default_factory=dict)
    settlements: dict[str, float] = field(default_factory=dict)
    degradations: list[dict[str, str | float]] = field(default_factory=list)

    def reserve(self, operation_key: str, estimated_usd: float) -> CalebBudgetEvaluation:
        if operation_key in self.reservations:
            return evaluate_budget_state(
                internal_cost_cap_usd=self.internal_cost_cap_usd,
                reserved_spend_usd=self.reserved_spend_usd,
                settled_spend_usd=self.settled_spend_usd,
                proposed_spend_usd=0.0,
            )

        evaluation = evaluate_budget_state(
            internal_cost_cap_usd=self.internal_cost_cap_usd,
            reserved_spend_usd=self.reserved_spend_usd,
            settled_spend_usd=self.settled_spend_usd,
            proposed_spend_usd=estimated_usd,
        )
        if not evaluation.allowed:
            self._record_degradation(evaluation)
            return evaluation

        self.reservations[operation_key] = round(estimated_usd, 6)
        self.reserved_spend_usd = round(self.reserved_spend_usd + estimated_usd, 6)
        if evaluation.level != "NORMAL":
            self._record_degradation(evaluation)
        return evaluation

    def settle(self, operation_key: str, actual_usd: float) -> None:
        if operation_key in self.settlements:
            return
        reserved = self.reservations.get(operation_key, 0.0)
        delta = actual_usd - reserved
        self.settlements[operation_key] = round(actual_usd, 6)
        self.settled_spend_usd = round(self.settled_spend_usd + actual_usd, 6)
        self.reserved_spend_usd = round(max(0.0, self.reserved_spend_usd + delta), 6)

    def _record_degradation(self, evaluation: CalebBudgetEvaluation) -> None:
        if not evaluation.reason:
            return
        if any(d.get("level") == evaluation.level for d in self.degradations):
            return
        self.degradations.append(
            {
                "level": evaluation.level,
                "utilizationRatio": evaluation.utilization_ratio,
                "reason": evaluation.reason,
            }
        )
