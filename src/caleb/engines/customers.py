"""Current Customers portfolio analyzer (CURRENT_CUSTOMERS) — ABC/Pareto & inactivity risk."""

from __future__ import annotations

from typing import Any


def analyze_customer_portfolio(
    customers: list[dict[str, Any]],
    customer_base_config: dict[str, Any],
) -> dict[str, Any]:
    """Classifies existing customers (from One CRM/Orders and/or CSV/Excel) into ABC tiers and inactivity risk.

    - Supports ``source="ONE_CRM"``, ``source="CSV_UPLOAD"``, or merged records.
    - Never invents missing order dates or revenue: missing fields stay ``"sin dato"``.
    """
    thresholds = customer_base_config.get("abcThresholds") or {
        "tierAPct": 80.0,
        "tierBPct": 15.0,
    }
    tier_a_pct = float(thresholds.get("tierAPct", 80.0))
    tier_b_pct = float(thresholds.get("tierBPct", 15.0))
    if tier_a_pct <= 0 or (tier_a_pct + tier_b_pct) >= 100.0:
        raise ValueError("abcThresholds (tierAPct + tierBPct) must be > 0 and < 100")

    warn_days = int(customer_base_config.get("inactivityDaysWarning", 60))
    churn_days = int(customer_base_config.get("inactivityDaysChurned", 180))

    # Deduplicate by customerId / taxId / normalized name when merging One + CSV
    merged: dict[str, dict[str, Any]] = {}
    for row in customers:
        key = str(
            row.get("customerId")
            or row.get("taxId")
            or str(row.get("name") or "").strip().lower()
        )
        if not key:
            continue
        rev = float(row.get("totalRevenue") or 0.0)
        orders_cnt = int(row.get("ordersCount") or 0)
        days_since = row.get("daysSinceLastOrder")
        src = str(row.get("source") or "ONE_CRM")

        if key not in merged:
            merged[key] = {
                "customerId": row.get("customerId") or key,
                "name": row.get("name") or "sin dato",
                "city": row.get("city") or "sin dato",
                "totalRevenue": rev,
                "ordersCount": orders_cnt,
                "daysSinceLastOrder": (
                    int(days_since)
                    if isinstance(days_since, (int, float))
                    else "sin dato"
                ),
                "sources": [src],
            }
        else:
            existing = merged[key]
            existing["totalRevenue"] = round(float(existing["totalRevenue"]) + rev, 2)
            existing["ordersCount"] = int(existing["ordersCount"]) + orders_cnt
            if isinstance(days_since, (int, float)):
                prev_days = existing["daysSinceLastOrder"]
                if prev_days == "sin dato" or int(days_since) < int(prev_days):
                    existing["daysSinceLastOrder"] = int(days_since)
            if src not in existing["sources"]:
                existing["sources"].append(src)

    sorted_customers = sorted(
        merged.values(),
        key=lambda c: float(c.get("totalRevenue", 0.0)),
        reverse=True,
    )

    grand_total = sum(float(c["totalRevenue"]) for c in sorted_customers)
    cumulative_rev = 0.0

    classified: list[dict[str, Any]] = []
    tier_counts = {"A": 0, "B": 0, "C": 0}
    risk_counts = {
        "ACTIVO": 0,
        "EN_RIESGO": 0,
        "INACTIVO_CHURN": 0,
        "SIN_HISTORIAL": 0,
    }

    for idx, cust in enumerate(sorted_customers):
        rev = float(cust["totalRevenue"])
        share_pct = round((rev / grand_total) * 100.0, 2) if grand_total > 0 else 0.0
        prev_cum_pct = (
            (cumulative_rev / grand_total) * 100.0 if grand_total > 0 else 100.0
        )
        cumulative_rev += rev
        cum_pct = round(
            (cumulative_rev / grand_total) * 100.0 if grand_total > 0 else 0.0, 2
        )

        if grand_total <= 0 or rev <= 0:
            abc_tier = "C"
        elif idx == 0 or prev_cum_pct < tier_a_pct:
            abc_tier = "A"
        elif prev_cum_pct < (tier_a_pct + tier_b_pct):
            abc_tier = "B"
        else:
            abc_tier = "C"

        tier_counts[abc_tier] += 1

        days = cust["daysSinceLastOrder"]
        if days == "sin dato" or days is None:
            inactivity_status = "SIN_HISTORIAL"
        elif int(days) >= churn_days:
            inactivity_status = "INACTIVO_CHURN"
        elif int(days) >= warn_days:
            inactivity_status = "EN_RIESGO"
        else:
            inactivity_status = "ACTIVO"

        risk_counts[inactivity_status] += 1

        if abc_tier == "A" and inactivity_status in ("EN_RIESGO", "INACTIVO_CHURN"):
            action = "PRIORIDAD_1_REACTIVACION_EJECUTIVA"
        elif abc_tier == "A":
            action = "DESARROLLO_CUENTA_CLAVE_CROSS_SELL"
        elif abc_tier == "B" and inactivity_status != "ACTIVO":
            action = "CAMPANA_RETENCION_COMERCIAL"
        else:
            action = "ATENCION_DIGITAL_AUTOMATIZADA"

        classified.append(
            {
                **cust,
                "revenueSharePct": share_pct,
                "cumulativeSharePct": cum_pct,
                "abcTier": abc_tier,
                "inactivityStatus": inactivity_status,
                "recommendedAction": action,
            }
        )

    return {
        "enabled": True,
        "customersAnalyzed": len(classified),
        "totalPortfolioRevenue": round(grand_total, 2),
        "tierACount": tier_counts["A"],
        "tierBCount": tier_counts["B"],
        "tierCCount": tier_counts["C"],
        "riskCounts": risk_counts,
        "customers": classified,
    }
