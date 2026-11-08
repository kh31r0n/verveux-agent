"""Executive Markdown report generator driven strictly by the versioned JSON report model."""

from __future__ import annotations

from typing import Any


def render_markdown_report(report_model: dict[str, Any]) -> str:
    """Renders the 12-section executive Markdown report from CalebReportModelV1."""
    sections = report_model.get("sections") or {}
    exec_sum = sections.get("executiveSummary") or {}
    zone_cards = sections.get("zoneCountryCards") or []
    seg_acc = sections.get("segmentsAndAccounts") or {}
    accounts = seg_acc.get("accounts") or []
    comp_demand = sections.get("competitionAndPublicDemand") or {}
    customers = sections.get("currentCustomersByRegion") or {}
    sensitivity = sections.get("sensitivityAnalysis") or {}
    ai_audit = sections.get("aiReliabilityAudit") or {}
    action_plan = sections.get("actionPlan") or {}
    methodology = sections.get("methodology") or {}
    sources = sections.get("sourcesWithDates") or []
    limits = sections.get("limitsAndGlossary") or {}

    lines: list[str] = [
        f"# {report_model.get('studyTitle', 'Estudio de Mercado B2B')}",
        "",
        f"- **Fecha de generación:** {report_model.get('generatedAt', 'sin dato')}",
        f"- **Nivel de profundidad:** {report_model.get('tier', 'PILOT')}",
        f"- **Módulos activos:** {', '.join(report_model.get('modules') or [])}",
        "",
        "## Resumen Ejecutivo",
        f"- **Entidades evaluadas:** {exec_sum.get('totalEvaluatedEntities', 0)}",
        f"- **Cuentas Prioridad A:** {exec_sum.get('priorityBandACount', 0)}",
        f"- **Cuentas Prioridad B:** {exec_sum.get('priorityBandBCount', 0)}",
        f"- **Cuentas Prioridad C:** {exec_sum.get('priorityBandCCount', 0)}",
        f"- **Zonas o países evaluados:** {exec_sum.get('zonesOrCountriesEvaluated', 0)}",
        f"- **Recomendación principal:** {exec_sum.get('topRecommendation', 'sin dato')}",
        "",
        "## Mapa y Fichas por Zona o País",
    ]

    if zone_cards:
        lines.append("| Código | Nombre | Puntaje | Decisión | Dimensiones sin dato |")
        lines.append("|---|---|---|---|---|")
        for zc in zone_cards:
            missing = ", ".join(zc.get("missingDimensions") or []) or "ninguna"
            lines.append(
                f"| {zc.get('code', 'sin dato')} | {zc.get('name', 'sin dato')} | "
                f"{zc.get('score', 'sin dato')} | {zc.get('decision', 'sin dato')} | {missing} |"
            )
    else:
        lines.append("Sin zonas o países evaluados.")

    lines.extend(
        [
            "",
            "## Segmentos y Cuentas Evaluadas",
        ]
    )
    if accounts:
        lines.append("| Nombre | Segmento | Puntaje | Banda | Ciudad | Sitio Web | Teléfono |")
        lines.append("|---|---|---|---|---|---|---|")
        for acc in accounts:
            lines.append(
                f"| {acc.get('name', 'sin dato')} | {acc.get('segmentCode', 'sin dato')} | "
                f"{acc.get('score', 'sin dato')} | {acc.get('priorityBand', 'sin dato')} | "
                f"{acc.get('city', 'sin dato')} | {acc.get('website', 'sin dato')} | "
                f"{acc.get('phone', 'sin dato')} |"
            )
    else:
        lines.append("Sin cuentas evaluadas.")

    lines.extend(
        [
            "",
            "## Competencia y Demanda Pública / Comercio Exterior",
        ]
    )
    trade_rows = comp_demand.get("tradeAndMacroSummary") or []
    if trade_rows:
        for tr in trade_rows:
            lines.append(
                f"- **{tr.get('country', 'sin dato')}**: Importaciones USD `{tr.get('importsUsd', 'sin dato')}`, "
                f"Arancel `{tr.get('appliedTariffPct', 'sin dato')}`, PIB USD `{tr.get('gdpUsd', 'sin dato')}`"
            )
    else:
        lines.append("- Demanda pública / comercio exterior: sin dato")

    lines.extend(
        [
            "",
            "## Cartera de Clientes Actuales",
            f"- **Clientes analizados:** {customers.get('customersAnalyzed', 0)}",
            "",
            "## Análisis de Sensibilidad",
            f"- **Semilla determinista:** {sensitivity.get('seed', 2026)}",
            f"- **Sorteos ejecutados:** {sensitivity.get('draws', 0)}",
            f"- **Estabilidad del líder (%):** {sensitivity.get('topRankStabilityPct', 'sin dato')}",
            "",
            "## Fiabilidad de la IA",
            f"- **Regla determinista:** {ai_audit.get('deterministicRule', '')}",
            f"- **Entidades con evidencia verificada:** {ai_audit.get('verifiedEntitiesCount', 0)}",
            f"- **Entidades por verificar:** {ai_audit.get('pendingVerificationCount', 0)}",
            "",
            "## Plan de Acción",
            f"- **Plan Inicial:** {action_plan.get('days30', '')}",
            f"- **Plan Intermedio:** {action_plan.get('days60', '')}",
            f"- **Plan Consolidación:** {action_plan.get('days90', '')}",
            "",
            "## Metodología",
            f"- {methodology.get('scoringModel', '')}",
            f"- {methodology.get('deduplication', '')}",
            "",
            "## Fuentes con su Fecha",
        ]
    )
    for src in sources:
        lines.append(
            f"- `{src.get('sourceKey', 'sin dato')}` — {src.get('url', 'sin dato')} "
            f"({src.get('status', 'OK')}, fecha: {src.get('fetchedAt', 'sin dato')})"
        )

    lines.extend(
        [
            "",
            "## Límites y Glosario",
            f"- **Tope de presupuesto interno (USD):** {limits.get('budgetCapUsd', 0.0)}",
            f"- **Gasto liquidado (USD):** {limits.get('settledSpendUsd', 0.0)}",
        ]
    )
    for warn in limits.get("qualityWarnings") or []:
        lines.append(f"- **Aviso de cobertura:** {warn}")
    for deg in limits.get("degradations") or []:
        lines.append(f"- **Degradación de presupuesto:** {deg.get('reason', 'sin dato')}")

    return "\n".join(lines) + "\n"
