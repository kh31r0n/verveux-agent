"""Unit and property tests for Caleb Phase 5 deliverable exporters (JSON Report v1, Markdown, CSV, OOXML Excel, CRM & Aurora)."""

from __future__ import annotations

import base64
import copy

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.caleb.exporters import (
    build_aurora_proposals,
    build_caleb_report_model_v1,
    build_crm_export_payload,
    build_excel_workbook_bytes,
    compute_deterministic_sensitivity,
    export_entities_csv,
    parse_mixed_locale_number,
    parse_xlsx_workbook_sheets,
    render_markdown_report,
    verify_prose_numbers_against_model,
)
from src.caleb.schema import DEFAULT_LOCAL_MARKET_CO_CONFIG
from src.graphs.caleb_graph import build_caleb_graph


def test_parse_mixed_locale_number_handles_decimal_and_thousand_commas_and_sin_dato() -> None:
    # Decimal comma coordinate
    assert parse_mixed_locale_number("-72,396132") == pytest.approx(-72.396132)
    # US thousand commas + dot decimal
    assert parse_mixed_locale_number("2,781,697.00") == pytest.approx(2781697.0)
    # LatAm dot thousands + comma decimal
    assert parse_mixed_locale_number("2.781.697,50") == pytest.approx(2781697.5)
    # "sin dato" is NEVER converted to 0
    assert parse_mixed_locale_number("sin dato") is None
    assert parse_mixed_locale_number("") is None
    assert parse_mixed_locale_number(None) is None


def test_report_model_12_sections_privacy_and_prose_number_property() -> None:
    cfg = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    state = {
        "study_run_id": "run-phase5-1",
        "tenant_id": "tenant-1",
        "config": cfg,
        "internal_cost_cap_usd": 44.70,
        "settled_spend_usd": 3.25,
        "scored_entities": [
            {
                "entityId": "ent-1",
                "name": "  Importadora Ferretera Andina S.A.S.  ",
                "taxId": "900111222-3",
                "segmentCode": "IMPORTADOR_MAYORISTA",
                "score": 88.5,
                "priorityBand": "A",
                "country": "CO",
                "city": "Bogotá",
                "zoneCode": "BOG_ZONA_INDUSTRIAL",
                "website": "https://ferreteraandina.com.co",
                "phone": "+57 601 7778899",
                "email": "ventas@ferreteraandina.com.co",
                "source": "SERPER_GOOGLE",
            },
            {
                "entityId": "ent-places",
                "name": "Ferretería Local Maps",
                "taxId": "sin dato",
                "segmentCode": "SUBDISTRIBUIDOR",
                "score": 65.0,
                "priorityBand": "B",
                "country": "CO",
                "city": "Medellín",
                "zoneCode": "ANT_MED",
                "website": "https://ferreterialocal.co",
                "phone": "+57 300 9998877",
                "source": "GOOGLE_PLACES",
            },
            {
                "entityId": "ent-natural",
                "name": "Juan Pérez Gómez",
                "taxId": "79123456",
                "isNaturalPerson": True,
                "segmentCode": "SUBDISTRIBUIDOR",
                "score": 45.0,
                "priorityBand": "C",
                "country": "CO",
                "city": "Cali",
                "aiRationale": "Texto que debe ser reemplazado",
            },
        ],
        "zone_country_analysis": [
            {"code": "BOG_ZONA_INDUSTRIAL", "name": "Bogotá Industrial", "score": 84.0, "decision": "GO"},
            {"code": "ANT_MED", "name": "Medellín", "score": 71.0, "decision": "EXPLORAR"},
        ],
    }

    report_model = build_caleb_report_model_v1(state)
    sections = report_model["sections"]
    expected_12 = {
        "executiveSummary",
        "geographyMap",
        "zoneCountryCards",
        "segmentsAndAccounts",
        "competitionAndPublicDemand",
        "currentCustomersByRegion",
        "sensitivityAnalysis",
        "aiReliabilityAudit",
        "actionPlan",
        "methodology",
        "sourcesWithDates",
        "limitsAndGlossary",
    }
    assert set(sections.keys()) == expected_12

    accounts = sections["segmentsAndAccounts"]["accounts"]
    # Natural person is pseudonymized and AI profile forced to "sin evidencia"
    nat = [a for a in accounts if a.get("isNaturalPerson")][0]
    assert nat["entityId"].startswith("persona-natural-anon-")
    assert "Juan Pérez" not in nat["name"]
    assert nat["aiRationale"] == "sin evidencia"

    # Google Places phone is stripped to "sin dato"
    places_acc = [a for a in accounts if a.get("source") == "GOOGLE_PLACES"][0]
    assert places_acc["phone"] == "sin dato"

    # Property: every number in the Markdown prose exists in the JSON report model
    md_prose = render_markdown_report(report_model)
    ok, unmatched = verify_prose_numbers_against_model(md_prose, report_model)
    assert ok is True, f"Unmatched numbers in prose: {unmatched}"

    # Deterministic sensitivity is 100% reproducible with the same seed
    s1 = compute_deterministic_sensitivity(state["zone_country_analysis"], seed=2026)
    s2 = compute_deterministic_sensitivity(state["zone_country_analysis"], seed=2026)
    assert s1 == s2


def test_ooxml_excel_workbook_structure_whitespace_normalization_and_csv() -> None:
    cfg = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    state = {
        "study_run_id": "run-xlsx-1",
        "config": cfg,
        "internal_cost_cap_usd": 11.70,
        "settled_spend_usd": 2.10,
        "scored_entities": [
            {
                "entityId": "  ent-1  ",
                "name": "  Suministros del Valle S.A.S.  ",
                "taxId": " 900222333-1 ",
                "segmentCode": "IMPORTADOR_MAYORISTA",
                "score": 82.0,
                "priorityBand": "A",
                "country": "CO",
                "city": "Cali",
                "zoneCode": "VAL_CALI",
                "website": " https://suministrosvalle.co ",
                "phone": "+57 602 3334455",
                "email": "info@suministrosvalle.co",
                "source": "SERPER_GOOGLE",
            }
        ],
        "top_k_profiles": [
            {
                "entityId": "ent-1",
                "name": "Suministros del Valle S.A.S.",
                "priorityBand": "A",
                "executiveSummary": "Líder regional en suministros.",
                "suggestedApproach": "Contacto técnico directo.",
            }
        ],
    }

    report_model = build_caleb_report_model_v1(state)
    xlsx_bytes = build_excel_workbook_bytes(report_model, state)
    sheets = parse_xlsx_workbook_sheets(xlsx_bytes)

    assert "Cuentas_Evaluadas" in sheets
    assert "Zonas_Paises" in sheets
    assert "Cartera_Clientes" in sheets
    assert "IA · Segunda Opinion" in sheets
    assert "IA · Perfiles Top K" in sheets
    assert "Configuracion" in sheets
    assert "Gasto_Presupuesto" in sheets

    # Verify leading/trailing whitespace was normalized in OOXML cells
    acc_rows = sheets["Cuentas_Evaluadas"]
    assert acc_rows[0][-4:] == ["Fuente", "Fecha", "URL", "Estado_Verificacion"]
    assert acc_rows[1][0] == "ent-1"
    assert acc_rows[1][1] == "Suministros del Valle S.A.S."
    assert acc_rows[1][-1] == "verificado"

    csv_text = export_entities_csv(report_model)
    assert "Suministros del Valle S.A.S." in csv_text
    assert "verificado" in csv_text


def test_crm_deduplication_and_aurora_single_country_proposals() -> None:
    cfg = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    cfg["deliverables"]["exportToOneCrm"]["enabled"] = True
    cfg["deliverables"]["exportToOneCrm"]["minPriorityBand"] = "B"
    cfg["geography"]["targetCountries"] = [
        {
            "countryCode": "EC",
            "countryName": "Ecuador",
            "currency": "USD",
            "priorityCities": ["Quito", "Guayaquil"],
            "enabledSources": ["SERPER_GOOGLE"],
        }
    ]

    state = {
        "study_run_id": "run-crm-1",
        "config": cfg,
        "scored_entities": [
            {
                "entityId": "e1",
                "name": "Importadora Andina S.A.S.",
                "taxId": "900111222-3",
                "priorityBand": "A",
                "score": 90.0,
                "website": "https://importadoraandina.co",
                "phone": "+57 300 1112233",
                "source": "GOOGLE_PLACES",
            },
            # Batch duplicate (same domain & normalized company name)
            {
                "entityId": "e2",
                "name": "Importadora Andina SAS",
                "taxId": "900111222-3",
                "priorityBand": "A",
                "score": 88.0,
                "website": "https://www.importadoraandina.co",
            },
            # Already in CRM duplicate
            {
                "entityId": "e3",
                "name": "Ferretería Existente CRM Ltda",
                "taxId": "800999888-1",
                "priorityBand": "A",
                "score": 85.0,
                "website": "https://existente.co",
            },
            # Unverified entity (no website, no taxId, no email) -> skipped
            {
                "entityId": "e4",
                "name": "Taller Sin Evidencia",
                "taxId": "sin dato",
                "priorityBand": "A",
                "score": 81.0,
                "website": "sin dato",
                "email": "sin dato",
            },
        ],
    }

    crm_payload = build_crm_export_payload(
        state, existing_crm_keys={"tax:800999888-1"}
    )
    assert crm_payload["exportedCount"] == 1
    assert crm_payload["skippedDuplicatesCount"] == 2
    assert crm_payload["skippedUnverifiedCount"] == 1

    prospect = crm_payload["prospects"][0]
    assert prospect["origin"] == "AI_PROSPECTING"
    assert prospect["convertedAt"] is None
    assert prospect["outboundBlockedUntilConverted"] is True
    # Google Places phone was stripped to "sin dato"
    assert prospect["phone"] == "sin dato"
    assert prospect["profileData"]["caleb"]["studyRunId"] == "run-crm-1"

    aurora = build_aurora_proposals(
        state,
        previous_aurora_config={"prospectingNiche": {"key": "old"}, "prospectingLocation": {"country": "CO"}},
    )
    assert aurora["requiresExplicitHumanAcceptance"] is True
    assert aurora["isMultiCountryStudy"] is True
    countries = [opt["country"] for opt in aurora["optionsByCountry"]]
    assert "CO" in countries and "EC" in countries
    for opt in aurora["optionsByCountry"]:
        assert opt["prospectingLocation"]["country"] in ("CO", "EC")
        assert opt["estimatedDailySpendUsd"] > 0
    assert aurora["previousConfigSnapshot"]["prospectingNiche"]["key"] == "old"


@pytest.mark.asyncio
async def test_caleb_graph_generates_all_phase5_deliverables_and_crm_sync() -> None:
    cfg = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    cfg["deliverables"]["humanCheckpoints"]["afterDiscovery"] = False
    cfg["deliverables"]["humanCheckpoints"]["beforeCrmExport"] = False
    cfg["deliverables"]["exportToOneCrm"]["enabled"] = True

    graph = build_caleb_graph(checkpointer=MemorySaver())
    final_state = await graph.ainvoke(
        {
            "study_run_id": "run-e2e-phase5",
            "tenant_id": "tenant-1",
            "config": cfg,
            "internal_cost_cap_usd": 44.70,
        },
        config={"configurable": {"thread_id": "caleb:tenant-1:run-e2e-phase5"}},
    )

    assert final_state["status"] == "COMPLETED"
    assert final_state["report_model"]["reportSchemaVersion"] == 1
    kinds = {a["kind"] for a in final_state["artifacts"]}
    assert kinds == {"JSON_REPORT_MODEL", "MARKDOWN_REPORT", "CSV_ENTITIES", "EXCEL_WORKBOOK"}

    xlsx_art = [a for a in final_state["artifacts"] if a["kind"] == "EXCEL_WORKBOOK"][0]
    xlsx_bytes = base64.b64decode(xlsx_art["contentBase64"])
    sheets = parse_xlsx_workbook_sheets(xlsx_bytes)
    assert "Cuentas_Evaluadas" in sheets
    assert "IA · Segunda Opinion" in sheets
    assert final_state["crm_sync_summary"]["syncedCompanies"] >= 1
