"""Phase 3 tests — Parameterized classification, rubric evaluation, competitor scale, network coverage, and customer ABC engines."""

from __future__ import annotations

import copy

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.caleb.engines import (
    analyze_customer_portfolio,
    analyze_network_and_coverage,
    classify_competitor_and_scale,
    classify_entity_segment,
    deduplicate_entities,
    evaluate_entity_rubric,
    evaluate_zone_or_country,
)
from src.caleb.schema import DEFAULT_LOCAL_MARKET_CO_CONFIG
from src.graphs.caleb_graph import build_caleb_graph


def test_segment_classifier_positive_negative_disqualifier_and_tie_break():
    taxonomy = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG["taxonomy"])
    disqualifiers = DEFAULT_LOCAL_MARKET_CO_CONFIG["businessProfile"]["disqualifiers"]

    # 1. Positive match on WHOLESALE_IMPORTER
    importer = {
        "name": "Importadora Mayorista Central S.A.S.",
        "signals": ["importador directo", "ventas al por mayor"],
    }
    res_imp = classify_entity_segment(importer, taxonomy, disqualifiers)
    assert res_imp["disqualified"] is False
    assert res_imp["segmentCode"] == "WHOLESALE_IMPORTER"
    assert res_imp["segmentConfidence"] == "ALTA"

    # 2. Disqualifier match
    retail = {
        "name": "Ferretería El Martillo",
        "description": "Ferretería doméstica minorista para el hogar",
    }
    res_disq = classify_entity_segment(retail, taxonomy, disqualifiers)
    assert res_disq["disqualified"] is True
    assert res_disq["segmentCode"] == "DISQUALIFIED"

    # 3. Tie-breaking by priorityWeight and fallback to defaultSegmentCode
    empty_entity = {"name": "Empresa Genérica S.A.S.", "signals": []}
    res_fallback = classify_entity_segment(empty_entity, taxonomy, disqualifiers)
    assert res_fallback["segmentCode"] == taxonomy["defaultSegmentCode"]
    assert res_fallback["segmentConfidence"] == "BAJA"


def test_deduplicate_entities_merges_verified_fields_without_overwriting_with_sin_dato():
    raw = [
        {
            "name": "Rodamientos del Norte S.A.S.",
            "taxId": "900.123.456-7",
            "city": "Bogotá",
            "website": "https://www.rodamientosnorte.co",
            "phone": "sin dato",
            "email": "ventas@rodamientosnorte.co",
            "source": "SERPER_GOOGLE",
            "signals": ["importador directo"],
        },
        {
            "name": "RODAMIENTOS DEL NORTE SAS",
            "taxId": "9001234567",
            "city": "Bogotá",
            "website": "sin dato",
            "phone": "+57 601 7440000",
            "email": "sin dato",
            "source": "PUBLIC_REGISTRY_CO_RUES",
            "signals": ["cobertura nacional"],
        },
    ]
    deduped = deduplicate_entities(raw)
    assert len(deduped) == 1
    merged = deduped[0]
    assert merged["phone"] == "+57 601 7440000"
    assert merged["email"] == "ventas@rodamientosnorte.co"
    assert merged["website"] == "https://www.rodamientosnorte.co"
    assert set(merged["sources"]) == {"SERPER_GOOGLE", "PUBLIC_REGISTRY_CO_RUES"}
    assert set(merged["signals"]) == {"importador directo", "cobertura nacional"}


def test_evaluate_entity_rubric_bands_missing_data_and_weight_validation():
    rubric = copy.deepcopy(
        DEFAULT_LOCAL_MARKET_CO_CONFIG["rubrics"]["entityRubric"]
    )
    product_lines = DEFAULT_LOCAL_MARKET_CO_CONFIG["businessProfile"]["productLines"]

    # High-affinity verified account -> Band A (100 points)
    ent_a = {
        "name": "Importadora Industrial S.A.S.",
        "website": "https://impindustrial.co",
        "phone": "+57 601 2223344",
        "isImporter": True,
        "signals": ["importador directo", "tornillería grado 8", "rodamientos industriales"],
    }
    scored_a = evaluate_entity_rubric(ent_a, rubric, product_lines)
    assert scored_a["score"] == 100.0
    assert scored_a["priorityBand"] == "A"

    # Entity with explicit "sin dato" on criteria -> 0 points on missing criteria, never invented
    ent_missing = {
        "name": "Taller Sin Datos",
        "website": "sin dato",
        "phone": "sin dato",
        "email": "sin dato",
        "criterionValues": {
            "product_affinity": "media",
            "operational_scale": "sin dato",
            "contactability_and_digital": "sin dato",
        },
    }
    scored_missing = evaluate_entity_rubric(ent_missing, rubric, product_lines)
    assert scored_missing["criterionBreakdown"]["operational_scale"]["value"] == "sin dato"
    assert scored_missing["criterionBreakdown"]["operational_scale"]["awardedPoints"] == 0.0
    assert "operational_scale" in scored_missing["missingCriteria"]
    assert scored_missing["priorityBand"] == "C"

    # Invalid weights raise ValueError
    bad_rubric = copy.deepcopy(rubric)
    bad_rubric["criteria"][0]["weight"] = 0.9
    with pytest.raises(ValueError, match="weights must be positive and sum to 1.0"):
        evaluate_entity_rubric(ent_a, bad_rubric, product_lines)


def test_evaluate_zone_or_country_go_explorar_no_priorizar_and_missing_dims():
    zone_rubric = copy.deepcopy(
        DEFAULT_LOCAL_MARKET_CO_CONFIG["rubrics"]["zoneOrCountryRubric"]
    )

    go_target = {
        "code": "GT",
        "name": "Guatemala",
        "dimensionScores": {
            "demand_density": 80.0,
            "competitive_whitespace": 75.0,
            "logistics_accessibility": 70.0,
        },
    }
    res_go = evaluate_zone_or_country(go_target, zone_rubric)
    assert res_go["decision"] == "GO"
    assert res_go["score"] == 75.5

    explore_target = {
        "code": "SV",
        "name": "El Salvador",
        "dimensionScores": {
            "demand_density": 65.0,
            "competitive_whitespace": 55.0,
            "logistics_accessibility": "sin dato",
        },
    }
    res_exp = evaluate_zone_or_country(explore_target, zone_rubric)
    # 65*0.4 (26) + 55*0.3 (16.5) + 0 = 42.5 -> NO_PRIORIZAR (< 45)
    assert res_exp["score"] == 42.5
    assert res_exp["decision"] == "NO_PRIORIZAR"
    assert "logistics_accessibility" in res_exp["missingDimensions"]


def test_competitor_classifier_and_explicit_scale_ranges():
    ref_brands = DEFAULT_LOCAL_MARKET_CO_CONFIG["businessProfile"]["referenceBrands"]

    tier1 = {
        "name": "Mayorista SKF Colombia S.A.",
        "isImporter": True,
        "branchesCount": 4,
        "brands": ["SKF"],
        "signals": ["importador directo", "cobertura nacional"],
    }
    res1 = classify_competitor_and_scale(tier1, ref_brands)
    assert res1["competitiveLevel"] == "NIVEL_1_IMPORTADOR_MAYORISTA"
    assert "USD 5M - 15M" in res1["estimatedAnnualRevenueRangeUsd"]
    assert res1["matchedReferenceBrands"][0]["name"] == "SKF"

    no_data = {"name": "Empresa Misteriosa", "website": "sin dato", "signals": []}
    res_none = classify_competitor_and_scale(no_data, ref_brands)
    assert res_none["competitiveLevel"] == "NO_CLASIFICADO"
    assert res_none["estimatedAnnualRevenueRangeUsd"] == "sin dato"
    assert res_none["scaleConfidence"] == "SIN_DATO"


def test_network_coverage_gaps_and_distribution_routes():
    zones = DEFAULT_LOCAL_MARKET_CO_CONFIG["geography"]["localZones"]
    segments = DEFAULT_LOCAL_MARKET_CO_CONFIG["taxonomy"]["segments"]
    zone_rubric = DEFAULT_LOCAL_MARKET_CO_CONFIG["rubrics"]["zoneOrCountryRubric"]

    entities = [
        {
            "name": "Hub Importador Bogotá",
            "zoneCode": "BOG_ZONA_INDUSTRIAL",
            "city": "Bogotá",
            "segmentCode": "WHOLESALE_IMPORTER",
            "priorityBand": "A",
            "competitiveLevel": "NIVEL_1_IMPORTADOR_MAYORISTA",
            "phone": "+57 601 3334455",
        }
    ]
    net = analyze_network_and_coverage(entities, zones, segments, zone_rubric)
    assert len(net["zones"]) == 2
    assert len(net["coverageGaps"]) > 0
    assert any(
        g["zoneCode"] == "ANT_VALLE_ABURRA" and g["gapSeverity"] == "VACIO_TOTAL"
        for g in net["coverageGaps"]
    )
    assert len(net["distributionRoutes"]) == 1
    assert net["distributionRoutes"][0]["hubZoneCode"] == "BOG_ZONA_INDUSTRIAL"
    assert net["distributionRoutes"][0]["targetZoneCode"] == "ANT_VALLE_ABURRA"


def test_customer_portfolio_merges_one_and_csv_into_abc_and_inactivity_tiers():
    cbc = {
        "sourceMode": "BOTH",
        "abcThresholds": {"tierAPct": 75.0, "tierBPct": 15.0},
        "inactivityDaysWarning": 60,
        "inactivityDaysChurned": 180,
    }
    customers = [
        {
            "customerId": "cust-1",
            "name": "Industrias Alfa S.A.",
            "totalRevenue": 50_000,
            "ordersCount": 5,
            "daysSinceLastOrder": 195,
            "source": "ONE_CRM",
        },
        {
            "customerId": "cust-1",
            "name": "Industrias Alfa S.A.",
            "totalRevenue": 30_000,
            "ordersCount": 3,
            "daysSinceLastOrder": 210,
            "source": "CSV_UPLOAD",
        },
        {
            "customerId": "cust-2",
            "name": "Metalmecánica Beta",
            "totalRevenue": 15_000,
            "ordersCount": 4,
            "daysSinceLastOrder": 75,
            "source": "ONE_CRM",
        },
        {
            "customerId": "cust-3",
            "name": "Taller Gamma",
            "totalRevenue": 5_000,
            "ordersCount": 1,
            "daysSinceLastOrder": "sin dato",
            "source": "CSV_UPLOAD",
        },
    ]
    res = analyze_customer_portfolio(customers, cbc)
    assert res["customersAnalyzed"] == 3
    assert res["totalPortfolioRevenue"] == 100_000.0
    assert res["tierACount"] == 1
    assert res["tierBCount"] == 1
    assert res["tierCCount"] == 1

    c1 = res["customers"][0]
    assert c1["abcTier"] == "A"
    assert c1["inactivityStatus"] == "INACTIVO_CHURN"
    assert c1["recommendedAction"] == "PRIORIDAD_1_REACTIVACION_EJECUTIVA"
    assert set(c1["sources"]) == {"ONE_CRM", "CSV_UPLOAD"}


@pytest.mark.asyncio
async def test_graph_skips_llm_rationale_when_budget_degraded_to_deterministic_only():
    graph = build_caleb_graph(checkpointer=MemorySaver())
    cfg = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    cfg["modules"] = ["LOCAL_MARKET", "CURRENT_CUSTOMERS"]
    cfg["customerBaseConfig"] = {
        "sourceMode": "BOTH",
        "abcThresholds": {"tierAPct": 80.0, "tierBPct": 15.0},
        "inactivityDaysWarning": 60,
        "inactivityDaysChurned": 180,
    }

    llm_calls: list[str] = []

    def fake_llm_rationale(entity: dict) -> str:
        llm_calls.append(entity["name"])
        return "Justificación LLM generada."

    # Start with settled_spend_usd = 8.8 out of 10.0 cap (88% -> DETERMINISTIC_SCORING_ONLY)
    res = await graph.ainvoke(
        {
            "tenant_id": "tenant-p3",
            "study_run_id": "run-p3",
            "config": cfg,
            "internal_cost_cap_usd": 10.0,
            "settled_spend_usd": 8.8,
            "checkpoint_approved": True,
            "raw_customers": [
                {
                    "customerId": "c1",
                    "name": "Cliente Clave Uno",
                    "totalRevenue": 90_000,
                    "ordersCount": 10,
                    "daysSinceLastOrder": 20,
                    "source": "ONE_CRM",
                }
            ],
            "candidate_entities": [
                {
                    "entityId": "cand-ambiguous",
                    "name": "Importadora Ambigua S.A.S.",
                    "city": "Bogotá",
                    "zoneCode": "BOG_ZONA_INDUSTRIAL",
                    "website": "https://impambigua.co",
                    "phone": "+57 601 9998877",
                    "source": "SERPER_GOOGLE",
                    "signals": ["importador directo", "tornillería grado 8"],
                    "criterionValues": {
                        "product_affinity": "alta",
                        "operational_scale": "sin dato",
                        "contactability_and_digital": "alta",
                    },
                }
            ],
        },
        config={
            "configurable": {
                "thread_id": "caleb:tenant-p3:run-p3",
                "llm_rationale_fn": fake_llm_rationale,
            }
        },
    )

    assert res["status"] == "COMPLETED"
    assert llm_calls == []
    assert res["internal_baseline"]["customersAnalyzed"] == 1
    assert res["scored_entities"][0]["priorityBand"] == "B"
    assert (
        "Scoring determinístico aplicado por política de presupuesto."
        in res["scored_entities"][0]["qualitativeRationale"]
    )
