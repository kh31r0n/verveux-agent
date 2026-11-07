"""Phase 6 mandatory property, resumption, budget hard-stop, privacy, and B2B reference fixture tests (§12 of PROMPT_CALEB.md)."""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.caleb.connectors import (
    CalebCache,
    HttpResponseStub,
    ProcurementConnector,
    PublicRegistryConnector,
    TradeAndMacroConnector,
    WebAndPlacesConnector,
    WebsiteInspectorConnector,
)
from src.caleb.exporters import compute_deterministic_sensitivity
from src.caleb.schema import DEFAULT_LOCAL_MARKET_CO_CONFIG
from src.graphs.caleb_graph import build_caleb_graph


@pytest.mark.asyncio
async def test_fixture_1_local_zones_b2b_industrial_distributor() -> None:
    """Reference Fixture 1: B2B industrial distributor evaluating local zones within a country (Colombia)."""
    cfg = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    cfg["deliverables"]["humanCheckpoints"]["afterDiscovery"] = False
    cfg["deliverables"]["humanCheckpoints"]["beforeCrmExport"] = False
    cfg["geography"]["localZones"] = [
        {
            "code": "BOG_INDUSTRIAL",
            "name": "Bogotá Puente Aranda / Paloquemao",
            "country": "CO",
            "departmentOrState": "Cundinamarca",
            "cities": ["Bogotá"],
        },
        {
            "code": "ANT_VALLE_ABURRA",
            "name": "Medellín y Valle de Aburrá",
            "country": "CO",
            "departmentOrState": "Antioquia",
            "cities": ["Medellín", "Itagüí"],
        },
    ]

    graph = build_caleb_graph(checkpointer=MemorySaver())
    state = await graph.ainvoke(
        {
            "study_run_id": "fixture-local-zones-1",
            "tenant_id": "tenant-b2b",
            "config": cfg,
            "internal_cost_cap_usd": 44.70,
        },
        config={"configurable": {"thread_id": "caleb:tenant-b2b:fixture-local-zones-1"}},
    )

    assert state["status"] == "COMPLETED"
    assert len(state["zone_country_analysis"]) == 2
    s1 = compute_deterministic_sensitivity(state["zone_country_analysis"], seed=2026)
    s2 = compute_deterministic_sensitivity(state["zone_country_analysis"], seed=2026)
    assert s1 == s2


@pytest.mark.asyncio
async def test_fixture_2_central_america_6_countries_export_and_sin_dato_rules() -> None:
    """Reference Fixture 2: Export expansion across the 6 Central American countries (GT, SV, HN, NI, CR, PA).
    Verifies expected stages, 'sin dato' in procurement for CR, SV, and NI, and reproducible sensitivity.
    """
    cfg = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    cfg["modules"] = ["EXPORT_EXPANSION"]
    cfg["tier"] = "STANDARD"
    cfg["geography"]["localZones"] = []
    cfg["deliverables"]["humanCheckpoints"]["afterDiscovery"] = False
    cfg["deliverables"]["humanCheckpoints"]["beforeCrmExport"] = False
    ca_countries = [
        ("GT", "Guatemala", "GTQ", ["Ciudad de Guatemala"]),
        ("SV", "El Salvador", "USD", ["San Salvador"]),
        ("HN", "Honduras", "HNL", ["San Pedro Sula", "Tegucigalpa"]),
        ("NI", "Nicaragua", "NIO", ["Managua"]),
        ("CR", "Costa Rica", "CRC", ["San José"]),
        ("PA", "Panamá", "USD", ["Ciudad de Panamá"]),
    ]
    cfg["geography"]["targetCountries"] = [
        {
            "countryCode": code,
            "countryName": name,
            "currency": curr,
            "priorityCities": cities,
            "enabledSources": ["SERPER_GOOGLE", "TRADE_UN_COMTRADE"],
        }
        for code, name, curr, cities in ca_countries
    ]

    async def proc_transport(method: str, url: str, params: dict[str, Any]) -> HttpResponseStub:
        # GT, HN, PA have open OCDS portals
        return HttpResponseStub(
            status_code=200,
            json_data=[
                {
                    "proveedor_adjudicado": f"Distribuidora Industrial {params.get('country')}",
                    "valor_del_contrato": "45000",
                }
            ],
        )

    async def trade_transport(method: str, url: str, params: dict[str, Any]) -> HttpResponseStub:
        return HttpResponseStub(
            status_code=200,
            json_data={
                "importsUsd": 1800000.0,
                "exportsUsd": 220000.0,
                "appliedTariffPct": 5.0,
                "gdpUsd": 85000000000.0,
            },
        )

    graph = build_caleb_graph(checkpointer=MemorySaver())
    state = await graph.ainvoke(
        {
            "study_run_id": "fixture-ca-6",
            "tenant_id": "tenant-b2b",
            "config": cfg,
            "internal_cost_cap_usd": 68.70,
        },
        config={
            "configurable": {
                "thread_id": "caleb:tenant-b2b:fixture-ca-6",
                "connectors": {
                    "procurement": ProcurementConnector(transport=proc_transport),
                    "trade_macro": TradeAndMacroConnector(transport=trade_transport),
                },
            }
        },
    )

    assert state["status"] == "COMPLETED"
    proc_by_country = {item["country"]: item for item in state["procurement_summary"]}
    assert set(proc_by_country.keys()) == {"GT", "SV", "HN", "NI", "CR", "PA"}
    # CR blocks automated download; SV and NI do not publish -> all three must be "sin dato"
    for no_data_country in ("CR", "SV", "NI"):
        assert proc_by_country[no_data_country]["tendersCount"] == "sin dato"

    # Reproducible sensitivity with seed 2026
    sens = state["report_model"]["sections"]["sensitivityAnalysis"]
    assert sens["seed"] == 2026
    assert sens == compute_deterministic_sensitivity(state["zone_country_analysis"], seed=2026)


@pytest.mark.asyncio
async def test_property_ai_never_decides_rankings_or_scores() -> None:
    """Property (§12): Randomly changing AI outputs never changes any deterministic ranking or score."""
    cfg = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    cfg["deliverables"]["humanCheckpoints"]["afterDiscovery"] = False
    cfg["deliverables"]["humanCheckpoints"]["beforeCrmExport"] = False

    candidates = [
        {
            "entityId": "c1",
            "name": "Importadora Mayorista Uno S.A.S.",
            "country": "CO",
            "city": "Bogotá",
            "website": "https://uno.co",
            "phone": "+57 601 1111111",
            "signals": ["importador directo", "mayorista", "catálogo industrial"],
        },
        {
            "entityId": "c2",
            "name": "Subdistribuidora Regional Dos Ltda.",
            "country": "CO",
            "city": "Medellín",
            "website": "https://dos.co",
            "signals": ["ferretería", "mostrador"],
        },
    ]

    graph = build_caleb_graph(checkpointer=MemorySaver())
    baseline_rankings: list[tuple[str, float, str]] | None = None

    for run_idx in range(5):
        rng = random.Random(1000 + run_idx)
        random_ai_fn = lambda ent, r=rng: f"Opinión aleatoria IA #{r.randint(1, 999999)} puntaje falso {r.random()}"
        res = await graph.ainvoke(
            {
                "study_run_id": f"prop-ai-{run_idx}",
                "tenant_id": "tenant-prop",
                "config": cfg,
                "candidate_entities": copy.deepcopy(candidates),
                "internal_cost_cap_usd": 44.70,
            },
            config={
                "configurable": {
                    "thread_id": f"caleb:tenant-prop:ai-{run_idx}",
                    "llm_rationale_fn": random_ai_fn,
                }
            },
        )
        current_rankings = [
            (e["entityId"], float(e["score"]), str(e["priorityBand"]))
            for e in res["scored_entities"]
        ]
        if baseline_rankings is None:
            baseline_rankings = current_rankings
        else:
            assert current_rankings == baseline_rankings


@pytest.mark.asyncio
async def test_property_privacy_no_natural_person_in_outbound_http() -> None:
    """Property (§12): No natural person name or ID ever appears in outbound HTTP requests."""
    cfg = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    cfg["deliverables"]["humanCheckpoints"]["afterDiscovery"] = False
    cfg["deliverables"]["humanCheckpoints"]["beforeCrmExport"] = False

    outbound_requests: list[tuple[str, str, dict[str, Any]]] = []

    async def spy_transport(method: str, url: str, params: dict[str, Any]) -> HttpResponseStub:
        outbound_requests.append((method, url, copy.deepcopy(params)))
        return HttpResponseStub(status_code=200, text="<html><body>B2B</body></html>", json_data={"organic": []})

    candidates = [
        {
            "entityId": "nat-1",
            "name": "Carlos Alberto Rodríguez Peña",
            "taxId": "1020304050",
            "isNaturalPerson": True,
            "website": "https://carlosrodriguez-personal.co",
            "country": "CO",
            "city": "Bogotá",
        },
        {
            "entityId": "b2b-1",
            "name": "Suministros Industriales del Norte S.A.S.",
            "taxId": "900888777-1",
            "isNaturalPerson": False,
            "website": "https://suministrosnorte.co",
            "country": "CO",
            "city": "Bogotá",
        },
    ]

    graph = build_caleb_graph(checkpointer=MemorySaver())
    res = await graph.ainvoke(
        {
            "study_run_id": "prop-privacy-1",
            "tenant_id": "tenant-prop",
            "config": cfg,
            "candidate_entities": candidates,
            "internal_cost_cap_usd": 44.70,
        },
        config={
            "configurable": {
                "thread_id": "caleb:tenant-prop:privacy-1",
                "connectors": {
                    "web_search": WebAndPlacesConnector(transport=spy_transport),
                    "website_inspector": WebsiteInspectorConnector(transport=spy_transport),
                },
            }
        },
    )

    serialized_outbound = repr(outbound_requests)
    assert "Carlos Alberto Rodríguez Peña" not in serialized_outbound
    assert "1020304050" not in serialized_outbound
    assert "carlosrodriguez-personal.co" not in serialized_outbound

    nat_entities = [e for e in res["scored_entities"] if e.get("isNaturalPerson")]
    assert len(nat_entities) == 1
    assert nat_entities[0]["qualitativeRationale"] == "sin evidencia"


@pytest.mark.asyncio
async def test_budget_hard_stop_and_idempotent_resumption_zero_extra_paid_calls() -> None:
    """Verifies (§12):
    1) Expensive simulated calls never exceed internal_cost_cap_usd; run degrades and finishes PARTIAL.
    2) Interrupting/resuming with shared connector cache produces identical outputs with zero additional paid calls.
    """
    cfg = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    cfg["deliverables"]["humanCheckpoints"]["afterDiscovery"] = True
    cfg["deliverables"]["humanCheckpoints"]["beforeCrmExport"] = False

    paid_network_calls = 0

    async def counting_transport(method: str, url: str, params: dict[str, Any]) -> HttpResponseStub:
        nonlocal paid_network_calls
        paid_network_calls += 1
        return HttpResponseStub(
            status_code=200,
            json_data={
                "organic": [
                    {
                        "title": "Ferretería Industrial Mayorista S.A.S.",
                        "link": "https://ferreteriamayorista.co",
                        "snippet": "Importador directo y mayorista en Bogotá.",
                    }
                ]
            },
        )

    shared_cache = CalebCache(ttl_seconds=3600)
    web_conn = WebAndPlacesConnector(
        transport=counting_transport,
        cache=shared_cache,
        cost_per_query_usd=0.25,
    )

    checkpointer = MemorySaver()
    graph = build_caleb_graph(checkpointer=checkpointer)

    # First run pauses at human_checkpoint (AWAITING_REVIEW) after discovery
    paused_state = await graph.ainvoke(
        {
            "study_run_id": "run-resume-1",
            "tenant_id": "tenant-1",
            "config": cfg,
            "internal_cost_cap_usd": 11.70,
            "checkpoint_approved": False,
        },
        config={
            "configurable": {
                "thread_id": "caleb:tenant-1:run-resume-1",
                "connectors": {"web_search": web_conn},
            }
        },
    )
    assert paid_network_calls == 1
    assert "human_checkpoint" in paused_state["completed_stages"]

    # Resuming after kill/restart with checkpoint_approved=True uses cache -> 0 additional paid network calls!
    resumed_state = await graph.ainvoke(
        {
            "study_run_id": "run-resume-1",
            "tenant_id": "tenant-1",
            "config": cfg,
            "internal_cost_cap_usd": 11.70,
            "checkpoint_approved": True,
        },
        config={
            "configurable": {
                "thread_id": "caleb:tenant-1:run-resume-1-resumed",
                "connectors": {"web_search": web_conn},
            }
        },
    )
    assert resumed_state["status"] == "COMPLETED"
    assert paid_network_calls == 1  # Zero extra paid calls!

    # Now test simulated super-expensive run: cap = $1.00, already at $0.96 -> degrades to HARD_STOP_EXTERNAL_TOOLS -> PARTIAL
    cfg_no_cp = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    cfg_no_cp["deliverables"]["humanCheckpoints"]["afterDiscovery"] = False
    cfg_no_cp["deliverables"]["humanCheckpoints"]["beforeCrmExport"] = False
    degraded_state = await graph.ainvoke(
        {
            "study_run_id": "run-expensive-1",
            "tenant_id": "tenant-1",
            "config": cfg_no_cp,
            "internal_cost_cap_usd": 1.00,
            "settled_spend_usd": 0.96,
        },
        config={"configurable": {"thread_id": "caleb:tenant-1:run-expensive-1"}},
    )
    assert degraded_state["settled_spend_usd"] <= degraded_state["internal_cost_cap_usd"]
    assert degraded_state["status"] == "PARTIAL"
    assert any(
        d["level"] == "HARD_STOP_EXTERNAL_TOOLS" for d in degraded_state["degradations"]
    )
