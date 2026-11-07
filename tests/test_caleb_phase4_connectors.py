"""Unit and integration tests for Caleb Phase 4 data connectors (zero external network calls)."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.caleb.budget import CalebBudgetTracker
from src.caleb.connectors import (
    CalebCache,
    CircuitBreaker,
    HttpResponseStub,
    PROCUREMENT_PORTAL_CATALOG,
    ProcurementConnector,
    PublicRegistryConnector,
    TradeAndMacroConnector,
    WebAndPlacesConnector,
    WebsiteInspectorConnector,
    validate_public_http_url,
)
from src.caleb.schema import (
    CALEB_REGIONAL_COUNTRY_CODES,
    DEFAULT_LOCAL_MARKET_CO_CONFIG,
)
from src.graphs.caleb_graph import build_caleb_graph


class TestWebAndPlacesConnector:
    @pytest.mark.asyncio
    async def test_search_organic_filters_junk_caches_and_charges_budget_once(self) -> None:
        calls = 0

        async def fake_transport(method: str, url: str, payload: dict[str, Any]) -> HttpResponseStub:
            nonlocal calls
            calls += 1
            return HttpResponseStub(
                status_code=200,
                json_data={
                    "organic": [
                        {
                            "title": "Suministros Industriales Andinos S.A.S.",
                            "link": "https://suministrosandinos.com.co/catalogo",
                            "snippet": "Importador directo y mayorista en Medellín.",
                        },
                        {
                            "title": "Oferta de empleo vendedor ferretero",
                            "link": "https://co.computrabajo.com/ofertas-de-trabajo/vendedor",
                            "snippet": "Vacante disponible",
                        },
                    ]
                },
            )

        budget = CalebBudgetTracker(internal_cost_cap_usd=10.0)
        cache = CalebCache(ttl_seconds=3600)
        connector = WebAndPlacesConnector(
            transport=fake_transport,
            cache=cache,
            budget_tracker=budget,
            cost_per_query_usd=0.05,
        )

        params = {
            "query": "distribuidor abrasivos medellin",
            "country": "CO",
            "city": "Medellín",
            "zoneCode": "ANT_MED",
            "mode": "search",
        }
        first = await connector.fetch(params)
        assert first.status == "OK"
        assert first.cached is False
        assert len(first.data) == 1
        assert first.data[0]["name"] == "Suministros Industriales Andinos S.A.S."
        assert budget.settled_spend_usd == pytest.approx(0.05)

        # Second identical call hits cache and does NOT increment budget or transport calls
        second = await connector.fetch(params)
        assert second.status == "OK"
        assert second.cached is True
        assert calls == 1
        assert budget.settled_spend_usd == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_budget_hard_stop_skips_query(self) -> None:
        async def should_not_be_called(method: str, url: str, payload: dict[str, Any]) -> HttpResponseStub:
            raise AssertionError("Transport should not be called when budget is exhausted")

        budget = CalebBudgetTracker(internal_cost_cap_usd=0.01, settled_spend_usd=0.01)
        connector = WebAndPlacesConnector(
            transport=should_not_be_called,
            budget_tracker=budget,
            cost_per_query_usd=0.05,
        )
        res = await connector.fetch({"query": "tornilleria bogota", "country": "CO"})
        assert res.status == "BUDGET_SKIPPED"
        assert res.data == []
        assert len(res.warnings) == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_consecutive_failures(self) -> None:
        async def failing_transport(method: str, url: str, payload: dict[str, Any]) -> HttpResponseStub:
            return HttpResponseStub(status_code=502, text="Bad Gateway")

        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_sec=300)
        connector = WebAndPlacesConnector(
            transport=failing_transport,
            cache=CalebCache(),
            circuit_breaker=breaker,
        )

        r1 = await connector.fetch({"query": "q1", "country": "CO"})
        r2 = await connector.fetch({"query": "q2", "country": "CO"})
        assert r1.status == "TIMEOUT_OR_ERROR"
        assert r2.status == "TIMEOUT_OR_ERROR"
        assert breaker.is_open() is True

        r3 = await connector.fetch({"query": "q3", "country": "CO"})
        assert r3.status == "CIRCUIT_OPEN"
        assert len(r3.warnings) == 1


class TestPublicRegistryAndProcurementConnectors:
    @pytest.mark.asyncio
    async def test_colombia_rues_and_non_open_registry_fallback(self) -> None:
        async def fake_rues(method: str, url: str, params: dict[str, Any]) -> HttpResponseStub:
            return HttpResponseStub(
                status_code=200,
                json_data=[
                    {
                        "razon_social": "FERRETERIA INDUSTRIAL DEL NORTE SAS",
                        "nit": "900123456-1",
                        "municipio": "Medellín",
                        "ciiu": "4663",
                        "estado_matricula": "ACTIVA",
                    }
                ],
            )

        connector = PublicRegistryConnector(transport=fake_rues)
        co_res = await connector.fetch({"country": "CO", "city": "Medellín", "keyword": "FERRETERIA"})
        assert co_res.status == "OK"
        assert co_res.data[0]["taxId"] == "900123456-1"
        assert co_res.data[0]["ciiu"] == "4663"

        ec_res = await connector.fetch({"country": "EC", "city": "Quito", "keyword": "FERRETERIA"})
        assert ec_res.status == "EMPTY"
        assert any("sin dato" in w for w in ec_res.warnings)

    @pytest.mark.asyncio
    async def test_procurement_all_19_countries_and_waf_resilience(self) -> None:
        assert set(PROCUREMENT_PORTAL_CATALOG.keys()) == set(CALEB_REGIONAL_COUNTRY_CODES)
        assert len(PROCUREMENT_PORTAL_CATALOG) == 19

        async def waf_transport(method: str, url: str, params: dict[str, Any]) -> HttpResponseStub:
            return HttpResponseStub(
                status_code=403,
                text="Attention Required! | Cloudflare",
                headers={"cf-ray": "8899aabbcc"},
            )

        connector = ProcurementConnector(transport=waf_transport)
        res_waf = await connector.fetch({"country": "EC", "keyword": "abrasivos"})
        assert res_waf.status == "BLOCKED_WAF_OR_403"
        assert res_waf.data["tendersCount"] == "sin dato"
        assert len(res_waf.warnings) == 1

        # Country without open API (e.g., Nicaragua NI) reports sin dato cleanly
        res_ni = await connector.fetch({"country": "NI", "keyword": "abrasivos"})
        assert res_ni.status == "EMPTY"
        assert res_ni.data["tendersCount"] == "sin dato"
        assert len(res_ni.warnings) == 1


class TestTradeMacroAndWebsiteInspector:
    @pytest.mark.asyncio
    async def test_trade_and_macro_connectors(self) -> None:
        async def fake_transport(method: str, url: str, params: dict[str, Any]) -> HttpResponseStub:
            return HttpResponseStub(
                status_code=200,
                json_data={
                    "importsUsd": 1250000.0,
                    "exportsUsd": 310000.0,
                    "appliedTariffPct": 5.0,
                    "gdpUsd": 118000000000.0,
                    "logisticsPerformanceIndex": 3.1,
                },
            )

        connector = TradeAndMacroConnector(transport=fake_transport)
        trade_res = await connector.fetch(
            {"country": "EC", "hsCodes6": ["731815"], "kind": "comtrade"}
        )
        assert trade_res.status == "OK"
        assert trade_res.data["importsUsd"] == 1250000.0
        assert trade_res.data["appliedTariffPct"] == 5.0

    def test_ssrf_protection_blocks_private_loopback_and_metadata_urls(self) -> None:
        for unsafe_url in [
            "http://127.0.0.1/admin",
            "http://localhost:8000/caleb/run",
            "http://169.254.169.254/latest/meta-data/",
            "http://192.168.1.10/config",
            "http://10.0.0.5/internal",
            "file:///etc/passwd",
        ]:
            ok, reason = validate_public_http_url(unsafe_url)
            assert ok is False, f"Expected {unsafe_url} to be blocked, got ok=True"
            assert reason is not None

        ok_public, reason_public = validate_public_http_url("https://suministrosandinos.com.co/nosotros")
        assert ok_public is True
        assert reason_public is None

    @pytest.mark.asyncio
    async def test_website_inspector_extracts_contacts_brands_and_respects_ssrf(self) -> None:
        async def fake_web(method: str, url: str, params: dict[str, Any]) -> HttpResponseStub:
            return HttpResponseStub(
                status_code=200,
                text="""
                <html>
                  <head><title>Suministros Andinos - Importador Directo</title></head>
                  <body>
                    <p>Somos importador directo y distribuidor autorizado de Norton y 3M en Colombia.</p>
                    <p>Teléfono: +57 604 4445566</p>
                    <a href="mailto:ventas@suministrosandinos.com.co">Correo ventas</a>
                    <a href="https://wa.me/573001112233">WhatsApp</a>
                  </body>
                </html>
                """,
            )

        inspector = WebsiteInspectorConnector(transport=fake_web)
        blocked = await inspector.fetch({"url": "http://169.254.169.254/latest/meta-data/"})
        assert blocked.status == "SSRF_BLOCKED"

        valid = await inspector.fetch(
            {
                "url": "https://suministrosandinos.com.co",
                "referenceBrands": ["Norton", "3M", "Bosch"],
                "productKeywords": ["abrasivos"],
            }
        )
        assert valid.status == "OK"
        assert valid.data["email"] == "ventas@suministrosandinos.com.co"
        assert "Norton" in valid.data["brandsFound"]
        assert "3M" in valid.data["brandsFound"]
        assert "importador directo" in valid.data["signalsFound"]
        assert valid.data["hasWhatsapp"] is True


class TestCalebGraphWithPhase4Connectors:
    @pytest.mark.asyncio
    async def test_graph_runs_end_to_end_with_fixture_connectors(self) -> None:
        cfg = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
        cfg["modules"] = ["LOCAL_MARKET", "EXPORT_EXPANSION"]
        cfg["geography"]["targetCountries"] = [
            {
                "countryCode": "EC",
                "countryName": "Ecuador",
                "currency": "USD",
                "priorityCities": ["Quito", "Guayaquil"],
                "enabledSources": ["SERPER_GOOGLE", "TRADE_UN_COMTRADE"],
            }
        ]
        cfg["deliverables"]["humanCheckpoints"]["afterDiscovery"] = False
        cfg["deliverables"]["humanCheckpoints"]["beforeCrmExport"] = False

        async def web_transport(method: str, url: str, payload: dict[str, Any]) -> HttpResponseStub:
            return HttpResponseStub(
                status_code=200,
                json_data={
                    "organic": [
                        {
                            "title": "Importadora Ferretera Andina S.A.S.",
                            "link": "https://ferreteraandina.com.co",
                            "snippet": "Importador directo y mayorista de tornillería y fijaciones en Bogotá.",
                        }
                    ]
                },
            )

        async def reg_transport(method: str, url: str, params: dict[str, Any]) -> HttpResponseStub:
            return HttpResponseStub(
                status_code=200,
                json_data=[
                    {
                        "razon_social": "Ferretería Industrial del Sur S.A.S.",
                        "nit": "900555666-7",
                        "municipio": "Bogotá",
                        "ciiu": "4663",
                        "estado_matricula": "ACTIVA",
                    }
                ],
            )

        async def proc_transport(method: str, url: str, params: dict[str, Any]) -> HttpResponseStub:
            return HttpResponseStub(
                status_code=403,
                text="Cloudflare Challenge",
                headers={"cf-ray": "12345"},
            )

        async def trade_transport(method: str, url: str, params: dict[str, Any]) -> HttpResponseStub:
            return HttpResponseStub(
                status_code=200,
                json_data={
                    "importsUsd": 950000.0,
                    "exportsUsd": 120000.0,
                    "appliedTariffPct": 0.0,
                    "gdpUsd": 118000000000.0,
                },
            )

        async def inspector_transport(method: str, url: str, params: dict[str, Any]) -> HttpResponseStub:
            return HttpResponseStub(
                status_code=200,
                text="<html><title>Ferretera Andina</title><body>Importador directo ventas@ferreteraandina.com.co +57 601 7778899</body></html>",
            )

        graph = build_caleb_graph(checkpointer=MemorySaver())
        final_state = await graph.ainvoke(
            {
                "study_run_id": "run-phase4-1",
                "tenant_id": "tenant-1",
                "config": cfg,
                "internal_cost_cap_usd": 44.70,
            },
            config={
                "configurable": {
                    "thread_id": "caleb:tenant-1:run-phase4-1",
                    "connectors": {
                        "web_search": WebAndPlacesConnector(transport=web_transport),
                        "registries": PublicRegistryConnector(transport=reg_transport),
                        "procurement": ProcurementConnector(transport=proc_transport),
                        "trade_macro": TradeAndMacroConnector(transport=trade_transport),
                        "website_inspector": WebsiteInspectorConnector(transport=inspector_transport),
                    },
                }
            },
        )

        assert final_state["status"] == "COMPLETED"
        assert len(final_state["scored_entities"]) >= 2
        names = [e["name"] for e in final_state["scored_entities"]]
        assert "Importadora Ferretera Andina S.A.S." in names
        assert "Ferretería Industrial del Sur S.A.S." in names
        assert len(final_state["trade_macro_summary"]) == 1
        assert len(final_state["source_citations"]) >= 4
        assert any("403" in w or "WAF" in w for w in final_state["quality_warnings"])
