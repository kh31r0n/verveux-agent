"""aurora's search sources: Serper credit exhaustion, Google Places, the domain
blocklist, metropolitan areas and the per-run credit budget (2026-10-16).

From 2026-10-13 to 10-16 the Serper account sat at a negative balance and every
call answered ``400 {"message":"Not enough credits"}``. The agent retried it as a
transient error, geo-fell-back, found nothing and reported every run COMPLETED.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src import main
from src.agents import prospecting_nodes as pn
from src.config import settings
from src.providers.base import UsageInfo
from src.services import google_places as gp
from src.services import serper


def _client_returning(*responses: httpx.Response):
    """A fake httpx.AsyncClient that answers each request with the next response."""
    queue = list(responses)
    for r in queue:
        r.request = httpx.Request("POST", "https://test.invalid/")
    calls: list[dict] = []

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **kw):
            calls.append({"url": url, **kw})
            return queue.pop(0) if len(queue) > 1 else queue[0]

        async def get(self, url, **kw):
            calls.append({"url": url, **kw})
            return queue.pop(0) if len(queue) > 1 else queue[0]

    return _Client, calls


def _req(resp: httpx.Response, url: str = "https://google.serper.dev/search"):
    resp.request = httpx.Request("POST", url)
    return resp


# ── Serper: credits exhausted ───────────────────────────────────────────────


class TestSerperQuotaExhausted:
    async def test_not_enough_credits_raises_once_without_retry(self):
        serper.start_serper_accounting()
        client, calls = _client_returning(
            _req(httpx.Response(400, json={"message": "Not enough credits", "statusCode": 400}))
        )
        with (
            patch.object(settings, "serper_api_key", "k"),
            patch("src.services.serper.httpx.AsyncClient", client),
            patch("src.services.serper.asyncio.sleep", AsyncMock()),
        ):
            with pytest.raises(serper.SerperQuotaExhaustedError):
                await serper.serper_post(serper.SERPER_ENDPOINT, {"q": "x"})
        assert len(calls) == 1
        assert serper.serper_call_count() == 1

    async def test_quota_error_is_a_serper_auth_error(self):
        # Every path that already treats a rejected key as fatal keeps working.
        assert issubclass(serper.SerperQuotaExhaustedError, serper.SerperAuthError)

    @pytest.mark.parametrize(
        "status,body,expected",
        [
            (400, '{"message":"Not enough credits"}', True),
            (402, "", True),
            (400, '{"message":"Invalid location"}', False),
            (429, "Not enough credits", False),
            (500, "credits", False),
        ],
    )
    def test_quota_response_detection(self, status, body, expected):
        assert serper.is_quota_exhausted_response(status, body) is expected

    async def test_other_400_is_a_per_query_miss_without_retry(self):
        serper.start_serper_accounting()
        client, calls = _client_returning(
            _req(httpx.Response(400, json={"message": "Invalid location"}))
        )
        with (
            patch.object(settings, "serper_api_key", "k"),
            patch("src.services.serper.httpx.AsyncClient", client),
            patch("src.services.serper.asyncio.sleep", AsyncMock()),
        ):
            assert await serper.serper_post(serper.SERPER_ENDPOINT, {"q": "x"}) is None
        assert len(calls) == 1

    async def test_5xx_is_still_retried(self):
        serper.start_serper_accounting()
        client, calls = _client_returning(_req(httpx.Response(503, text="down")))
        with (
            patch.object(settings, "serper_api_key", "k"),
            patch("src.services.serper.httpx.AsyncClient", client),
            patch("src.services.serper.asyncio.sleep", AsyncMock()),
        ):
            assert await serper.serper_post(serper.SERPER_ENDPOINT, {"q": "x"}) is None
        assert len(calls) == 3

    async def test_exhaustion_does_not_trigger_the_geo_fallback(self):
        """The fallback re-sends without `location`; on an empty account that
        is one more doomed (and billed) call per query."""
        post = AsyncMock(side_effect=serper.SerperQuotaExhaustedError("no credits"))
        with patch("src.agents.prospecting_nodes._serper_post", post):
            with pytest.raises(serper.SerperQuotaExhaustedError):
                await pn._serper_search(
                    "colegios en Medellín Colombia", pn.DEFAULT_LOCATION
                )
        assert post.await_count == 1

    async def test_account_balance(self):
        client, _ = _client_returning(httpx.Response(200, json={"balance": 2500}))
        with (
            patch.object(settings, "serper_api_key", "k"),
            patch("src.services.serper.httpx.AsyncClient", client),
        ):
            assert await serper.serper_account_balance() == 2500

    async def test_account_balance_failure_is_none(self):
        class _Boom:
            def __init__(self, *a, **kw):
                raise httpx.ConnectError("down")

        with (
            patch.object(settings, "serper_api_key", "k"),
            patch("src.services.serper.httpx.AsyncClient", _Boom),
        ):
            assert await serper.serper_account_balance() is None


# ── Run wrapper: FAILED with a reason the backend acts on ───────────────────


class TestRunFailureReason:
    @pytest.mark.parametrize(
        "exc,reason,provider",
        [
            (serper.SerperQuotaExhaustedError("x"), "search_quota_exhausted", "serper"),
            (gp.PlacesQuotaExhaustedError("x"), "search_quota_exhausted", "google_places"),
            (serper.SerperAuthError("x"), "search_auth_failed", "serper"),
            (gp.PlacesAuthError("x"), "search_auth_failed", "google_places"),
        ],
    )
    def test_search_failures_map_to_contract_reasons(self, exc, reason, provider):
        assert main.prospecting_failure(exc) == {"reason": reason, "provider": provider}

    def test_anything_else_is_an_agent_error(self):
        assert main.prospecting_failure(RuntimeError("boom")) == {
            "reason": "agent_error: boom"
        }

    async def test_exhausted_run_reports_failed_with_reason_and_usage(self):
        class _Graph:
            async def ainvoke(self, inputs, config):
                serper._count_serper_call()
                raise serper.SerperQuotaExhaustedError("Not enough credits")

            async def aget_state(self, config):
                class _S:
                    values = {"turn_usage": [{"node": "refine_queries"}]}

                return _S()

        report = AsyncMock(return_value={})
        with patch("src.main.report_prospecting_run", report):
            await main._run_prospecting(_Graph(), {}, {"configurable": {}}, "run-1")

        args, kwargs = report.await_args
        assert args == ("run-1", "FAILED")
        assert kwargs["metrics"]["reason"] == "search_quota_exhausted"
        assert kwargs["metrics"]["provider"] == "serper"
        assert kwargs["metrics"]["serper_calls"] == 1
        assert kwargs["metrics"]["places_calls"] == 0
        assert kwargs["usage"] == [{"node": "refine_queries"}]

    async def test_web_search_node_lets_exhaustion_through(self):
        serper.start_serper_accounting()
        with patch(
            "src.agents.prospecting_nodes._serper_post",
            AsyncMock(side_effect=serper.SerperQuotaExhaustedError("x")),
        ):
            with pytest.raises(serper.SerperQuotaExhaustedError):
                await pn.web_search_node(
                    {"queries": ["colegios en Medellín Colombia"]}, {"configurable": {}}
                )


# ── Google Places ───────────────────────────────────────────────────────────

# Trimmed from a real `places:searchText` answer captured 2026-10-16.
PLACES_RESPONSE = {
    "places": [
        {
            "displayName": {"text": "Colegio Benedictino de Santa María", "languageCode": "es"},
            "websiteUri": "https://www.colegiobenedictino.edu.co/",
            "nationalPhoneNumber": "(604) 3314270",
            "formattedAddress": "Cra. 40b #53B Sur - 30, Zona 7, Envigado, Antioquia",
            "businessStatus": "OPERATIONAL",
            "addressComponents": [
                {"longText": "Zona 7", "types": ["sublocality_level_1", "sublocality", "political"]},
                {"longText": "Envigado", "types": ["locality", "political"]},
                {"longText": "Envigado", "types": ["administrative_area_level_2", "political"]},
                {"longText": "Antioquia", "types": ["administrative_area_level_1", "political"]},
            ],
        },
        {
            "displayName": {"text": "Jardín Cerrado"},
            "businessStatus": "CLOSED_PERMANENTLY",
            "formattedAddress": "Medellín",
        },
        {
            "displayName": {"text": "Colegio Rural"},
            "formattedAddress": "Vereda X, Guarne, Antioquia",
            "addressComponents": [
                {"longText": "Guarne", "types": ["administrative_area_level_2", "political"]},
            ],
        },
    ]
}


class TestGooglePlaces:
    def test_parse_rows(self):
        rows = gp.parse_places_response(PLACES_RESPONSE)
        assert [r["title"] for r in rows] == [
            "Colegio Benedictino de Santa María",
            "Colegio Rural",
        ]
        first = rows[0]
        assert first["url"] == "https://www.colegiobenedictino.edu.co/"
        assert first["phone"] == "(604) 3314270"
        assert first["locality"] == "Envigado"
        assert first["source"] == "google_places"
        # Falls back to the municipality when there is no `locality`.
        assert rows[1]["locality"] == "Guarne"
        assert rows[1]["url"] == ""

    async def test_request_shape(self):
        gp.start_places_accounting()
        client, calls = _client_returning(httpx.Response(200, json=PLACES_RESPONSE))
        with (
            patch.object(settings, "google_places_api_key", "AIza-test"),
            patch("src.services.google_places.httpx.AsyncClient", client),
        ):
            rows = await gp.places_text_search("colegios en Envigado Colombia")
        assert len(rows) == 2
        call = calls[0]
        assert call["url"] == gp.TEXT_SEARCH_ENDPOINT
        assert call["headers"]["X-Goog-Api-Key"] == "AIza-test"
        assert "places.addressComponents" in call["headers"]["X-Goog-FieldMask"]
        assert call["json"]["regionCode"] == "CO"
        assert call["json"]["pageSize"] == settings.prospecting_places_page_size
        assert gp.places_call_count() == 1

    async def test_no_key_no_request(self):
        gp.start_places_accounting()
        with patch.object(settings, "google_places_api_key", ""):
            assert await gp.places_text_search("x") == []
        assert gp.places_call_count() == 0

    async def test_forbidden_is_fatal(self):
        client, calls = _client_returning(
            httpx.Response(403, json={"error": {"status": "PERMISSION_DENIED"}})
        )
        with (
            patch.object(settings, "google_places_api_key", "k"),
            patch("src.services.google_places.httpx.AsyncClient", client),
        ):
            with pytest.raises(gp.PlacesAuthError):
                await gp.places_text_search("x")
        assert len(calls) == 1

    async def test_persistent_429_is_quota_exhausted(self):
        client, calls = _client_returning(
            httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED"}})
        )
        with (
            patch.object(settings, "google_places_api_key", "k"),
            patch("src.services.google_places.httpx.AsyncClient", client),
            patch("src.services.google_places.asyncio.sleep", AsyncMock()),
        ):
            with pytest.raises(gp.PlacesQuotaExhaustedError):
                await gp.places_text_search("x")
        assert len(calls) == 3

    async def test_transient_429_recovers(self):
        client, _ = _client_returning(
            httpx.Response(429, json={}), httpx.Response(200, json=PLACES_RESPONSE)
        )
        with (
            patch.object(settings, "google_places_api_key", "k"),
            patch("src.services.google_places.httpx.AsyncClient", client),
            patch("src.services.google_places.asyncio.sleep", AsyncMock()),
        ):
            assert len(await gp.places_text_search("x")) == 2

    def test_provider_selection(self):
        with patch.object(settings, "google_places_api_key", "k"):
            assert pn.places_provider() == "google"
            with patch.object(settings, "prospecting_places_provider", "serper"):
                assert pn.places_provider() == "serper"
            with patch.object(settings, "prospecting_places_enabled", False):
                assert pn.places_provider() == "off"
        # "google" without a key falls back to Serper /places.
        with patch.object(settings, "google_places_api_key", ""):
            assert pn.places_provider() == "serper"

    async def test_web_search_uses_google_places_not_serper_places(self):
        serper.start_serper_accounting()
        serper_places = AsyncMock(return_value=[])
        google = AsyncMock(return_value=gp.parse_places_response(PLACES_RESPONSE))
        with (
            patch.object(settings, "google_places_api_key", "k"),
            patch("src.agents.prospecting_nodes._serper_search", AsyncMock(return_value=[])),
            patch("src.agents.prospecting_nodes._serper_places", serper_places),
            patch("src.agents.prospecting_nodes.places_text_search", google),
        ):
            out = await pn.web_search_node(
                {"queries": ["colegios en Medellín Colombia"]}, {"configurable": {}}
            )
        serper_places.assert_not_awaited()
        assert google.await_count == 1
        assert len(out["search_results"]) == 2
        assert out["metrics"]["places_provider"] == "google"


# ── Domain blocklist ────────────────────────────────────────────────────────


class TestBlocklist:
    @pytest.mark.parametrize(
        "url,blocked",
        [
            ("https://www.paginasamarillas.com.co/cali/servicios/x", True),
            ("https://co.computrabajo.com/empleos", True),
            ("https://es.scribd.com/document/1", True),
            ("https://www.cali.gov.co/educacion", True),
            ("https://www.waze.com/live-map", True),
            ("https://colegiobenedictino.edu.co/", False),
            ("https://www.facebook.com/colegio", False),
            ("", False),
        ],
    )
    def test_is_blocked_url(self, url, blocked):
        assert pn.is_blocked_url(url) is blocked

    def test_env_extends_the_list(self):
        with patch.object(settings, "prospecting_blocked_domains", "directorio.com.co, x.co"):
            assert pn.is_blocked_url("https://www.directorio.com.co/a") is True

    async def test_blocked_organic_never_reaches_extraction(self):
        organic = [
            {"title": "PH Cali", "url": "https://www.paginasamarillas.com.co/cali/x", "snippet": ""},
            {"title": "Conjunto", "url": "https://conjunto.com.co/", "snippet": ""},
        ]
        listing = [
            {
                "title": "Administradora PH",
                "url": "https://www.paginasamarillas.com.co/empresas/a",
                "snippet": "Cali",
                "phone": "602 555",
                "address": "Cali",
            }
        ]
        with (
            patch("src.agents.prospecting_nodes._serper_search", AsyncMock(return_value=organic)),
            patch("src.agents.prospecting_nodes._places_search", AsyncMock(return_value=listing)),
        ):
            out = await pn.web_search_node(
                {"queries": ["administración PH en Cali Colombia"]}, {"configurable": {}}
            )
        titles = {r["title"]: r for r in out["search_results"]}
        assert "PH Cali" not in titles
        assert titles["Conjunto"]["url"] == "https://conjunto.com.co/"
        # The listing survives — the business exists — without the directory URL.
        assert titles["Administradora PH"]["url"] == ""
        assert titles["Administradora PH"]["phone"] == "602 555"
        assert out["metrics"]["blocked_results"] == 2

    async def test_social_page_skips_the_fetch_but_still_extracts(self):
        fetch = AsyncMock(return_value="")

        class _Provider:
            name = "fake"
            last_usage = UsageInfo()

            async def stream_chat(self, model, messages):
                yield '{"is_match": true, "name": "Colegio X", "city": "Cali"}'

        with (
            patch("src.agents.prospecting_nodes._fetch_page_text", fetch),
            patch("src.agents.prospecting_nodes.get_provider", return_value=_Provider()),
        ):
            out = await pn.extract_and_enrich_node(
                {
                    "result": {
                        "title": "Colegio X - Facebook",
                        "url": "https://www.facebook.com/colegiox",
                        "snippet": "Colegio en Cali",
                    },
                    "location": pn.DEFAULT_LOCATION,
                },
                {"configurable": {"llm_model": "m"}},
            )
        fetch.assert_not_awaited()
        assert out["candidates"][0]["customName"] == "Colegio X"


# ── Metropolitan areas ──────────────────────────────────────────────────────


class TestMetroAreas:
    MEDELLIN = {"country": "Colombia", "gl": "co", "cities": ["Medellín"]}

    @pytest.mark.parametrize(
        "city", ["Envigado", "Sabaneta, Antioquia", "Itagui", "Bello", "Medellín"]
    )
    def test_metro_municipality_is_in_region(self, city):
        assert pn.city_in_region(city, self.MEDELLIN) is True

    def test_other_metro_is_out(self):
        assert pn.city_in_region("Soacha", self.MEDELLIN) is False
        assert pn.city_in_region("Yumbo", {**self.MEDELLIN, "cities": ["Cali"]}) is True

    def test_only_in_colombia(self):
        mexico = {"country": "México", "gl": "mx", "cities": ["Medellín"]}
        assert pn.city_in_region("Envigado", mexico) is False

    def test_unknown_stays_unknown(self):
        assert pn.city_in_region(None, self.MEDELLIN) is None

    async def test_listing_locality_beats_the_llm_city(self):
        class _Provider:
            name = "fake"
            last_usage = UsageInfo()

            async def stream_chat(self, model, messages):
                # The page header says Bogotá (a head office), the listing says Envigado.
                yield '{"is_match": true, "name": "Colegio Y", "city": "Bogotá"}'

        with (
            patch("src.agents.prospecting_nodes._fetch_page_text", AsyncMock(return_value="x")),
            patch("src.agents.prospecting_nodes.get_provider", return_value=_Provider()),
        ):
            out = await pn.extract_and_enrich_node(
                {
                    "result": {
                        "title": "Colegio Y",
                        "url": "https://colegioy.edu.co",
                        "locality": "Envigado",
                    },
                    "location": self.MEDELLIN,
                },
                {"configurable": {"llm_model": "m"}},
            )
        cand = out["candidates"][0]
        assert cand["city"] == "Envigado"
        assert cand["outOfRegion"] is False


# ── Credit control ──────────────────────────────────────────────────────────


class TestCreditControl:
    async def test_budget_stops_issuing_queries(self):
        serper.start_serper_accounting()

        async def _search(query, location=None):
            serper._count_serper_call()
            serper._count_serper_call()
            return []

        search = AsyncMock(side_effect=_search)
        with (
            patch.object(settings, "prospecting_max_serper_calls_per_run", 4),
            patch("src.agents.prospecting_nodes._serper_search", search),
            patch("src.agents.prospecting_nodes._places_search", AsyncMock(return_value=[])),
        ):
            out = await pn.web_search_node(
                {"queries": [f"q{i} en Cali Colombia" for i in range(5)]},
                {"configurable": {}},
            )
        assert search.await_count == 2
        assert len(out["searched_queries"]) == 2

    async def test_report_carries_balance_places_and_extract_failures(self):
        serper.start_serper_accounting()
        gp.start_places_accounting()
        gp._count_places_call()
        reported: dict = {}

        async def _capture(run_id, status, metrics=None, usage=None):
            reported.update(metrics or {})

        with (
            patch(
                "src.agents.prospecting_nodes.backend_client.report_prospecting_run",
                AsyncMock(side_effect=_capture),
            ),
            patch(
                "src.agents.prospecting_nodes.serper_account_balance",
                AsyncMock(return_value=120),
            ),
        ):
            await pn.report_node(
                {
                    "run_id": "r1",
                    "turn_usage": [],
                    "extract_failures": ["rate_limited", "error", "rate_limited"],
                },
                {"configurable": {}},
            )
        assert reported["serper_balance"] == 120
        assert reported["places_calls"] == 1
        assert reported["extract_failed"] == 3
        assert reported["extract_rate_limited"] == 2
