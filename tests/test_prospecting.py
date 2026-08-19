"""Tests for the prospecting graph (aurora) — dedupe normalization, the Send
fan-out, the create/report contract, and the self-improvement mechanisms
(strategy memory, the refinement loop, human-feedback injection, niche/location
targeting, and the Serper /search + /places combine).

MemorySaver + mocked backend/search/LLM calls, driving astream() and inspecting
the node-update sequence (mirrors test_leads_graph.py). Strategy-memory tests use
langgraph.store.memory.InMemoryStore via registry.set_store — no DB needed.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from src.agents import prospecting_nodes as pn
from src.agents.dedup import (
    normalize_domain,
    normalize_name,
    prospect_external_id,
)
from src.agents.prospecting_nodes import (
    _is_quality,
    _plan_queries,
    _render_feedback_block,
    city_in_region,
    clean_email,
    clean_website,
    evaluate_quality_node,
    extract_and_enrich_node,
    load_memory_node,
    query_matches_niche,
    serper_location_for,
    web_search_node,
)
from src.config import settings
from src.graphs import registry
from src.graphs.prospecting_graph import build_prospecting_graph


# ── Normalization / identity ────────────────────────────────────────────────


class TestDedupNormalization:
    def test_accent_case_punctuation_fold(self):
        assert normalize_name("Colegio San José!!") == "colegio san jose"
        assert normalize_name("  COLEGIO   san  jose ") == "colegio san jose"

    def test_external_id_is_stable_across_variants(self):
        a = prospect_external_id("Colegio San José")
        b = prospect_external_id("colegio san jose")
        assert a == b
        assert a.startswith("prospector:")

    def test_external_id_empty_for_blank_name(self):
        assert prospect_external_id("") == ""
        assert prospect_external_id("   ") == ""

    def test_domain_normalization(self):
        assert normalize_domain("https://www.colegio.edu.co/inicio") == "colegio.edu.co"
        assert normalize_domain("colegio.edu.co") == "colegio.edu.co"
        assert normalize_domain("http://sub.colegio.com?x=1") == "colegio.com"
        assert normalize_domain("") == ""


# ── Query planning (niche + location + strategy memory) ─────────────────────


# aurora is industry-agnostic: there is no built-in niche, so every test that
# plans queries supplies its own.
TEST_NICHE = {
    "key": "gimnasios",
    "label": "gimnasios y centros deportivos",
    "search_terms": ["gimnasios", "centros deportivos"],
}


def _plan(run_date, **kw):
    kw.setdefault("niche", TEST_NICHE)
    return _plan_queries(run_date, **kw)


class TestPlanQueries:
    def test_rotation_differs_by_run_date(self):
        q1 = _plan("2026-07-21")
        q2 = _plan("2026-07-22")
        assert q1 and q2
        # Different dates should not start on the same city query.
        assert q1[0] != q2[0]

    def test_respects_search_budget(self):
        queries = _plan("2026-07-21")
        assert len(queries) <= settings.prospecting_max_searches

    def test_no_niche_plans_nothing(self):
        # The deliberate dead end: without a tenant-configured niche there is
        # nothing to search for, and inventing one would prospect an industry
        # the tenant never asked for.
        assert _plan_queries("2026-07-21") == []
        assert _plan_queries("2026-07-21", niche={"key": "x", "search_terms": []}) == []

    def test_niche_and_location_drive_queries(self):
        queries = _plan_queries(
            "2026-07-21",
            niche={"key": "gyms", "label": "gimnasios", "search_terms": ["gimnasios"]},
            location={"country": "México", "cities": ["Guadalajara"]},
        )
        assert "gimnasios en Guadalajara México" in queries

    def test_best_queries_seeded_first_and_avoid_filters_seeds(self):
        queries = _plan(
            "2026-07-21",
            strategy={"best_queries": ["gimnasios que sirven", "gimnasios malos"],
                      "avoid_patterns": ["gimnasios malos"]},
        )
        assert queries[0] == "gimnasios que sirven"
        assert "gimnasios malos" not in queries

    def test_avoid_never_filters_the_deterministic_floor(self):
        # An avoid pattern equal to a floor query must NOT remove it — memory can
        # only augment, never empty, the discovery surface.
        floor = _plan("2026-07-21")
        target = floor[0]
        with_avoid = _plan("2026-07-21", strategy={"avoid_patterns": [target]})
        assert target in with_avoid


# ── Human-feedback rendering ────────────────────────────────────────────────


class TestFeedbackBlock:
    def test_empty_when_no_feedback(self):
        assert _render_feedback_block([], 6) == ""

    def test_renders_good_and_bad_examples(self):
        block = _render_feedback_block(
            [
                {"verdict": "GOOD", "customName": "Colegio Bien", "note": "buen fit"},
                {"verdict": "BAD", "customName": "Tienda X"},
            ],
            6,
        )
        assert "BUENO: Colegio Bien" in block
        assert "buen fit" in block
        assert "MALO: Tienda X" in block

    def test_caps_per_verdict(self):
        fb = [{"verdict": "GOOD", "customName": f"C{i}"} for i in range(20)]
        block = _render_feedback_block(fb, 3)
        assert block.count("BUENO:") == 3


# ── Graph flow helpers ──────────────────────────────────────────────────────


def _config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id, "openai_api_key": "sk-test"},
        "recursion_limit": 40,
    }


def _node_names(chunks: list) -> list[str]:
    out: list[str] = []
    for c in chunks:
        if not isinstance(c, dict):
            continue
        for key in c.keys():
            if not key.startswith("__"):
                out.append(key)
    return out


async def _run(graph, inputs: dict, thread_id: str) -> list[dict]:
    chunks = []
    async for c in graph.astream(
        inputs, config=_config(thread_id), stream_mode="updates"
    ):
        chunks.append(c)
    return chunks


# Patches shared by the single-pass graph tests: search endpoints, feedback read,
# and the backend calls. `max_iterations=1` disables the refinement loop so these
# assert the classic single-pass behaviour.
def _single_pass_patches(search_mock, extract_mock=None, dedup=None, create=None):
    patches = [
        patch.object(settings, "prospecting_max_iterations", 1),
        patch("src.agents.prospecting_nodes._serper_search", search_mock),
        patch("src.agents.prospecting_nodes._serper_places", AsyncMock(return_value=[])),
        patch(
            "src.agents.prospecting_nodes.backend_client.get_prospect_feedback",
            AsyncMock(return_value=[]),
        ),
        patch(
            "src.agents.prospecting_nodes.backend_client.check_prospect_duplicates",
            dedup or AsyncMock(return_value={}),
        ),
        patch(
            "src.agents.prospecting_nodes.backend_client.report_prospecting_run",
            AsyncMock(return_value={"ok": True}),
        ),
    ]
    if extract_mock is not None:
        patches.append(
            patch("src.graphs.prospecting_graph.extract_and_enrich_node", extract_mock)
        )
    if create is not None:
        patches.append(
            patch(
                "src.agents.prospecting_nodes.backend_client.create_prospect_contact",
                create,
            )
        )
    return patches


class TestGraphFlow:
    async def test_send_fanout_one_extract_per_result(self):
        results = [
            {"title": "A", "url": "https://a.example.co", "snippet": ""},
            {"title": "B", "url": "https://b.example.co", "snippet": ""},
            {"title": "C", "url": "https://c.example.co", "snippet": ""},
        ]

        async def fake_search(query, *args):
            return results if "gimnasios" in query else []

        async def fake_extract(state, *args):
            return {}

        import contextlib

        with contextlib.ExitStack() as stack:
            for p in _single_pass_patches(
                AsyncMock(side_effect=fake_search),
                extract_mock=AsyncMock(side_effect=fake_extract),
            ):
                stack.enter_context(p)
            graph = build_prospecting_graph(MemorySaver())
            chunks = await _run(
                graph,
                {"tenant_id": "t1", "run_id": "r1", "run_date": "2026-07-21",
                 "niche": TEST_NICHE},
                "prospecting:t1:2026-07-21",
            )

        names = _node_names(chunks)
        assert names.count("extract_and_enrich") == len(results)
        assert "dedupe_check" in names
        assert "evaluate_quality" in names
        assert "report" in names

    async def test_zero_results_still_reports(self):
        report_mock = AsyncMock(return_value={"ok": True})
        import contextlib

        with contextlib.ExitStack() as stack:
            for p in _single_pass_patches(AsyncMock(return_value=[])):
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    "src.agents.prospecting_nodes.backend_client.report_prospecting_run",
                    report_mock,
                )
            )
            graph = build_prospecting_graph(MemorySaver())
            chunks = await _run(
                graph,
                {"tenant_id": "t1", "run_id": "r0", "run_date": "2026-07-21",
                 "niche": TEST_NICHE},
                "prospecting:t1:zero",
            )

        names = _node_names(chunks)
        assert "dedupe_check" in names
        assert "report" in names
        report_mock.assert_awaited_once()
        assert report_mock.await_args.args[1] == "COMPLETED"

    async def test_survivors_created_and_duplicates_skipped(self):
        cand_a = {
            "externalId": prospect_external_id("Colegio A"),
            "customName": "Colegio A",
            "normalizedName": normalize_name("Colegio A"),
            "email": None,
            "website": "https://a.edu.co",
            "domain": "a.edu.co",
            "city": "Bogotá",
            "notes": None,
            "sourceUrl": "https://a.edu.co",
            "is_match": True,
            "query": "q",
        }
        cand_b = {**cand_a, "externalId": prospect_external_id("Colegio B"),
                  "customName": "Colegio B"}

        async def fake_extract(state, *args):
            url = (state.get("result") or {}).get("url", "")
            return {"candidates": [cand_a if "a" in url else cand_b]}

        create_mock = AsyncMock(
            return_value={"ok": True, "contactId": "c1", "deduped": False}
        )
        dedup_mock = AsyncMock(
            return_value={cand_a["externalId"]: {"exists": True, "reason": "identity"}}
        )
        search_mock = AsyncMock(
            return_value=[
                {"title": "A", "url": "https://a.edu.co", "snippet": ""},
                {"title": "B", "url": "https://b.edu.co", "snippet": ""},
            ]
        )
        import contextlib

        with contextlib.ExitStack() as stack:
            for p in _single_pass_patches(
                search_mock,
                extract_mock=AsyncMock(side_effect=fake_extract),
                dedup=dedup_mock,
                create=create_mock,
            ):
                stack.enter_context(p)
            graph = build_prospecting_graph(MemorySaver())
            await _run(
                graph,
                {"tenant_id": "t1", "run_id": "r2", "run_date": "2026-07-21",
                 "niche": TEST_NICHE},
                "prospecting:t1:create",
            )

        create_mock.assert_awaited_once()
        created_body = create_mock.await_args.args[1]
        assert created_body["externalId"] == cand_b["externalId"]


# ── Refinement loop ─────────────────────────────────────────────────────────


class TestRefinementLoop:
    async def test_loops_then_stops_at_iteration_cap(self):
        async def fake_search(query, *args):
            return [{"title": "X", "url": f"https://{abs(hash(query)) % 997}.edu.co",
                     "snippet": ""}]

        async def fake_extract(state, *args):
            name = f"Colegio {abs(hash((state.get('result') or {}).get('url',''))) % 9999}"
            return {"candidates": [{
                "externalId": prospect_external_id(name),
                "customName": name,
                "normalizedName": normalize_name(name),
                "email": "info@x.edu.co",
                "website": "https://x.edu.co",
                "domain": "x.edu.co",
                "city": None, "notes": None, "sourceUrl": "https://x.edu.co",
                "is_match": True, "query": state.get("query", ""),
            }]}

        async def fake_refine(state, *args):
            return {"iteration": int(state.get("iteration", 0)) + 1,
                    "queries": [f"refined-{state.get('iteration')}"],
                    "search_results": []}

        import contextlib

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(settings, "prospecting_max_iterations", 2))
            stack.enter_context(
                patch.object(settings, "prospecting_min_quality_prospects", 100)
            )
            stack.enter_context(
                patch("src.agents.prospecting_nodes._serper_search",
                      AsyncMock(side_effect=fake_search))
            )
            stack.enter_context(
                patch("src.agents.prospecting_nodes._serper_places",
                      AsyncMock(return_value=[]))
            )
            stack.enter_context(
                patch("src.graphs.prospecting_graph.extract_and_enrich_node",
                      AsyncMock(side_effect=fake_extract))
            )
            stack.enter_context(
                patch("src.graphs.prospecting_graph.refine_queries_node",
                      AsyncMock(side_effect=fake_refine))
            )
            for p in [
                patch("src.agents.prospecting_nodes.backend_client.get_prospect_feedback",
                      AsyncMock(return_value=[])),
                patch("src.agents.prospecting_nodes.backend_client.check_prospect_duplicates",
                      AsyncMock(return_value={})),
                patch("src.agents.prospecting_nodes.backend_client.create_prospect_contact",
                      AsyncMock(return_value={"ok": True, "contactId": "c", "deduped": False})),
                patch("src.agents.prospecting_nodes.backend_client.report_prospecting_run",
                      AsyncMock(return_value={"ok": True})),
            ]:
                stack.enter_context(p)
            graph = build_prospecting_graph(MemorySaver())
            chunks = await _run(
                graph,
                {"tenant_id": "t1", "run_id": "rl", "run_date": "2026-07-21",
                 "niche": TEST_NICHE},
                "prospecting:t1:loop",
            )

        names = _node_names(chunks)
        # Exactly two search passes (max_iterations=2) and one refine in between.
        assert names.count("web_search") == 2
        assert names.count("refine_queries") == 1
        assert "create_contacts" in names


# ── Strategy memory (InMemoryStore) ─────────────────────────────────────────


class TestStrategyMemory:
    async def test_reflect_persists_and_load_reads_back(self):
        store = InMemoryStore()
        query_tag = "gimnasios en Bogotá Colombia"

        async def fake_extract(state, *args):
            name = "Colegio Persistente"
            return {"candidates": [{
                "externalId": prospect_external_id(name),
                "customName": name,
                "normalizedName": normalize_name(name),
                "email": "info@persistente.edu.co",
                "website": "https://persistente.edu.co",
                "domain": "persistente.edu.co",
                "city": None, "notes": None,
                "sourceUrl": "https://persistente.edu.co",
                "is_match": True, "query": query_tag,
            }]}

        import contextlib

        registry.set_store(store)
        try:
            with contextlib.ExitStack() as stack:
                for p in _single_pass_patches(
                    AsyncMock(return_value=[
                        {"title": "P", "url": "https://persistente.edu.co", "snippet": ""}
                    ]),
                    extract_mock=AsyncMock(side_effect=fake_extract),
                    create=AsyncMock(
                        return_value={"ok": True, "contactId": "c", "deduped": False}
                    ),
                ):
                    stack.enter_context(p)
                graph = build_prospecting_graph(MemorySaver())
                await _run(
                    graph,
                    {"tenant_id": "t1", "run_id": "rm", "run_date": "2026-07-21",
                 "niche": TEST_NICHE},
                    "prospecting:t1:mem",
                )

            item = await store.aget(("t1", TEST_NICHE["key"], "strategy"), "latest")
            assert item is not None
            assert query_tag in item.value["best_queries"]
            assert item.value["runs_observed"] == 1

            # And plan_searches seeds those best queries first on the next run.
            planned = _plan("2026-07-21", strategy=item.value)
            assert planned[0] == query_tag
        finally:
            registry.set_store(None)

    async def test_missing_store_degrades_gracefully(self):
        # load_memory with no store returns empty strategy, never raises.
        registry.set_store(None)
        with patch(
            "src.agents.prospecting_nodes.backend_client.get_prospect_feedback",
            AsyncMock(return_value=[]),
        ):
            out = await load_memory_node(
                {"tenant_id": "t1", "niche": TEST_NICHE}, {"configurable": {}}
            )
        assert out["strategy_memory"] == {}
        assert out["iteration"] == 0


# ── Feedback injection into the extraction prompt ───────────────────────────


class TestFeedbackInjection:
    async def test_load_memory_wires_feedback_block(self):
        registry.set_store(None)
        with patch(
            "src.agents.prospecting_nodes.backend_client.get_prospect_feedback",
            AsyncMock(return_value=[
                {"verdict": "GOOD", "customName": "Colegio Ideal", "note": "match"}
            ]),
        ):
            out = await load_memory_node(
                {"tenant_id": "t1", "niche": TEST_NICHE}, {"configurable": {}}
            )
        assert "Colegio Ideal" in out["feedback_block"]

    async def test_extract_injects_feedback_and_niche_into_prompt(self):
        captured = {}

        class FakeProvider:
            name = "openai"
            last_usage = SimpleNamespace(
                input_tokens=1, output_tokens=1,
                cached_input_tokens=0, reasoning_tokens=0,
            )

            async def stream_chat(self, model, messages):
                captured["messages"] = messages
                yield ('{"name": "Colegio Z", "is_match": true, "email": '
                       '"a@b.co", "website": null, "city": null, "notes": null}')

        with (
            patch("src.agents.prospecting_nodes._fetch_page_text",
                  AsyncMock(return_value="contenido de la página")),
            patch("src.agents.prospecting_nodes.get_provider",
                  return_value=FakeProvider()),
            patch("src.agents.prospecting_nodes.resolve_model", return_value="gpt-4o"),
        ):
            out = await extract_and_enrich_node(
                {
                    "result": {"url": "https://z.edu.co", "title": "", "snippet": ""},
                    "tenant_id": "t1",
                    "niche_label": "colegios privados",
                    "feedback_block": "EJEMPLOS DE REVISIÓN HUMANA:\n- BUENO: Colegio Ideal",
                    "query": "q-feedback",
                },
                {"configurable": {}},
            )

        system = captured["messages"][0]["content"]
        assert "colegios privados" in system  # niche context injected
        assert "Colegio Ideal" in system  # human feedback injected
        assert out["candidates"][0]["query"] == "q-feedback"
        assert out["candidates"][0]["is_match"] is True

    async def test_legacy_is_school_key_is_no_longer_accepted(self):
        class FakeProvider:
            name = "openai"
            last_usage = SimpleNamespace(
                input_tokens=1, output_tokens=1,
                cached_input_tokens=0, reasoning_tokens=0,
            )

            async def stream_chat(self, model, messages):
                yield ('{"name": "Org Legacy", "is_school": true, '
                       '"email": null, "website": "https://l.example.co", '
                       '"city": null, "notes": null}')

        with (
            patch("src.agents.prospecting_nodes._fetch_page_text",
                  AsyncMock(return_value="pagina")),
            patch("src.agents.prospecting_nodes.get_provider",
                  return_value=FakeProvider()),
            patch("src.agents.prospecting_nodes.resolve_model", return_value="gpt-4o"),
        ):
            out = await extract_and_enrich_node(
                {"result": {"url": "https://l.example.co"}, "niche_label": "gimnasios"},
                {"configurable": {}},
            )
        # No candidate: only `is_match` counts now. Usage is still recorded — the
        # LLM call happened.
        assert "candidates" not in out
        assert len(out["turn_usage"]) == 1


# ── Serper /search + /places combine ────────────────────────────────────────


class TestSearchPlacesCombine:
    async def test_union_of_search_and_places_tagged_with_query(self):
        with (
            patch("src.agents.prospecting_nodes._serper_search",
                  AsyncMock(return_value=[
                      {"title": "Web", "url": "https://web.edu.co", "snippet": ""}])),
            patch("src.agents.prospecting_nodes._serper_places",
                  AsyncMock(return_value=[
                      {"title": "Local", "url": "https://local.edu.co",
                       "snippet": "Calle 1", "phone": "123", "address": "Calle 1"}])),
        ):
            out = await web_search_node(
                {"queries": ["q1"], "location": {}}, {"configurable": {}}
            )
        urls = {r["url"] for r in out["search_results"]}
        assert urls == {"https://web.edu.co", "https://local.edu.co"}
        assert all(r["query"] == "q1" for r in out["search_results"])
        assert out["searched_queries"] == ["q1"]

    async def test_tolerates_one_endpoint_returning_nothing(self):
        # /search empty, /places still yields → node stands.
        with (
            patch("src.agents.prospecting_nodes._serper_search",
                  AsyncMock(return_value=[])),
            patch("src.agents.prospecting_nodes._serper_places",
                  AsyncMock(return_value=[
                      {"title": "Local", "url": "https://only-places.edu.co",
                       "snippet": "", "phone": "", "address": ""}])),
        ):
            out = await web_search_node(
                {"queries": ["q1"], "location": {}}, {"configurable": {}}
            )
        assert [r["url"] for r in out["search_results"]] == ["https://only-places.edu.co"]

    async def test_places_disabled_returns_empty(self):
        with patch.object(settings, "prospecting_places_enabled", False):
            out = await pn._serper_places("q", {})
        assert out == []


# ── evaluate_quality threshold ──────────────────────────────────────────────


class TestEvaluateQuality:
    async def test_counts_only_contactable_survivors(self):
        survivors = [
            {"email": "a@b.co", "domain": "b.co"},   # quality
            {"email": None, "domain": "c.co"},        # quality (website)
            {"email": None, "domain": ""},            # not quality
        ]
        with patch.object(settings, "prospecting_min_quality_prospects", 2):
            out = await evaluate_quality_node(
                {"deduped_candidates": survivors}, {"configurable": {}}
            )
        assert out["quality_count"] == 2
        assert out["enough_quality"] is True


# ── Serper credit accounting ─────────────────────────────────────────────────


class TestSerperAccounting:
    """`_serper_post` is the single billable call site, so it is the only place
    the counter moves — including retries, which Serper charges for too."""

    async def test_counts_every_attempt_including_retries(self):
        pn.start_serper_accounting()

        class _FlakyClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **kw):
                raise httpx.ConnectError("boom")

        with (
            patch.object(settings, "serper_api_key", "k"),
            patch("src.services.serper.httpx.AsyncClient", _FlakyClient),
            patch("src.services.serper.asyncio.sleep", AsyncMock()),
        ):
            assert await pn._serper_post("https://x/search", {}) is None

        # Three attempts were made, so three credits were spent.
        assert pn.serper_call_count() == 3

    async def test_rejected_key_raises_once_instead_of_retrying(self):
        """401/403 is a verdict on the credential, not a transient failure: it
        must abort the run (so the CRM shows a real reason instead of a
        0-result success) and must not be retried, since Serper bills retries.
        """
        pn.start_serper_accounting()

        class _ForbiddenClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **kw):
                return httpx.Response(403, text="Forbidden")

        with (
            patch.object(settings, "serper_api_key", "bad-key"),
            patch("src.services.serper.httpx.AsyncClient", _ForbiddenClient),
            patch("src.services.serper.asyncio.sleep", AsyncMock()),
        ):
            with pytest.raises(pn.SerperAuthError):
                await pn._serper_post("https://x/search", {})

        # One attempt, one credit — no backoff loop against a dead credential.
        assert pn.serper_call_count() == 1

    async def test_rejected_key_fails_the_whole_search_node(self):
        """The node deliberately does not degrade to []: a bad key fails every
        query, and a silent empty run is indistinguishable from an empty niche."""
        pn.start_serper_accounting()

        with (
            patch.object(settings, "serper_api_key", "bad-key"),
            patch(
                "src.agents.prospecting_nodes._serper_post",
                AsyncMock(side_effect=pn.SerperAuthError("403")),
            ),
        ):
            with pytest.raises(pn.SerperAuthError):
                await pn.web_search_node(
                    {"queries": ["colegios privados en Chía Colombia"]},
                    {"configurable": {}},
                )

    async def test_no_key_means_no_request_and_no_credit(self):
        pn.start_serper_accounting()
        with patch.object(settings, "serper_api_key", ""):
            assert await pn._serper_post("https://x/search", {}) is None
        assert pn.serper_call_count() == 0

    async def test_counter_is_a_silent_noop_when_accounting_never_started(self):
        # A graph invoked without the run wrapper (tests, Studio) must not crash.
        pn._serper_calls.set(None)
        with patch.object(settings, "serper_api_key", ""):
            await pn._serper_post("https://x/search", {})
        assert pn.serper_call_count() == 0

    async def test_report_node_forwards_the_count(self):
        pn.start_serper_accounting()
        pn._count_serper_call()
        pn._count_serper_call()

        reported: dict = {}

        async def _capture(run_id, status, metrics=None, usage=None):
            reported.update(metrics or {})

        with patch(
            "src.agents.prospecting_nodes.backend_client.report_prospecting_run",
            AsyncMock(side_effect=_capture),
        ):
            out = await pn.report_node(
                {"run_id": "r1", "turn_usage": []}, {"configurable": {}}
            )

        assert reported["serper_calls"] == 2
        assert out["metrics"]["serper_calls"] == 2


# ── Region targeting ────────────────────────────────────────────────────────

SABANA = {
    "country": "Colombia",
    "gl": "co",
    "hl": "es",
    "cities": ["Chía", "Cajicá", "Sopó", "Tenjo", "Tabio", "Zipaquirá"],
}


class TestCityInRegion:
    def test_accent_and_case_insensitive(self):
        # The CRM holds both spellings for the same municipality; a raw ==
        # would flag half the real matches as drift.
        assert city_in_region("Chia", SABANA) is True
        assert city_in_region("CHÍA", SABANA) is True
        assert city_in_region("cajica", SABANA) is True

    def test_qualified_city_string_still_matches(self):
        assert city_in_region("Zipaquirá, Cundinamarca", SABANA) is True
        assert city_in_region("Chía (Sabana Centro)", SABANA) is True

    def test_other_city_is_out_of_region(self):
        assert city_in_region("Manizales", SABANA) is False
        assert city_in_region("Bogotá", SABANA) is False
        assert city_in_region("Cartagena de Indias", SABANA) is False

    def test_unknown_is_none_not_false(self):
        # "Couldn't place it" must never be reported as "placed elsewhere".
        assert city_in_region(None, SABANA) is None
        assert city_in_region("   ", SABANA) is None
        assert city_in_region("Chía", {"cities": []}) is None


class TestSerperGeoTargeting:
    def test_derives_location_from_city_named_in_query(self):
        assert (
            serper_location_for("Colegio en Chía Colombia", SABANA)
            == "Chía, Colombia"
        )

    def test_longest_city_match_wins(self):
        loc = {"country": "Colombia", "cities": ["Marta", "Santa Marta"]}
        assert serper_location_for("colegios en Santa Marta", loc) == (
            "Santa Marta, Colombia"
        )

    def test_none_when_query_names_no_configured_city(self):
        assert serper_location_for("colegios privados Colombia", SABANA) is None

    def test_disabled_by_setting(self, monkeypatch):
        monkeypatch.setattr(settings, "prospecting_geo_targeting_enabled", False)
        assert serper_location_for("Colegio en Chía Colombia", SABANA) is None


class TestQualityExcludesDrift:
    def test_out_of_region_never_counts_as_quality(self):
        # Otherwise a run full of Manizales schools looks "good enough" and the
        # refinement loop never fires.
        assert _is_quality({"email": "a@b.co", "outOfRegion": True}) is False
        assert _is_quality({"email": "a@b.co", "outOfRegion": False}) is True
        assert _is_quality({"domain": "b.co"}) is True


class TestExtractRegionTagging:
    def _provider(self, city):
        city_json = "null" if city is None else f'"{city}"'

        class FakeProvider:
            name = "openai"
            last_usage = SimpleNamespace(
                input_tokens=1, output_tokens=1,
                cached_input_tokens=0, reasoning_tokens=0,
            )

            async def stream_chat(self, model, messages):
                self.messages = messages
                yield (
                    '{"name": "Colegio X", "is_match": true, "email": "a@b.co",'
                    f' "website": null, "city": {city_json}, "notes": null}}'
                )

        return FakeProvider()

    async def _extract(self, city, location=SABANA):
        provider = self._provider(city)
        with (
            patch("src.agents.prospecting_nodes._fetch_page_text",
                  AsyncMock(return_value="contenido")),
            patch("src.agents.prospecting_nodes.get_provider",
                  return_value=provider),
            patch("src.agents.prospecting_nodes.resolve_model",
                  return_value="gpt-4o"),
        ):
            out = await extract_and_enrich_node(
                {
                    "result": {"url": "https://x.edu.co"},
                    "tenant_id": "t1",
                    "niche_label": "colegios",
                    "location": location,
                    "query": "q",
                },
                {"configurable": {}},
            )
        return out, provider

    async def test_target_cities_injected_into_prompt(self):
        _, provider = await self._extract("Chía")
        system = provider.messages[0]["content"]
        assert "Zipaquirá" in system and "REGIÓN OBJETIVO" in system

    async def test_in_region_candidate_not_flagged(self):
        out, _ = await self._extract("Chía")
        assert out["candidates"][0]["outOfRegion"] is False

    async def test_out_of_region_candidate_flagged(self):
        # The real-world failure: a national directory page ranked for a Chía
        # query and the LLM correctly reports the org sits in Manizales.
        out, _ = await self._extract("Manizales")
        assert out["candidates"][0]["outOfRegion"] is True

    async def test_unknown_city_is_not_flagged(self):
        out, _ = await self._extract(None)
        assert out["candidates"][0]["outOfRegion"] is False


class TestFieldSanitization:
    """A single malformed field 400s the whole contact backend-side, so these
    are dropped/repaired at the source rather than costing us the prospect."""

    def test_schemeless_website_gets_https(self):
        assert clean_website("colegio.edu.co") == "https://colegio.edu.co"
        assert clean_website("www.colegio.edu.co/inicio") == (
            "https://www.colegio.edu.co/inicio"
        )

    def test_existing_scheme_preserved(self):
        assert clean_website("http://a.edu.co/x") == "http://a.edu.co/x"

    def test_unusable_website_dropped_not_guessed(self):
        assert clean_website(None) is None
        assert clean_website("  ") is None
        assert clean_website("no es una url") is None
        assert clean_website("localhost") is None
        assert clean_website("ftp://a.edu.co") is None

    def test_trailing_punctuation_stripped(self):
        assert clean_website("colegio.edu.co.") == "https://colegio.edu.co"

    def test_valid_email_kept_partial_dropped(self):
        assert clean_email("info@colegio.edu.co") == "info@colegio.edu.co"
        assert clean_email("mailto:info@colegio.edu.co") == "info@colegio.edu.co"
        assert clean_email("info@colegio") is None
        assert clean_email("no-es-email") is None
        assert clean_email(None) is None


class TestRunIsolation:
    """`candidates`/`searched_queries`/`seen_urls` are `operator.add` channels, so
    two runs sharing a checkpoint thread pool their candidates — the second run
    re-posts the first's prospects and re-inflates its own metrics. Nothing in
    the graph can undo that (an input value cannot reset a reduced channel), so
    the thread key is the only guard."""

    def _thread_id(self, run_id: str, run_date: str) -> str:
        from src.main import prospecting_thread_id

        return prospecting_thread_id("t1", run_id, run_date)

    def test_same_day_runs_do_not_share_a_thread(self):
        a = self._thread_id("run-a", "2026-07-28")
        b = self._thread_id("run-b", "2026-07-28")
        assert a != b

    def test_a_retry_of_the_same_run_resumes_its_thread(self):
        # retryFailedRun re-arms the SAME run row, so resume must still work.
        assert self._thread_id("run-a", "2026-07-28") == self._thread_id(
            "run-a", "2026-07-28"
        )

    def test_falls_back_to_run_date_when_run_id_is_absent(self):
        assert self._thread_id("", "2026-07-28") == "prospecting:t1:2026-07-28"


# ── Niche anchoring (industry drift) ────────────────────────────────────────

COLEGIOS = {
    "key": "colegios",
    "label": "Colegios",
    "search_terms": ["Colegio", "jardin", "kinder"],
}


class TestQueryMatchesNiche:
    """The observed failure: with niche "Colegios" the LLM refiner proposed
    fitness-gym queries and crawled spinningcentergym.com, because in Colombia a
    "Gimnasio Campestre" is a school. The prompt now names the tenant's terms,
    but this deterministic gate is what actually stops it."""

    def test_on_niche_query_passes(self):
        assert query_matches_niche("Colegio en Chía Colombia", COLEGIOS) is True

    def test_plural_and_accent_variants_pass(self):
        # The tenant writes "Colegio"; a good query says "colegios privados".
        assert query_matches_niche("colegios privados en Chía", COLEGIOS) is True
        assert query_matches_niche("jardines infantiles en Sopó", COLEGIOS) is True

    def test_the_real_drift_is_rejected(self):
        assert query_matches_niche("gimnasios en Cajicá", COLEGIOS) is False
        assert query_matches_niche(
            "centros deportivos y gimnasios Chía", COLEGIOS
        ) is False

    def test_related_but_different_industry_rejected(self):
        assert query_matches_niche("universidades en Chía", COLEGIOS) is False
        assert query_matches_niche("guarderías en Tabio", COLEGIOS) is False

    def test_multiword_term_requires_all_its_words(self):
        # "jardines infantiles" must not green-light botanical gardens.
        niche = {"key": "n", "search_terms": ["jardines infantiles"]}
        assert query_matches_niche("jardin infantil en Chía", niche) is True
        assert query_matches_niche("jardin botanico en Chía", niche) is False

    def test_no_terms_configured_is_permissive(self):
        # This is a drift filter, not an authorization check.
        assert query_matches_niche("lo que sea", {"key": "n"}) is True


class TestRefinementStaysOnNiche:
    def _provider(self, queries):
        import json as _json

        class FakeProvider:
            name = "openai"
            last_usage = SimpleNamespace(
                input_tokens=1, output_tokens=1,
                cached_input_tokens=0, reasoning_tokens=0,
            )

            async def stream_chat(self, model, messages):
                self.messages = messages
                yield _json.dumps(queries)

        return FakeProvider()

    async def _refine(self, queries):
        from src.agents.prospecting_nodes import refine_queries_node

        provider = self._provider(queries)
        with (
            patch("src.agents.prospecting_nodes.get_provider", return_value=provider),
            patch("src.agents.prospecting_nodes.resolve_model", return_value="gpt-4o"),
        ):
            out = await refine_queries_node(
                {"niche": COLEGIOS, "location": SABANA, "iteration": 0},
                {"configurable": {}},
            )
        return out, provider

    async def test_off_niche_proposals_are_dropped(self):
        out, _ = await self._refine([
            "directorio de colegios en Chía",
            "gimnasios en Cajicá",
            "spinning y crossfit Zipaquirá",
        ])
        assert out["queries"] == ["directorio de colegios en Chía"]
        assert out["metrics"]["off_niche_queries"] == 2

    async def test_tenant_terms_reach_the_prompt(self):
        _, provider = await self._refine(["colegios en Chía"])
        system = provider.messages[0]["content"]
        assert "Colegio" in system and "kinder" in system

    async def test_all_off_niche_yields_no_queries_but_still_advances(self):
        # Empty is the safe outcome: the iteration cap ends the loop instead of
        # spending Serper credits off-niche.
        out, _ = await self._refine(["gimnasios en Chía", "spa en Sopó"])
        assert out["queries"] == []
        assert out["iteration"] == 1


class TestMemoryStaysOnNiche:
    def test_off_niche_seed_is_not_planned(self):
        planned = _plan_queries(
            "2026-07-21",
            niche=COLEGIOS,
            location=SABANA,
            strategy={"best_queries": ["gimnasios en Cajicá", "colegios en Chía"]},
        )
        assert planned[0] == "colegios en Chía"
        assert "gimnasios en Cajicá" not in planned

    async def test_off_niche_query_is_not_promoted_to_best(self):
        from src.agents.prospecting_nodes import reflect_memory_node

        registry.set_store(None)
        out = await reflect_memory_node(
            {
                "tenant_id": "t1",
                "niche": COLEGIOS,
                "searched_queries": ["gimnasios en Cajicá", "colegios en Chía"],
                "candidates": [
                    {"query": "gimnasios en Cajicá", "domain": "gym.co"},
                    {"query": "colegios en Chía", "domain": "col.edu.co"},
                ],
            },
            {"configurable": {}},
        )
        best = out["strategy_memory"]["best_queries"]
        assert "colegios en Chía" in best
        # Yielded a prospect, but must not occupy a scarce best_queries slot.
        assert "gimnasios en Cajicá" not in best
