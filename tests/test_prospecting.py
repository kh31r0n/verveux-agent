"""Tests for the prospecting graph (aurora) — dedupe normalization, the Send
fan-out, and the create/report contract.

MemorySaver + mocked backend/search/LLM calls, driving astream() and inspecting
the node-update sequence (mirrors test_leads_graph.py).
"""

from unittest.mock import AsyncMock, patch

from langgraph.checkpoint.memory import MemorySaver

from src.agents.dedup import (
    normalize_domain,
    normalize_name,
    prospect_external_id,
)
from src.agents.prospecting_nodes import _plan_queries
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


class TestPlanQueries:
    def test_rotation_differs_by_run_date(self):
        q1 = _plan_queries("2026-07-21")
        q2 = _plan_queries("2026-07-22")
        assert q1 and q2
        # Different dates should not start on the same city query.
        assert q1[0] != q2[0]

    def test_respects_search_budget(self):
        from src.config import settings

        queries = _plan_queries("2026-07-21")
        assert len(queries) <= settings.prospecting_max_searches


# ── Graph flow ──────────────────────────────────────────────────────────────


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id, "openai_api_key": "sk-test"}}


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


class TestGraphFlow:
    async def test_send_fanout_one_extract_per_result(self):
        graph = build_prospecting_graph(MemorySaver())

        results = [
            {"title": "A", "url": "https://a.edu.co", "snippet": ""},
            {"title": "B", "url": "https://b.edu.co", "snippet": ""},
            {"title": "C", "url": "https://c.edu.co", "snippet": ""},
        ]

        async def fake_search(query: str):
            # First query returns the three hits; the rest none (dedup by url).
            return results if query.endswith("Colombia") and "colegios" in query else []

        extract_calls = {"n": 0}

        async def fake_extract(state, *args):
            extract_calls["n"] += 1
            return {}

        with (
            patch(
                "src.agents.prospecting_nodes._serper_search",
                AsyncMock(side_effect=fake_search),
            ),
            patch(
                "src.graphs.prospecting_graph.extract_and_enrich_node",
                AsyncMock(side_effect=fake_extract),
            ),
            patch(
                "src.agents.prospecting_nodes.backend_client.check_prospect_duplicates",
                AsyncMock(return_value={}),
            ),
            patch(
                "src.agents.prospecting_nodes.backend_client.report_prospecting_run",
                AsyncMock(return_value={"ok": True}),
            ),
        ):
            # Rebuild so the patched node is wired into the graph.
            graph = build_prospecting_graph(MemorySaver())
            chunks = await _run(
                graph,
                {"tenant_id": "t1", "run_id": "r1", "run_date": "2026-07-21"},
                "prospecting:t1:2026-07-21",
            )

        names = _node_names(chunks)
        # extract_and_enrich fired once per unique search result.
        assert names.count("extract_and_enrich") == len(results)
        assert "dedupe_check" in names
        assert "report" in names

    async def test_zero_results_still_reports(self):
        report_mock = AsyncMock(return_value={"ok": True})
        with (
            patch(
                "src.agents.prospecting_nodes._serper_search",
                AsyncMock(return_value=[]),
            ),
            patch(
                "src.agents.prospecting_nodes.backend_client.check_prospect_duplicates",
                AsyncMock(return_value={}),
            ),
            patch(
                "src.agents.prospecting_nodes.backend_client.report_prospecting_run",
                report_mock,
            ),
        ):
            graph = build_prospecting_graph(MemorySaver())
            chunks = await _run(
                graph,
                {"tenant_id": "t1", "run_id": "r0", "run_date": "2026-07-21"},
                "prospecting:t1:zero",
            )

        names = _node_names(chunks)
        assert "dedupe_check" in names
        assert "report" in names
        report_mock.assert_awaited_once()
        assert report_mock.await_args.args[1] == "COMPLETED"

    async def test_survivors_created_and_duplicates_skipped(self):
        # Two candidates; one already exists in the CRM.
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
        }
        cand_b = {**cand_a, "externalId": prospect_external_id("Colegio B"), "customName": "Colegio B"}

        async def fake_extract(state, *args):
            # Each Send delivers a distinct result url; alternate the candidate.
            url = (state.get("result") or {}).get("url", "")
            return {"candidates": [cand_a if "a" in url else cand_b]}

        create_mock = AsyncMock(return_value={"ok": True, "contactId": "c1", "deduped": False})
        dedup_mock = AsyncMock(
            return_value={cand_a["externalId"]: {"exists": True, "reason": "identity"}}
        )

        with (
            patch(
                "src.agents.prospecting_nodes._serper_search",
                AsyncMock(
                    return_value=[
                        {"title": "A", "url": "https://a.edu.co", "snippet": ""},
                        {"title": "B", "url": "https://b.edu.co", "snippet": ""},
                    ]
                ),
            ),
            patch(
                "src.graphs.prospecting_graph.extract_and_enrich_node",
                AsyncMock(side_effect=fake_extract),
            ),
            patch(
                "src.agents.prospecting_nodes.backend_client.check_prospect_duplicates",
                dedup_mock,
            ),
            patch(
                "src.agents.prospecting_nodes.backend_client.create_prospect_contact",
                create_mock,
            ),
            patch(
                "src.agents.prospecting_nodes.backend_client.report_prospecting_run",
                AsyncMock(return_value={"ok": True}),
            ),
        ):
            graph = build_prospecting_graph(MemorySaver())
            await _run(
                graph,
                {"tenant_id": "t1", "run_id": "r2", "run_date": "2026-07-21"},
                "prospecting:t1:create",
            )

        # Only candidate B survived dedupe → exactly one create call.
        create_mock.assert_awaited_once()
        created_body = create_mock.await_args.args[1]
        assert created_body["externalId"] == cand_b["externalId"]
