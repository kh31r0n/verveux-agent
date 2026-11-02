"""Unit and contract tests for Caleb StudyConfig v1, budget tracker, and 17-node LangGraph."""

from __future__ import annotations

import copy

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.caleb.budget import CalebBudgetTracker
from src.caleb.schema import (
    CALEB_REGIONAL_COUNTRY_CODES,
    DEFAULT_LOCAL_MARKET_CO_CONFIG,
    compute_study_config_hash,
    migrate_study_config_to_v1,
    validate_study_config_v1,
)
from src.graphs.caleb_graph import build_caleb_graph
from src.graphs.registry import AGENT_TYPE_FALLBACK, CODE_NAME_REGISTRY


def test_default_local_market_config_validates_and_migrates_from_v0():
    valid, model, errors = validate_study_config_v1(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    assert valid is True
    assert errors == []
    assert model is not None
    assert model.schemaVersion == 1

    # v0 migration test
    v0 = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    del v0["schemaVersion"]
    del v0["safeguards"]
    migrated = migrate_study_config_to_v1(v0)
    assert migrated["schemaVersion"] == 1
    assert migrated["safeguards"]["allowUnverifiedContacts"] is False

    valid_v0, model_v0, errors_v0 = validate_study_config_v1(v0)
    assert valid_v0 is True
    assert errors_v0 == []
    assert model_v0 is not None


def test_regional_countries_include_all_19_countries():
    assert len(CALEB_REGIONAL_COUNTRY_CODES) == 19
    assert {"CO", "GT", "SV", "HN", "NI", "CR", "PA", "BZ"}.issubset(
        CALEB_REGIONAL_COUNTRY_CODES
    )
    assert {
        "EC",
        "PE",
        "CL",
        "AR",
        "BR",
        "BO",
        "PY",
        "UY",
        "VE",
        "GY",
        "SR",
    }.issubset(CALEB_REGIONAL_COUNTRY_CODES)


def test_locked_safeguards_cannot_be_disabled():
    tampered = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    tampered["safeguards"]["allowHallucinatedMetrics"] = True
    valid, model, errors = validate_study_config_v1(tampered)
    assert valid is False
    assert model is None
    assert any(e["code"] == "LOCKED_SAFEGUARD_VIOLATION" for e in errors)


def test_tier_limits_and_hash_determinism():
    cfg_a = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    cfg_b = dict(reversed(list(cfg_a.items())))
    assert compute_study_config_hash(cfg_a) == compute_study_config_hash(cfg_b)

    # Exceed PILOT zone limit (max 2)
    pilot_cfg = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    pilot_cfg["tier"] = "PILOT"
    pilot_cfg["geography"]["localZones"].append(
        {
            "code": "VAL_CALI",
            "name": "Cali — Yumbo",
            "country": "CO",
            "cities": ["Cali"],
        }
    )
    valid, _model, errors = validate_study_config_v1(pilot_cfg)
    assert valid is False
    assert any(
        e["field"] == "geography" and e["code"] == "TIER_LIMIT_EXCEEDED"
        for e in errors
    )


def test_budget_tracker_reservation_settlement_and_degradation():
    tracker = CalebBudgetTracker(internal_cost_cap_usd=10.0)

    # 1. Reserve $7.20 (72%) -> REDUCE_TOP_K_AND_QUERIES
    eval1 = tracker.reserve("op-1", 7.2)
    assert eval1.allowed is True
    assert eval1.level == "REDUCE_TOP_K_AND_QUERIES"

    # 2. Settle op-1 at $7.00 actual
    tracker.settle("op-1", 7.0)
    assert tracker.settled_spend_usd == 7.0
    assert tracker.reserved_spend_usd == 7.0

    # 3. Reserve $1.70 ($8.70 = 87%) -> DETERMINISTIC_SCORING_ONLY
    eval2 = tracker.reserve("op-2", 1.7)
    assert eval2.allowed is True
    assert eval2.level == "DETERMINISTIC_SCORING_ONLY"

    # 4. Attempt to reserve $1.00 ($9.70 = 97% >= 95%) -> HARD_STOP_EXTERNAL_TOOLS
    eval3 = tracker.reserve("op-3", 1.0)
    assert eval3.allowed is False
    assert eval3.level == "HARD_STOP_EXTERNAL_TOOLS"


@pytest.mark.asyncio
async def test_caleb_graph_compiles_and_executes_end_to_end():
    assert "caleb" in CODE_NAME_REGISTRY
    assert AGENT_TYPE_FALLBACK["market-study"] == "caleb"

    graph = build_caleb_graph(checkpointer=MemorySaver())
    cfg = copy.deepcopy(DEFAULT_LOCAL_MARKET_CO_CONFIG)
    cfg["deliverables"]["exportToOneCrm"]["enabled"] = True

    # 1) When human checkpoint is not yet approved -> ends in AWAITING_REVIEW
    res_review = await graph.ainvoke(
        {
            "tenant_id": "tenant-test-1",
            "study_run_id": "run-test-1",
            "config": cfg,
            "internal_cost_cap_usd": 44.7,
            "checkpoint_approved": False,
        },
        config={"configurable": {"thread_id": "caleb:tenant-test-1:run-test-1"}},
    )
    assert res_review["status"] == "AWAITING_REVIEW"
    assert res_review["awaiting_review_stage"] == "human_checkpoint"

    # 2) When checkpoint_approved=True -> completes all 17 stages and syncs to CRM
    res_done = await graph.ainvoke(
        {
            "tenant_id": "tenant-test-1",
            "study_run_id": "run-test-2",
            "config": cfg,
            "internal_cost_cap_usd": 44.7,
            "checkpoint_approved": True,
        },
        config={"configurable": {"thread_id": "caleb:tenant-test-1:run-test-2"}},
    )
    assert res_done["status"] == "COMPLETED"
    assert "finalize_study_run" in res_done["completed_stages"]
    assert "sync_to_one_crm" in res_done["completed_stages"]
    assert len(res_done["scored_entities"]) == 1
    assert res_done["scored_entities"][0]["priorityBand"] == "A"
    assert len(res_done["artifacts"]) == 3
