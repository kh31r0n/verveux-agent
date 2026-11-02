"""Caleb graph (MARKET_STUDY) — 17-stage parameterized B2B market study pipeline."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..caleb.budget import CalebBudgetTracker
from ..caleb.schema import CALEB_TIER_LIMITS, validate_study_config_v1


class CalebStudyState(TypedDict, total=False):
    tenant_id: str
    study_run_id: str
    config: dict[str, Any]
    internal_cost_cap_usd: float
    reserved_spend_usd: float
    settled_spend_usd: float
    degradations: list[dict[str, Any]]
    completed_stages: list[str]
    current_stage: str
    status: str
    awaiting_review_stage: str | None
    checkpoint_approved: bool
    planned_quotas: dict[str, Any]
    internal_baseline: dict[str, Any]
    candidate_entities: list[dict[str, Any]]
    scored_entities: list[dict[str, Any]]
    top_k_profiles: list[dict[str, Any]]
    zone_country_analysis: list[dict[str, Any]]
    quality_warnings: list[str]
    artifacts: list[dict[str, Any]]
    crm_sync_summary: dict[str, Any]
    failure_reason: str | None


def _record_stage(state: CalebStudyState, stage_key: str) -> list[str]:
    completed = list(state.get("completed_stages") or [])
    if stage_key not in completed:
        completed.append(stage_key)
    return completed


async def validate_and_freeze_config_node(state: CalebStudyState) -> dict[str, Any]:
    raw_cfg = state.get("config") or {}
    valid, model, errors = validate_study_config_v1(raw_cfg)
    if not valid or model is None:
        msg = "; ".join(f"{e['field']}: {e['message']}" for e in errors)
        return {
            "current_stage": "validate_and_freeze_config",
            "status": "FAILED",
            "failure_reason": f"Invalid StudyConfig v1: {msg}",
            "completed_stages": _record_stage(state, "validate_and_freeze_config"),
        }
    return {
        "config": model.model_dump(exclude_none=True),
        "current_stage": "validate_and_freeze_config",
        "status": "RUNNING",
        "completed_stages": _record_stage(state, "validate_and_freeze_config"),
    }


async def plan_stages_and_budget_node(state: CalebStudyState) -> dict[str, Any]:
    if state.get("status") == "FAILED":
        return {}
    cfg = state["config"]
    tier = cfg["tier"]
    quotas = dict(CALEB_TIER_LIMITS[tier])
    cap = float(state.get("internal_cost_cap_usd") or 11.7)
    return {
        "current_stage": "plan_stages_and_budget",
        "planned_quotas": quotas,
        "internal_cost_cap_usd": cap,
        "reserved_spend_usd": float(state.get("reserved_spend_usd") or 0.0),
        "settled_spend_usd": float(state.get("settled_spend_usd") or 0.0),
        "degradations": list(state.get("degradations") or []),
        "completed_stages": _record_stage(state, "plan_stages_and_budget"),
    }


async def ingest_internal_baseline_node(state: CalebStudyState) -> dict[str, Any]:
    if state.get("status") == "FAILED":
        return {}
    cfg = state["config"]
    modules = cfg.get("modules") or []
    baseline = state.get("internal_baseline") or {
        "enabled": "CURRENT_CUSTOMERS" in modules,
        "customersAnalyzed": 0,
        "tierACount": 0,
        "tierBCount": 0,
        "tierCCount": 0,
    }
    return {
        "current_stage": "ingest_internal_Baseline",
        "internal_baseline": baseline,
        "completed_stages": _record_stage(state, "ingest_internal_Baseline"),
    }


async def discover_candidates_node(state: CalebStudyState) -> dict[str, Any]:
    if state.get("status") == "FAILED":
        return {}
    tracker = CalebBudgetTracker(
        internal_cost_cap_usd=float(state.get("internal_cost_cap_usd") or 11.7),
        reserved_spend_usd=float(state.get("reserved_spend_usd") or 0.0),
        settled_spend_usd=float(state.get("settled_spend_usd") or 0.0),
        degradations=list(state.get("degradations") or []),
    )
    eval_res = tracker.reserve("discover_candidates:serper", 0.50)
    candidates = list(state.get("candidate_entities") or [])
    if eval_res.allowed and not candidates:
        candidates = [
            {
                "entityId": "cand-1",
                "name": "Suministros Industriales Andinos S.A.S.",
                "country": "CO",
                "city": "Bogotá",
                "zoneCode": "BOG_ZONA_INDUSTRIAL",
                "website": "https://suministrosandinos.example.com",
                "phone": "+57 601 5550101",
                "source": "SERPER_GOOGLE",
                "signals": ["importador directo", "catálogo industrial"],
            }
        ]
        tracker.settle("discover_candidates:serper", 0.35)

    return {
        "current_stage": "discover_candidates",
        "candidate_entities": candidates,
        "reserved_spend_usd": tracker.reserved_spend_usd,
        "settled_spend_usd": tracker.settled_spend_usd,
        "degradations": tracker.degradations,
        "completed_stages": _record_stage(state, "discover_candidates"),
    }


async def collect_public_registries_node(state: CalebStudyState) -> dict[str, Any]:
    if state.get("status") == "FAILED":
        return {}
    return {
        "current_stage": "collect_public_registries",
        "completed_stages": _record_stage(state, "collect_public_registries"),
    }


async def collect_procurement_activity_node(state: CalebStudyState) -> dict[str, Any]:
    if state.get("status") == "FAILED":
        return {}
    return {
        "current_stage": "collect_procurement_activity",
        "completed_stages": _record_stage(state, "collect_procurement_activity"),
    }


async def collect_trade_flows_node(state: CalebStudyState) -> dict[str, Any]:
    if state.get("status") == "FAILED":
        return {}
    return {
        "current_stage": "collect_trade_flows",
        "completed_stages": _record_stage(state, "collect_trade_flows"),
    }


async def inspect_websites_and_brands_node(state: CalebStudyState) -> dict[str, Any]:
    if state.get("status") == "FAILED":
        return {}
    return {
        "current_stage": "inspect_websites_and_brands",
        "completed_stages": _record_stage(state, "inspect_websites_and_brands"),
    }


async def classify_and_deduplicate_node(state: CalebStudyState) -> dict[str, Any]:
    if state.get("status") == "FAILED":
        return {}
    default_seg = state["config"]["taxonomy"]["defaultSegmentCode"]
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in state.get("candidate_entities") or []:
        key = (item.get("name") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        enriched = dict(item)
        enriched.setdefault("segmentCode", default_seg)
        deduped.append(enriched)
    return {
        "current_stage": "classify_and_deduplicate",
        "candidate_entities": deduped,
        "completed_stages": _record_stage(state, "classify_and_deduplicate"),
    }


async def score_entities_node(state: CalebStudyState) -> dict[str, Any]:
    if state.get("status") == "FAILED":
        return {}
    scored: list[dict[str, Any]] = []
    for item in state.get("candidate_entities") or []:
        has_web = bool(item.get("website") and item.get("website") != "sin dato")
        has_phone = bool(item.get("phone") and item.get("phone") != "sin dato")
        score = 78.0 if (has_web and has_phone) else 52.0
        band = "A" if score >= 70 else ("B" if score >= 45 else "C")
        entry = dict(item)
        entry["score"] = score
        entry["priorityBand"] = band
        scored.append(entry)
    scored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return {
        "current_stage": "score_entities",
        "scored_entities": scored,
        "completed_stages": _record_stage(state, "score_entities"),
    }


async def deep_research_top_k_node(state: CalebStudyState) -> dict[str, Any]:
    if state.get("status") == "FAILED":
        return {}
    quotas = state.get("planned_quotas") or {"maxTopKDeepResearch": 5}
    top_k = int(quotas.get("maxTopKDeepResearch", 5))
    scored = list(state.get("scored_entities") or [])
    profiles = [
        {
            "entityId": e.get("entityId"),
            "name": e.get("name"),
            "priorityBand": e.get("priorityBand"),
            "score": e.get("score"),
            "commercialSummary": "Cuenta B2B verificada con presencia activa en línea.",
            "verifiedSource": e.get("source", "sin dato"),
        }
        for e in scored[:top_k]
    ]
    return {
        "current_stage": "deep_research_top_k",
        "top_k_profiles": profiles,
        "completed_stages": _record_stage(state, "deep_research_top_k"),
    }


async def analyze_network_and_zones_node(state: CalebStudyState) -> dict[str, Any]:
    if state.get("status") == "FAILED":
        return {}
    cfg = state["config"]
    zones = (cfg.get("geography") or {}).get("localZones") or []
    countries = (cfg.get("geography") or {}).get("targetCountries") or []
    analysis: list[dict[str, Any]] = []
    for z in zones:
        analysis.append(
            {
                "code": z["code"],
                "name": z["name"],
                "kind": "ZONE",
                "score": 76.0,
                "decision": "GO",
            }
        )
    for c in countries:
        analysis.append(
            {
                "code": c["countryCode"],
                "name": c["countryName"],
                "kind": "COUNTRY",
                "score": 68.0,
                "decision": "EXPLORAR",
            }
        )
    return {
        "current_stage": "analyze_network_and_zones",
        "zone_country_analysis": analysis,
        "completed_stages": _record_stage(state, "analyze_network_and_zones"),
    }


async def quality_gate_node(state: CalebStudyState) -> dict[str, Any]:
    if state.get("status") == "FAILED":
        return {}
    warnings: list[str] = list(state.get("quality_warnings") or [])
    for item in state.get("scored_entities") or []:
        if not item.get("source"):
            warnings.append(f"Entidad sin fuente verificable: {item.get('name')}")
    return {
        "current_stage": "quality_gate",
        "quality_warnings": warnings,
        "completed_stages": _record_stage(state, "quality_gate"),
    }


def route_after_quality_gate(state: CalebStudyState) -> str:
    if state.get("status") == "FAILED":
        return "finalize_study_run"
    cfg = state.get("config") or {}
    checkpoints = (cfg.get("deliverables") or {}).get("humanCheckpoints") or {}
    needs_review = bool(
        checkpoints.get("afterDiscovery") or checkpoints.get("beforeCrmExport")
    )
    if needs_review and not state.get("checkpoint_approved"):
        return "human_checkpoint"
    return "generate_reports_and_exports"


async def human_checkpoint_node(state: CalebStudyState) -> dict[str, Any]:
    return {
        "current_stage": "human_checkpoint",
        "status": "AWAITING_REVIEW" if not state.get("checkpoint_approved") else "RUNNING",
        "awaiting_review_stage": (
            "human_checkpoint" if not state.get("checkpoint_approved") else None
        ),
        "completed_stages": _record_stage(state, "human_checkpoint"),
    }


async def generate_reports_and_exports_node(state: CalebStudyState) -> dict[str, Any]:
    if state.get("status") == "FAILED":
        return {}
    cfg = state["config"]
    deliv = cfg.get("deliverables") or {}
    artifacts: list[dict[str, Any]] = []
    if deliv.get("generateMarkdownReport", True):
        artifacts.append(
            {
                "kind": "MARKDOWN_REPORT",
                "filename": "informe_ejecutivo_caleb.md",
                "title": cfg.get("studyTitle", "Estudio de Mercado B2B"),
            }
        )
    if deliv.get("generateCsvExports", True):
        artifacts.append(
            {
                "kind": "CSV_ENTITIES",
                "filename": "entidades_calificadas.csv",
                "rowCount": len(state.get("scored_entities") or []),
            }
        )
    if deliv.get("generateExcelWorkbook", True):
        artifacts.append(
            {
                "kind": "EXCEL_WORKBOOK",
                "filename": "estudio_mercado_caleb.xlsx",
            }
        )
    return {
        "current_stage": "generate_reports_and_exports",
        "artifacts": artifacts,
        "completed_stages": _record_stage(state, "generate_reports_and_exports"),
    }


def route_after_reports(state: CalebStudyState) -> str:
    cfg = state.get("config") or {}
    crm_cfg = (cfg.get("deliverables") or {}).get("exportToOneCrm") or {}
    if crm_cfg.get("enabled"):
        return "sync_to_one_crm"
    return "finalize_study_run"


async def sync_to_one_crm_node(state: CalebStudyState) -> dict[str, Any]:
    scored = state.get("scored_entities") or []
    band_a = [e for e in scored if e.get("priorityBand") == "A"]
    return {
        "current_stage": "sync_to_one_crm",
        "crm_sync_summary": {
            "syncedCompanies": len(band_a),
            "syncedContacts": len(band_a),
        },
        "completed_stages": _record_stage(state, "sync_to_one_crm"),
    }


async def finalize_study_run_node(state: CalebStudyState) -> dict[str, Any]:
    current_status = state.get("status") or "RUNNING"
    degradations = state.get("degradations") or []
    if current_status == "FAILED":
        final_status = "FAILED"
    elif current_status == "AWAITING_REVIEW":
        final_status = "AWAITING_REVIEW"
    elif any(d.get("level") == "HARD_STOP_EXTERNAL_TOOLS" for d in degradations):
        final_status = "PARTIAL"
    else:
        final_status = "COMPLETED"

    return {
        "current_stage": "finalize_study_run",
        "status": final_status,
        "completed_stages": _record_stage(state, "finalize_study_run"),
    }


def build_caleb_graph(checkpointer: Any = None):
    graph = StateGraph(CalebStudyState)

    graph.add_node("validate_and_freeze_config", validate_and_freeze_config_node)
    graph.add_node("plan_stages_and_budget", plan_stages_and_budget_node)
    graph.add_node("ingest_internal_Baseline", ingest_internal_baseline_node)
    graph.add_node("discover_candidates", discover_candidates_node)
    graph.add_node("collect_public_registries", collect_public_registries_node)
    graph.add_node("collect_procurement_activity", collect_procurement_activity_node)
    graph.add_node("collect_trade_flows", collect_trade_flows_node)
    graph.add_node("inspect_websites_and_brands", inspect_websites_and_brands_node)
    graph.add_node("classify_and_deduplicate", classify_and_deduplicate_node)
    graph.add_node("score_entities", score_entities_node)
    graph.add_node("deep_research_top_k", deep_research_top_k_node)
    graph.add_node("analyze_network_and_zones", analyze_network_and_zones_node)
    graph.add_node("quality_gate", quality_gate_node)
    graph.add_node("human_checkpoint", human_checkpoint_node)
    graph.add_node("generate_reports_and_exports", generate_reports_and_exports_node)
    graph.add_node("sync_to_one_crm", sync_to_one_crm_node)
    graph.add_node("finalize_study_run", finalize_study_run_node)

    graph.add_edge(START, "validate_and_freeze_config")
    graph.add_edge("validate_and_freeze_config", "plan_stages_and_budget")
    graph.add_edge("plan_stages_and_budget", "ingest_internal_Baseline")
    graph.add_edge("ingest_internal_Baseline", "discover_candidates")
    graph.add_edge("discover_candidates", "collect_public_registries")
    graph.add_edge("collect_public_registries", "collect_procurement_activity")
    graph.add_edge("collect_procurement_activity", "collect_trade_flows")
    graph.add_edge("collect_trade_flows", "inspect_websites_and_brands")
    graph.add_edge("inspect_websites_and_brands", "classify_and_deduplicate")
    graph.add_edge("classify_and_deduplicate", "score_entities")
    graph.add_edge("score_entities", "deep_research_top_k")
    graph.add_edge("deep_research_top_k", "analyze_network_and_zones")
    graph.add_edge("analyze_network_and_zones", "quality_gate")

    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        [
            "human_checkpoint",
            "generate_reports_and_exports",
            "finalize_study_run",
        ],
    )
    graph.add_edge("human_checkpoint", "generate_reports_and_exports")

    graph.add_conditional_edges(
        "generate_reports_and_exports",
        route_after_reports,
        ["sync_to_one_crm", "finalize_study_run"],
    )
    graph.add_edge("sync_to_one_crm", "finalize_study_run")
    graph.add_edge("finalize_study_run", END)

    return graph.compile(checkpointer=checkpointer)
