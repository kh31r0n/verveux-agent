"""Prospecting graph (aurora) — autonomous niche discovery with self-improvement.

Not conversational: the backend scheduler triggers one run per opted-in tenant
per day via ``POST /prospecting/run``. The graph loads cross-run strategy memory
and human feedback, plans niche+location searches, queries the web (Serper
``/search`` + ``/places``), fans out extraction per result (LangGraph ``Send``
map-reduce), deduplicates against the CRM, evaluates quality and — when a run
falls short — refines its own queries and searches again (bounded conditional
loop). It then files ``AI_PROSPECTING`` contacts, updates strategy memory, and
reports terminal status + token usage back to the backend.

    START -> load_memory -> plan_searches -> web_search
          -> [Send fan-out] extract_and_enrich -> dedupe_check -> evaluate_quality
          -> (route) refine_queries -> web_search        (loop, iteration-capped)
          -> (route) create_contacts -> reflect_memory -> report -> END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..agents.prospecting_nodes import (
    ProspectingState,
    create_contacts_node,
    dedupe_check_node,
    evaluate_quality_node,
    extract_and_enrich_node,
    fan_out_to_extract,
    load_memory_node,
    plan_searches_node,
    reflect_memory_node,
    refine_queries_node,
    report_node,
    route_from_evaluate,
    web_search_node,
)


def build_prospecting_graph(checkpointer):
    graph = StateGraph(ProspectingState)

    graph.add_node("load_memory", load_memory_node)
    graph.add_node("plan_searches", plan_searches_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("extract_and_enrich", extract_and_enrich_node)
    graph.add_node("dedupe_check", dedupe_check_node)
    graph.add_node("evaluate_quality", evaluate_quality_node)
    graph.add_node("refine_queries", refine_queries_node)
    graph.add_node("create_contacts", create_contacts_node)
    graph.add_node("reflect_memory", reflect_memory_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "plan_searches")
    graph.add_edge("plan_searches", "web_search")

    # Map: one extract_and_enrich per search result. When there are no results
    # the fan-out returns ["dedupe_check"] and LangGraph proceeds straight there.
    graph.add_conditional_edges(
        "web_search",
        fan_out_to_extract,
        ["extract_and_enrich", "dedupe_check"],
    )

    # Reduce: all parallel extract branches converge on dedupe_check.
    graph.add_edge("extract_and_enrich", "dedupe_check")
    graph.add_edge("dedupe_check", "evaluate_quality")

    # Refinement loop: not enough quality prospects (and iterations left) →
    # refine_queries → web_search; otherwise proceed to create_contacts.
    graph.add_conditional_edges(
        "evaluate_quality",
        route_from_evaluate,
        ["refine_queries", "create_contacts"],
    )
    graph.add_edge("refine_queries", "web_search")

    graph.add_edge("create_contacts", "reflect_memory")
    graph.add_edge("reflect_memory", "report")
    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer)
