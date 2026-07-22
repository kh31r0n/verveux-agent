"""Prospecting graph (aurora) — autonomous school/kindergarten discovery.

Not conversational: the backend scheduler triggers one run per opted-in tenant
per day via ``POST /prospecting/run``. The graph plans searches, queries the web,
fans out extraction per result (LangGraph ``Send`` map-reduce — the first use in
this codebase), deduplicates against the CRM, files ``AI_PROSPECTING`` contacts,
and reports terminal status + token usage back to the backend.

    START -> plan_searches -> web_search
          -> [Send fan-out] extract_and_enrich
          -> dedupe_check -> create_contacts -> report -> END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..agents.prospecting_nodes import (
    ProspectingState,
    create_contacts_node,
    dedupe_check_node,
    extract_and_enrich_node,
    fan_out_to_extract,
    plan_searches_node,
    report_node,
    web_search_node,
)


def build_prospecting_graph(checkpointer):
    graph = StateGraph(ProspectingState)

    graph.add_node("plan_searches", plan_searches_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("extract_and_enrich", extract_and_enrich_node)
    graph.add_node("dedupe_check", dedupe_check_node)
    graph.add_node("create_contacts", create_contacts_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "plan_searches")
    graph.add_edge("plan_searches", "web_search")

    # Map: one extract_and_enrich per search result. When there are no results
    # the fan-out returns [] and LangGraph proceeds straight to dedupe_check.
    graph.add_conditional_edges(
        "web_search",
        fan_out_to_extract,
        ["extract_and_enrich", "dedupe_check"],
    )

    # Reduce: all parallel extract branches converge on dedupe_check.
    graph.add_edge("extract_and_enrich", "dedupe_check")
    graph.add_edge("dedupe_check", "create_contacts")
    graph.add_edge("create_contacts", "report")
    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer)
