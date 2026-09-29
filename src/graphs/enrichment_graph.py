"""Enrichment graph (sherlock) — one-time contact website enrichment.

Not conversational: the backend scheduler claims ONE eligible contact per sweep
tick and triggers a run via ``POST /enrichment/run``. The graph loads cross-run
strategy memory and human feedback, fetches the contact's own website through the
SSRF-hardened fetcher (landing page + a bounded set of same-site contact/about
pages), extracts evidence-backed phone candidates plus a description of the
business, writes a short sales-approach recommendation, updates strategy memory,
and reports terminal status + token usage back to the backend — which validates
the phone as E.164 and merges the result onto the contact.

    START -> load_memory -> (route) discover_website ?-> fetch_pages -> extract
          -> (route) refine_paths -> fetch_pages     (loop, iteration-capped)
          -> (route) strategy -> reflect_memory -> report -> END

`discover_website` runs only for a contact that arrived without a website (an
aurora prospect whose extractor found no domain): it searches for the business by
name and must both match it deterministically and have an LLM confirm it before
the crawl is allowed to start. Nothing confirmed means the run reports without
fetching anything.

The refinement loop is the "no phone found yet" path: an LLM proposes further
same-site paths to try and the graph fetches again, bounded by
``settings.sherlock_max_iterations``.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..agents.enrichment_nodes import (
    EnrichmentState,
    discover_website_node,
    extract_node,
    fetch_pages_node,
    load_memory_node,
    refine_paths_node,
    reflect_memory_node,
    report_node,
    route_from_discovery,
    route_from_extract,
    route_from_memory,
    strategy_node,
)


def build_enrichment_graph(checkpointer):
    graph = StateGraph(EnrichmentState)

    graph.add_node("load_memory", load_memory_node)
    graph.add_node("discover_website", discover_website_node)
    graph.add_node("fetch_pages", fetch_pages_node)
    graph.add_node("extract", extract_node)
    graph.add_node("refine_paths", refine_paths_node)
    graph.add_node("strategy", strategy_node)
    graph.add_node("reflect_memory", reflect_memory_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "load_memory")

    # A contact with a website on file goes straight to the crawl; one without
    # has to have a site discovered and confirmed first, and stops at `report`
    # when none can be.
    graph.add_conditional_edges(
        "load_memory",
        route_from_memory,
        ["discover_website", "fetch_pages"],
    )
    graph.add_conditional_edges(
        "discover_website",
        route_from_discovery,
        ["fetch_pages", "report"],
    )
    graph.add_edge("fetch_pages", "extract")

    # Refinement loop: nothing phone-shaped found (and iterations left) →
    # refine_paths → fetch_pages; otherwise proceed to the strategy write-up.
    graph.add_conditional_edges(
        "extract",
        route_from_extract,
        ["refine_paths", "strategy"],
    )
    graph.add_edge("refine_paths", "fetch_pages")

    graph.add_edge("strategy", "reflect_memory")
    graph.add_edge("reflect_memory", "report")
    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer)
