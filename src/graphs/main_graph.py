"""Backward-compatibility shim — delegates to sales_graph.

All new code should import from ``registry`` or ``sales_graph`` directly.
"""

from .sales_graph import build_sales_graph


def build_graph(checkpointer):
    """Legacy entrypoint — builds the SALES graph."""
    return build_sales_graph(checkpointer)
