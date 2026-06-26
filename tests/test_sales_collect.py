"""Unit tests for the helper(s) in src.agents.sales_collect.

Covers the ``_compute_mentioned_product_ids`` signal that NestJS reads off
the SSE ``done`` event to decide whether to attach a product image to the
outbound WhatsApp reply.

UX contract:
  * Exactly one distinct catalog product referenced → ``[product_id]``
  * Zero or >1 distinct products → ``[]`` (text-only fallback)
"""

import pytest

from src.agents.sales_collect import _compute_mentioned_product_ids, _format_money


CATALOG = [
    {"product_id": "P1", "name": "Arroz integral", "description": "1kg", "price": 4.50, "stock": 12},
    {"product_id": "P2", "name": "Pollo entero",   "description": "fresco", "price": 9.99,  "stock": 4},
    {"product_id": "P3", "name": "Aceite de oliva","description": "500ml",  "price": 7.20,  "stock": 8},
]


def _item(pid: str) -> dict:
    """Mimics ProductResolver.resolve_many output shape."""
    return {"product_id": pid, "name": pid, "qty": 1, "operation": "add"}


class TestComputeMentionedProductIds:
    def test_empty_inputs_returns_empty(self):
        assert _compute_mentioned_product_ids([], [], CATALOG) == []

    def test_single_cart_item_returns_pid(self):
        assert _compute_mentioned_product_ids([_item("P1")], [], CATALOG) == ["P1"]

    def test_single_reference_resolves_via_name(self):
        # Pure info query: user asked about a product, didn't add to cart.
        assert _compute_mentioned_product_ids([], ["Arroz integral"], CATALOG) == ["P1"]

    def test_single_reference_fuzzy_match(self):
        # Substring/fuzzy match still works because we use the heuristic resolver.
        result = _compute_mentioned_product_ids([], ["arroz"], CATALOG)
        assert result == ["P1"]

    def test_cart_item_and_matching_reference_deduplicates(self):
        # Same product referenced twice (once as cart op, once by name).
        result = _compute_mentioned_product_ids(
            [_item("P1")], ["Arroz integral"], CATALOG
        )
        assert result == ["P1"]

    def test_multi_product_returns_empty(self):
        # User asked about two distinct products — skip image, fall through to text.
        result = _compute_mentioned_product_ids(
            [_item("P1")], ["Pollo entero"], CATALOG
        )
        assert result == []

    def test_two_cart_items_returns_empty(self):
        result = _compute_mentioned_product_ids(
            [_item("P1"), _item("P2")], [], CATALOG
        )
        assert result == []

    def test_two_references_returns_empty(self):
        result = _compute_mentioned_product_ids(
            [], ["Arroz integral", "Pollo entero"], CATALOG
        )
        assert result == []

    def test_unresolvable_reference_is_skipped(self):
        # Garbage that the resolver can't match (no catalog hit) is ignored —
        # so a single resolved cart op still wins.
        result = _compute_mentioned_product_ids(
            [_item("P1")], ["xyz nonsense"], CATALOG
        )
        assert result == ["P1"]

    def test_only_unresolvable_references_returns_empty(self):
        result = _compute_mentioned_product_ids([], ["xyz nonsense"], CATALOG)
        assert result == []

    def test_empty_and_whitespace_names_are_skipped(self):
        result = _compute_mentioned_product_ids(
            [], ["", "   ", "Arroz integral"], CATALOG
        )
        assert result == ["P1"]

    def test_non_string_names_are_skipped(self):
        # Defensive: the LLM might emit a number or None inside the JSON list.
        result = _compute_mentioned_product_ids(
            [], [None, 42, "Arroz integral"], CATALOG  # type: ignore[list-item]
        )
        assert result == ["P1"]

    def test_empty_catalog_returns_empty(self):
        # No catalog → resolver can't match anything, but cart-op pids still pass through.
        assert _compute_mentioned_product_ids([_item("P1")], [], []) == ["P1"]
        assert _compute_mentioned_product_ids([], ["Arroz"], []) == []

    def test_cart_item_without_product_id_is_skipped(self):
        # Defensive: a malformed item dict shouldn't poison the set.
        result = _compute_mentioned_product_ids(
            [{"name": "broken"}, _item("P2")], [], CATALOG
        )
        assert result == ["P2"]


class TestFormatMoney:
    """Catalog price formatter — defensive against odd shapes from upstream."""

    def test_int(self):
        assert _format_money(5) == "5.00"

    def test_float(self):
        assert _format_money(4.5) == "4.50"

    def test_numeric_string(self):
        assert _format_money("4.50") == "4.50"

    def test_zero_default(self):
        assert _format_money(0) == "0.00"

    def test_decimal_shaped_dict_falls_back(self):
        # decimal.js Decimal instances serialized via Object.entries arrive
        # as {s, e, d} dicts. Should not crash the format string.
        assert _format_money({"s": 1, "e": 0, "d": [4.5]}) == "0.00"

    def test_none_falls_back(self):
        assert _format_money(None) == "0.00"

    def test_garbage_string_falls_back(self):
        assert _format_money("not a number") == "0.00"
