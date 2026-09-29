"""Unit tests for restaurant_order_collect — extraction parsing (Gemini fence
regression), catalog resolution into the shared cart, completion logic, and
the post-checkout fresh-start reset.

The provider is faked: call 1 is the extraction pass (canned payload,
optionally markdown-fenced like Gemini emits), call 2 the conversational
reply. Backend sync is AsyncMock-patched.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from langchain_core.messages import HumanMessage

from src.agents.restaurant_order_collect import restaurant_order_collect_node
from src.providers.base import UsageInfo

_CATALOG = [
    {"product_id": "p-tacos", "name": "Tacos al pastor", "price": 12.5, "stock": 10},
    {"product_id": "p-agua", "name": "Agua de horchata", "price": 3.0, "stock": 10},
    {"product_id": "p-flan", "name": "Flan casero", "price": 5.0, "stock": 10},
]

_CONFIG = {"configurable": {"thread_id": "t:1", "openai_api_key": "sk-test"}}


class _FakeProvider:
    name = "fake"

    def __init__(self, extraction: str, reply: str = "¡Listo!"):
        self.extraction = extraction
        self.reply = reply
        self._calls = 0
        self.last_usage = UsageInfo(input_tokens=10, output_tokens=5)

    def stream_chat(self, *, model: str, messages: list[dict]):
        self._calls += 1
        payload = self.extraction if self._calls == 1 else self.reply

        async def _gen():
            yield payload

        return _gen()


def _state(message: str = "quiero 2 tacos al pastor", **overrides) -> dict:
    state = {
        "messages": [HumanMessage(content=message)],
        "thread_id": "t:1",
        "conversation_id": "conv-1",
        "contact_id": "contact-1",
        "product_catalog": _CATALOG,
        "user_context": {"name": "Ana"},
        "language": "es",
        "agent_type": "restaurant",
        "agent_code_name": "giulia",
    }
    state.update(overrides)
    return state


async def _run(state: dict, extraction: str, sync_ok: bool = True) -> tuple[dict, AsyncMock]:
    provider = _FakeProvider(extraction)
    sync_mock = AsyncMock(return_value=sync_ok)
    with (
        patch("src.agents.restaurant_order_collect.get_provider", return_value=provider),
        patch("src.agents.restaurant_order_collect.resolve_model", return_value="test-model"),
        patch("src.agents.restaurant_order_collect.get_stream_writer", return_value=lambda *_: None),
        patch("src.agents.restaurant_order_collect.sync_full_cart_to_backend", new=sync_mock),
    ):
        result = await restaurant_order_collect_node(state, _CONFIG)
    return result, sync_mock


class TestFencedExtraction:
    async def test_gemini_fenced_json_is_applied(self):
        """Regression for the production bug: Gemini wraps the extraction JSON
        in ```json fences; before the fix json.loads failed silently and the
        order never progressed."""
        extraction = (
            '```json\n'
            + json.dumps({
                "items": [{"name": "Tacos al pastor", "quantity": 2, "operation": "add", "notes": ""}],
                "service_type": "pickup",
            })
            + "\n```"
        )
        result, sync_mock = await _run(_state(), extraction)

        assert result["restaurant_order_data"]["service_type"] == "pickup"
        assert len(result["cart"]) == 1
        assert result["cart"][0]["product_id"] == "p-tacos"
        assert result["cart"][0]["qty"] == 2
        assert result["restaurant_order_complete"] is True
        sync_mock.assert_awaited_once()

    async def test_garbage_extraction_does_not_crash(self):
        result, sync_mock = await _run(_state(), "no soy json")

        assert result["cart"] == []
        assert result["restaurant_order_data"] == {}
        assert result["restaurant_order_complete"] is False
        sync_mock.assert_not_awaited()


class TestCartOperations:
    async def test_resolved_item_carries_catalog_price(self):
        extraction = json.dumps({
            "items": [{"name": "agua de horchata", "quantity": 1, "operation": "add", "notes": "sin hielo"}],
        })
        result, _ = await _run(_state(), extraction)

        item = result["cart"][0]
        assert item["product_id"] == "p-agua"
        assert item["price"] == 3.0
        assert item["notes"] == "sin hielo"

    async def test_remove_and_update_quantity_operations(self):
        extraction = json.dumps({
            "items": [
                {"name": "Tacos al pastor", "quantity": 5, "operation": "update_quantity"},
                {"name": "Flan casero", "quantity": 1, "operation": "remove"},
            ],
        })
        pre_cart = [
            {"product_id": "p-tacos", "name": "Tacos al pastor", "qty": 2, "price": 12.5, "notes": ""},
            {"product_id": "p-flan", "name": "Flan casero", "qty": 1, "price": 5.0, "notes": ""},
        ]
        result, _ = await _run(_state(cart=pre_cart), extraction)

        by_id = {i["product_id"]: i for i in result["cart"]}
        assert by_id["p-tacos"]["qty"] == 5
        assert "p-flan" not in by_id

    async def test_unresolved_dish_blocks_completion(self):
        extraction = json.dumps({
            "items": [{"name": "sushi volcánico", "quantity": 1, "operation": "add"}],
            "service_type": "pickup",
        })
        result, sync_mock = await _run(_state(), extraction)

        assert result["cart"] == []
        assert result["pending_unknown_items"], "unresolved dish should surface"
        assert result["pending_unknown_items"][0]["name"] == "sushi volcánico"
        assert result["restaurant_order_complete"] is False
        sync_mock.assert_not_awaited()


class TestCompletion:
    async def test_delivery_without_address_is_incomplete(self):
        extraction = json.dumps({
            "items": [{"name": "Tacos al pastor", "quantity": 1, "operation": "add"}],
            "service_type": "delivery",
        })
        result, _ = await _run(_state(), extraction)
        assert result["restaurant_order_complete"] is False

    async def test_delivery_with_address_is_complete(self):
        extraction = json.dumps({
            "items": [{"name": "Tacos al pastor", "quantity": 1, "operation": "add"}],
            "service_type": "delivery",
            "delivery_address": "Cra 7 # 12-34",
        })
        result, _ = await _run(_state(), extraction)
        assert result["restaurant_order_complete"] is True
        assert result["restaurant_order_data"]["delivery_address"] == "Cra 7 # 12-34"

    async def test_service_type_spanish_phrases_normalized(self):
        extraction = json.dumps({
            "items": [{"name": "Tacos al pastor", "quantity": 1, "operation": "add"}],
            "service_type": "a domicilio",
        })
        result, _ = await _run(_state(), extraction)
        assert result["restaurant_order_data"]["service_type"] == "delivery"

    async def test_phase_always_collect(self):
        result, _ = await _run(_state(), "{}")
        assert result["restaurant_phase"] == "collect"


class TestFreshStart:
    async def test_post_checkout_state_is_reset(self):
        """After a completed checkout (execute_confirmed), a new order intent
        starts with an empty cart and clears the lifecycle flags."""
        extraction = json.dumps({
            "items": [{"name": "Flan casero", "quantity": 1, "operation": "add"}],
        })
        stale_cart = [
            {"product_id": "p-tacos", "name": "Tacos al pastor", "qty": 2, "price": 12.5, "notes": ""},
        ]
        result, _ = await _run(
            _state(
                cart=stale_cart,
                restaurant_order_data={"service_type": "pickup"},
                execute_confirmed=True,
                restaurant_order_confirmed=True,
            ),
            extraction,
        )

        assert [i["product_id"] for i in result["cart"]] == ["p-flan"]
        assert result["restaurant_order_data"] == {}
        assert result["execute_confirmed"] is False
        assert result["restaurant_order_confirmed"] is False
