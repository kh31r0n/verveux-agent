"""CATALOG capability gating on the Python side (T-P1 / T-P2 / T-P5).

When the tenant disables catalog access, no node may put product data in front
of the model: faq_response swaps the catalog block for a degraded instruction,
and the catalog-collection nodes early-return a polite deflection without ever
calling the LLM. Absent flag (old checkpoints / older backend) behaves as today.
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import HumanMessage

from src.agents.faq_response import faq_response_node
from src.agents.capability_gate import CATALOG_DEGRADED_INSTRUCTION
from src.providers.base import UsageInfo


_SECRET_PRODUCT = "ProductoSecretoXYZ"


class _CapturingProvider:
    name = "fake"

    def __init__(self) -> None:
        self.captured_messages: list[dict] | None = None
        self.called = False
        self.last_usage = UsageInfo(input_tokens=10, output_tokens=5)

    def stream_chat(self, *, model: str, messages: list[dict]):
        self.called = True
        self.captured_messages = messages

        async def _gen():
            yield "ok"

        return _gen()


def _catalog() -> list[dict]:
    return [
        {"product_id": "p1", "name": _SECRET_PRODUCT, "description": "d", "price": 5, "stock": 3},
    ]


def _faq_state(*, catalog_access, with_catalog=True) -> dict:
    state: dict = {
        "messages": [HumanMessage(content="¿Qué productos tienen?")],
        "thread_id": "t:helena:1",
        "agent_type": "sales",
        "faqs": [{"id": "f1", "question": "¿Horario?", "answer": "9-5", "category": "", "priority": 0, "score": 0.9}],
        "user_context": {},
        "product_catalog": _catalog() if with_catalog else [],
    }
    if catalog_access is not None:
        state["catalog_access_enabled"] = catalog_access
    return state


async def _run_faq(state: dict) -> tuple[dict, _CapturingProvider]:
    provider = _CapturingProvider()
    events: list[dict] = []
    with (
        patch("src.agents.faq_response.get_stream_writer", return_value=(lambda e: events.append(e))),
        patch("src.agents.faq_response.get_provider", return_value=provider),
        patch("src.agents.faq_response.resolve_model", return_value="m"),
    ):
        result = await faq_response_node(state, {"configurable": {}})
    return result, provider


def _system_prompt(provider: _CapturingProvider) -> str:
    assert provider.captured_messages is not None
    return provider.captured_messages[0]["content"]


class TestFaqResponseGating:
    async def test_off_prompt_has_no_catalog_and_has_degraded_instruction(self):
        # T-P1: OFF -> zero catalog references, degraded instruction present.
        _, provider = await _run_faq(_faq_state(catalog_access=False))
        sys = _system_prompt(provider)
        assert _SECRET_PRODUCT not in sys
        assert "SIN ACCESO AL CATÁLOGO" in sys
        # FAQ knowledge survives — FAQs are out of scope for this toggle.
        assert "¿Horario?" in sys

    async def test_on_prompt_includes_catalog(self):
        _, provider = await _run_faq(_faq_state(catalog_access=True))
        sys = _system_prompt(provider)
        assert _SECRET_PRODUCT in sys
        assert "SIN ACCESO AL CATÁLOGO" not in sys

    async def test_absent_flag_behaves_as_on(self):
        # T-P5: missing flag => allowed (unchanged behavior for old backends).
        _, provider = await _run_faq(_faq_state(catalog_access=None))
        assert _SECRET_PRODUCT in _system_prompt(provider)


# ── Collection nodes early-return the degraded reply without calling the LLM ──

def _collect_state(node_agent_type: str, catalog_access: bool) -> dict:
    return {
        "messages": [HumanMessage(content="quiero comprar cosas")],
        "thread_id": "t:agent:1",
        "agent_type": node_agent_type,
        "language": "es",
        "product_catalog": [],
        "catalog_access_enabled": catalog_access,
        "user_context": {},
        "contact_id": "c1",
        "conversation_id": "conv1",
    }


async def _run_node(module_path: str, node_fn, state: dict):
    """Run a catalog-collection node with get_provider/resolve_model patched, and
    assert (via the returned provider) that the LLM was never reached when gated.
    """
    import contextlib

    provider = _CapturingProvider()
    events: list[dict] = []
    writer = lambda e: events.append(e)  # noqa: E731
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch(f"{module_path}.get_stream_writer", return_value=writer))
        # emit_degraded_catalog_reply grabs its own stream writer.
        stack.enter_context(
            patch("src.agents.capability_gate.get_stream_writer", return_value=writer)
        )
        stack.enter_context(patch(f"{module_path}.get_provider", return_value=provider))
        stack.enter_context(patch(f"{module_path}.resolve_model", return_value="m"))
        result = await node_fn(state, {"configurable": {}})
    return result, provider, events


class TestCollectionNodesGating:
    async def test_sales_collect_off_deflects_without_llm(self):
        from src.agents.sales_collect import sales_collect_node

        result, provider, events = await _run_node(
            "src.agents.sales_collect", sales_collect_node, _collect_state("sales", False)
        )
        assert provider.called is False
        reply = result["messages"][0].content
        assert "negocio" in reply.lower()
        assert result["turn_usage"] == []
        # The reply was streamed as a token event for NestJS to reconstruct.
        assert any(e.get("type") == "token" for e in events)

    async def test_menu_inquiry_off_deflects_without_llm(self):
        from src.agents.menu_inquiry import menu_inquiry_node

        result, provider, _ = await _run_node(
            "src.agents.menu_inquiry", menu_inquiry_node, _collect_state("restaurant", False)
        )
        assert provider.called is False
        assert "negocio" in result["messages"][0].content.lower()

    async def test_course_inquiry_off_deflects_without_llm(self):
        from src.agents.course_inquiry import course_inquiry_node

        result, provider, _ = await _run_node(
            "src.agents.course_inquiry", course_inquiry_node, _collect_state("school", False)
        )
        assert provider.called is False
        assert "negocio" in result["messages"][0].content.lower()

    async def test_restaurant_order_collect_off_deflects_without_llm(self):
        from src.agents.restaurant_order_collect import restaurant_order_collect_node

        result, provider, _ = await _run_node(
            "src.agents.restaurant_order_collect",
            restaurant_order_collect_node,
            _collect_state("restaurant", False),
        )
        assert provider.called is False
        assert "negocio" in result["messages"][0].content.lower()
