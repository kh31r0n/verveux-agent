"""Tests for the greeting_response node and its routing.

The node renders the greeting with one LLM call using the tenant's
{AGENT_TYPE}_GREETING prompt, falling back to the original deterministic
templates when the LLM call fails.

Part A — direct node unit tests on the FALLBACK path (templates, persona,
         language) — the fake provider raises, so template output is asserted.
Part B — routing matrix: greeting + known name reaches greeting_response
         (not faq_response) on every agent graph.
Part C — gates, precedence, and negatives (no-name paths unchanged).
Part D — LLM path: prompt resolution, placeholder rendering, usage records.

All graph tests follow the project convention: patch nodes at the graph
module namespace BEFORE building with MemorySaver, stream with
stream_mode="updates", and assert on update keys. The greeting node itself
always runs for real, with `get_provider` patched at its module namespace.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agents.greeting_response import greeting_response_node, has_contact_name
from src.providers.base import UsageInfo
from src.graphs.appointments_graph import build_appointments_graph
from src.graphs.camila_graph import build_camila_graph
from src.graphs.restaurant_graph import build_restaurant_graph
from src.graphs.sales_graph import build_sales_graph
from src.graphs.school_graph import build_school_graph
from src.schemas.intent import IntentType, StructuredIntent


def _node_names(results):
    return [list(r.keys())[0] for r in results if isinstance(r, dict)]


def _base_state(message: str, code_name: str, agent_type: str, **overrides):
    state = {
        "messages": [HumanMessage(content=message)],
        "thread_id": f"test:{code_name}:1",
        "tenant_id": "tenant-1",
        "conversation_id": "conv-1",
        "contact_id": "contact-1",
        "user_context": {"name": "Ana"},  # default: name known
        "attachments": [],
        "faqs": [],
        "agent_code_name": code_name,
        "agent_type": agent_type,
    }
    state.update(overrides)
    return state


def _config(code_name: str):
    return {
        "configurable": {
            "thread_id": f"test:{code_name}:1",
            "openai_api_key": "sk-test",
            "llm_provider": "openai",
            "llm_model": "gpt-test",
        }
    }


def _writer_recorder():
    """Recorder for get_stream_writer — direct node calls run outside a
    runnable context, where the real get_stream_writer() raises."""
    events: list[dict] = []
    return events, (lambda evt: events.append(evt))


class _FakeProvider:
    """Provider double for the greeting LLM call.

    `fail=True` raises on stream (exercises the deterministic template
    fallback); otherwise streams `chunks` and records the messages payload
    of every call in `self.calls`.
    """

    name = "fake"

    def __init__(self, chunks: tuple[str, ...] = ("¡Hola!",), fail: bool = False):
        self.chunks = chunks
        self.fail = fail
        self.calls: list[list[dict]] = []
        self.last_usage = UsageInfo(input_tokens=10, output_tokens=5)

    def stream_chat(self, *, model: str, messages: list[dict]):
        self.calls.append(messages)

        async def _gen():
            if self.fail:
                raise RuntimeError("provider unavailable")
            for chunk in self.chunks:
                yield chunk

        return _gen()


async def _call_node(
    state: dict,
    config: dict | None = None,
    provider: _FakeProvider | None = None,
) -> tuple[dict, list[dict]]:
    # Default: failing provider → the node renders the deterministic template.
    provider = provider or _FakeProvider(fail=True)
    events, write = _writer_recorder()
    with (
        patch(
            "src.agents.greeting_response.get_stream_writer", return_value=write
        ),
        patch(
            "src.agents.greeting_response.get_provider", return_value=provider
        ),
        patch(
            "src.agents.greeting_response.resolve_model", return_value="model-test"
        ),
    ):
        result = await greeting_response_node(state, config or {"configurable": {}})
    return result, events


# ── Part A — direct node unit tests ──────────────────────────────────────────


class TestGreetingResponseNode:
    async def test_camila_spanish_greeting(self):
        state = _base_state(
            "Hola", "camila", "school", user_context={"name": "Patricia Hernández"}
        )
        result, events = await _call_node(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        text = result["messages"][0].content
        assert "Camila" in text
        assert "secretaria académica" in text
        assert "Patricia Hernández" in text
        # No cross-domain leakage from the store-flavoured FAQ prompt
        assert "tienda" not in text.lower()
        # Failed LLM call → nothing contributed to turn_usage
        assert result["turn_usage"] == []
        # The backend builds the WhatsApp reply exclusively from token events
        assert events == [{"type": "token", "content": text}]

    @pytest.mark.parametrize(
        ("code_name", "agent_type", "persona", "role_fragment"),
        [
            ("camila", "school", "Camila", "secretaria académica"),
            ("sofia", "school", "Sofía", "asistente académica"),
            ("helena", "sales", "Helena", "asesora de ventas"),
            ("giulia", "restaurant", "Giulia", "asistente del restaurante"),
            ("marco", "appointments", "Marco", "asistente de citas"),
        ],
    )
    async def test_per_agent_persona_and_role(
        self, code_name, agent_type, persona, role_fragment
    ):
        state = _base_state("Hola", code_name, agent_type)
        result, _ = await _call_node(state)
        text = result["messages"][0].content
        assert persona in text
        assert role_fragment in text

    async def test_english_template(self):
        state = _base_state("Hi", "helena", "sales", language="en")
        result, _ = await _call_node(state)
        text = result["messages"][0].content
        assert text.startswith("Hi, Ana!")
        assert "your sales assistant" in text

    async def test_portuguese_template(self):
        state = _base_state("Olá", "giulia", "restaurant", language="pt")
        result, _ = await _call_node(state)
        text = result["messages"][0].content
        assert text.startswith("Olá, Ana!")
        assert "sua assistente do restaurante" in text

    @pytest.mark.parametrize("language", ["fr", "", None, "ES", "es-CO"])
    async def test_unknown_or_variant_language_falls_back_to_spanish(self, language):
        state = _base_state("Hola", "camila", "school", language=language)
        result, _ = await _call_node(state)
        assert "¿En qué puedo ayudarte hoy?" in result["messages"][0].content

    async def test_channel_persona_overrides_default(self):
        state = _base_state(
            "Hola", "camila", "school", agent_persona_name="Doña Rosa"
        )
        result, _ = await _call_node(state)
        text = result["messages"][0].content
        assert "Doña Rosa" in text
        assert "Camila" not in text

    async def test_latch_without_name_renders_cleanly(self):
        # camila can route here with only the school_name_captured latch set
        # (backend persist failed) — the template must not dangle a comma.
        state = _base_state(
            "Hola", "camila", "school", user_context={}, school_name_captured=True
        )
        result, _ = await _call_node(state)
        text = result["messages"][0].content
        assert text.startswith("¡Hola! ")
        assert "¡Hola,!" not in text

    async def test_unknown_code_name_falls_back_to_agent_type_role(self):
        # Covers APPOINTMENTS_EXTRA_CODE_NAMES: extra code names reuse the
        # appointments builder but are absent from the profile table.
        state = _base_state("Hola", "lucia", "appointments")
        result, _ = await _call_node(state)
        text = result["messages"][0].content
        assert "Lucia" in text
        assert "asistente de citas" in text

    async def test_unknown_everything_uses_generic_role(self):
        state = _base_state("Hola", "", "")
        result, _ = await _call_node(state)
        text = result["messages"][0].content
        assert "Asistente" in text
        assert "tu asistente virtual" in text

    def test_has_contact_name_truth_table(self):
        assert has_contact_name({"user_context": {"name": "Ana"}}) is True
        assert has_contact_name({"user_context": {"name": "  Ana  "}}) is True
        assert has_contact_name({"user_context": {"name": ""}}) is False
        assert has_contact_name({"user_context": {"name": "   "}}) is False
        assert has_contact_name({"user_context": {}}) is False
        assert has_contact_name({"user_context": None}) is False
        assert has_contact_name({}) is False
        assert has_contact_name({"user_context": "not-a-dict"}) is False


# ── Part B — routing matrix over all five graphs ─────────────────────────────

CASES = [
    ("camila", "school", build_camila_graph, "src.graphs.camila_graph.camila_triage_node"),
    ("helena", "sales", build_sales_graph, "src.graphs.sales_graph.triage_node"),
    ("sofia", "school", build_school_graph, "src.graphs.school_graph.triage_node"),
    ("giulia", "restaurant", build_restaurant_graph, "src.graphs.restaurant_graph.triage_node"),
    ("marco", "appointments", build_appointments_graph, "src.graphs.appointments_graph.triage_node"),
]


def _greeting_triage_mock():
    return AsyncMock(
        return_value={
            "intent": IntentType.GREETING.value,
            "structured_intent": StructuredIntent(
                intent=IntentType.GREETING, confidence=0.99
            ),
        }
    )


class TestGreetingRouting:
    @pytest.mark.parametrize(
        ("code_name", "agent_type", "builder", "triage_target"), CASES
    )
    async def test_greeting_with_known_name_routes_to_greeting_response(
        self, code_name, agent_type, builder, triage_target
    ):
        provider = _FakeProvider(chunks=("¡Hola, Ana!", " ¿En qué te ayudo?"))
        with (
            patch(triage_target, new=_greeting_triage_mock()),
            patch(
                "src.agents.greeting_response.get_provider", return_value=provider
            ),
            patch(
                "src.agents.greeting_response.resolve_model",
                return_value="model-test",
            ),
        ):
            graph = builder(MemorySaver())
            results = []
            async for chunk in graph.astream(
                _base_state("Hola", code_name, agent_type),
                config=_config(code_name),
                stream_mode="updates",
            ):
                results.append(chunk)

        names = _node_names(results)
        assert "greeting_response" in names
        assert "faq_response" not in names

        greeting_update = next(r["greeting_response"] for r in results if "greeting_response" in r)
        # One LLM call → exactly one usage record for billing
        assert [u["node"] for u in greeting_update["turn_usage"]] == [
            "greeting_response"
        ]
        messages = greeting_update["messages"]
        assert len(messages) == 1
        assert messages[0].content == "¡Hola, Ana! ¿En qué te ayudo?"


# ── Part C — gates, precedence, negatives ────────────────────────────────────


class TestGreetingGates:
    async def test_camila_greeting_without_name_still_routes_to_name_capture(self):
        mock_name = AsyncMock(
            return_value={
                "messages": [AIMessage(content="¿Cuál es tu nombre completo?")],
                "name_capture_reply_sent": True,
            }
        )
        with (
            patch("src.graphs.camila_graph.camila_triage_node", new=_greeting_triage_mock()),
            patch("src.graphs.camila_graph.name_capture_node", new=mock_name),
        ):
            graph = build_camila_graph(MemorySaver())
            results = []
            async for chunk in graph.astream(
                _base_state(
                    "Hola", "camila", "school",
                    user_context={}, school_name_captured=False,
                ),
                config=_config("camila"),
                stream_mode="updates",
            ):
                results.append(chunk)

        names = _node_names(results)
        assert "name_capture" in names
        assert "greeting_response" not in names
        assert "faq_response" not in names

    async def test_school_greeting_without_name_routes_to_name_capture(self):
        # Proactive name capture: a greeting from an unknown contact now asks
        # for the name instead of falling through to FAQ.
        mock_name = AsyncMock(
            return_value={
                "messages": [AIMessage(content="¿Podrías decirme tu nombre?")],
                "name_capture_reply_sent": True,
            }
        )
        with (
            patch("src.graphs.school_graph.triage_node", new=_greeting_triage_mock()),
            patch("src.graphs.school_graph.name_capture_node", new=mock_name),
        ):
            graph = build_school_graph(MemorySaver())
            results = []
            async for chunk in graph.astream(
                _base_state("Hola", "sofia", "school", user_context={}),
                config=_config("sofia"),
                stream_mode="updates",
            ):
                results.append(chunk)

        names = _node_names(results)
        assert "name_capture" in names
        assert "greeting_response" not in names
        assert "faq_response" not in names

    async def test_sales_faq_intent_does_not_hit_greeting_response(self):
        mock_triage = AsyncMock(return_value={"intent": "faq"})
        mock_faq = AsyncMock(
            return_value={"messages": [AIMessage(content="Nuestro horario es...")]}
        )
        with (
            patch("src.graphs.sales_graph.triage_node", new=mock_triage),
            patch("src.graphs.sales_graph.faq_response_node", new=mock_faq),
        ):
            graph = build_sales_graph(MemorySaver())
            results = []
            async for chunk in graph.astream(
                _base_state("¿Cuál es el horario?", "helena", "sales"),
                config=_config("helena"),
                stream_mode="updates",
            ):
                results.append(chunk)

        names = _node_names(results)
        assert "faq_response" in names
        assert "greeting_response" not in names

    async def test_appointments_mid_booking_flow_wins_over_stale_greeting(self):
        # A follow-up "hola" mid-booking: triage skips re-classification
        # (returns {}) and the router's booking_intent check must win even
        # though the stale intent says greeting and the name is known.
        mock_triage = AsyncMock(return_value={})
        mock_collect = AsyncMock(return_value={})
        with (
            patch("src.graphs.appointments_graph.triage_node", new=mock_triage),
            patch("src.graphs.appointments_graph.appointment_collect_node", new=mock_collect),
        ):
            graph = build_appointments_graph(MemorySaver())
            results = []
            async for chunk in graph.astream(
                _base_state(
                    "hola", "marco", "appointments",
                    intent="greeting", booking_intent="book",
                ),
                config=_config("marco"),
                stream_mode="updates",
            ):
                results.append(chunk)

        names = _node_names(results)
        assert "appointment_collect" in names
        assert "greeting_response" not in names


# ── Part D — LLM path: prompt resolution, placeholders, usage ────────────────


class TestGreetingLlmPath:
    async def test_llm_reply_is_used_and_usage_recorded(self):
        provider = _FakeProvider(chunks=("¡Buenos días, Ana! 😊 ", "Soy Helena."))
        state = _base_state("buenos días", "helena", "sales")
        result, events = await _call_node(state, provider=provider)

        text = result["messages"][0].content
        assert text == "¡Buenos días, Ana! 😊 Soy Helena."
        assert events == [{"type": "token", "content": text}]
        assert result["turn_usage"] == [
            {
                "node": "greeting_response",
                "provider": "fake",
                "model": "model-test",
                "input_tokens": 10,
                "output_tokens": 5,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
            }
        ]

    async def test_default_prompt_renders_placeholders_into_system_message(self):
        provider = _FakeProvider()
        state = _base_state("Hola", "helena", "sales")
        await _call_node(state, provider=provider)

        [messages] = provider.calls
        system = messages[0]
        assert system["role"] == "system"
        assert "Helena" in system["content"]
        assert "tu asesora de ventas" in system["content"]
        assert "Ana" in system["content"]
        assert "Always respond in Spanish." in system["content"]
        # No unrendered placeholders left behind
        assert "{persona}" not in system["content"]
        assert "{role}" not in system["content"]
        assert "{name}" not in system["content"]
        assert "{language_rule}" not in system["content"]
        # The user's actual greeting (full trailing burst) is the user turn
        assert messages[1] == {"role": "user", "content": "Hola"}

    async def test_tenant_prompt_override_is_used(self):
        provider = _FakeProvider()
        state = _base_state("Hola", "helena", "sales")
        config = {
            "configurable": {
                "prompts": {
                    "SALES_GREETING": {
                        "content": "PERSONALIZADO {persona} ({role}) para {name}. {language_rule}",
                    }
                }
            }
        }
        await _call_node(state, config=config, provider=provider)

        [messages] = provider.calls
        assert messages[0]["content"] == (
            "PERSONALIZADO Helena (tu asesora de ventas) para Ana. "
            "Always respond in Spanish."
        )

    async def test_prompt_key_follows_agent_type(self):
        provider = _FakeProvider()
        state = _base_state("Hola", "giulia", "restaurant")
        config = {
            "configurable": {
                "prompts": {
                    "RESTAURANT_GREETING": {"content": "SALUDO RESTAURANTE {persona}"},
                    "SALES_GREETING": {"content": "SALUDO TIENDA {persona}"},
                }
            }
        }
        await _call_node(state, config=config, provider=provider)

        [messages] = provider.calls
        assert messages[0]["content"] == "SALUDO RESTAURANTE Giulia"

    async def test_unknown_placeholders_stay_literal(self):
        provider = _FakeProvider()
        state = _base_state("Hola", "helena", "sales")
        config = {
            "configurable": {
                "prompts": {
                    "SALES_GREETING": {"content": "Hola {name} — cupón: {codigo_promo}"}
                }
            }
        }
        await _call_node(state, config=config, provider=provider)

        [messages] = provider.calls
        assert messages[0]["content"] == "Hola Ana — cupón: {codigo_promo}"

    async def test_malformed_braces_do_not_crash(self):
        provider = _FakeProvider()
        state = _base_state("Hola", "helena", "sales")
        config = {
            "configurable": {
                "prompts": {"SALES_GREETING": {"content": "Saluda { con estilo"}}
            }
        }
        result, _ = await _call_node(state, config=config, provider=provider)

        [messages] = provider.calls
        assert messages[0]["content"] == "Saluda { con estilo"
        assert result["messages"][0].content  # still greeted

    async def test_empty_llm_reply_falls_back_to_template(self):
        provider = _FakeProvider(chunks=("   ",))
        state = _base_state("Hola", "helena", "sales")
        result, events = await _call_node(state, provider=provider)

        text = result["messages"][0].content
        assert "Helena" in text and "asesora de ventas" in text
        # Fallback path contributes no usage record
        assert result["turn_usage"] == []
        assert events == [{"type": "token", "content": text}]

    async def test_multi_message_burst_reaches_the_llm(self):
        provider = _FakeProvider()
        state = _base_state("Hola", "helena", "sales")
        state["messages"] = [
            HumanMessage(content="buenos días"),
            HumanMessage(content="espero estés muy bien!"),
        ]
        await _call_node(state, provider=provider)

        [messages] = provider.calls
        assert messages[1] == {
            "role": "user",
            "content": "buenos días\nespero estés muy bien!",
        }
