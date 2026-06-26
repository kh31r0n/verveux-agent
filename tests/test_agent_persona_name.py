"""
Round-trip tests for `agent_persona_name`: the request body field defaults
to an empty string, accepts an explicit value, and flows into AgentState so
downstream prompt nodes can read it.
"""

from src.main import ChatStreamRequest


class TestChatStreamRequestPersona:
    def test_defaults_to_empty_string(self):
        req = ChatStreamRequest(thread_id="t-1", message="Hola")
        assert req.agent_persona_name == ""

    def test_accepts_explicit_persona(self):
        req = ChatStreamRequest(
            thread_id="t-1",
            message="Hola",
            agent_persona_name="Helena",
        )
        assert req.agent_persona_name == "Helena"

    def test_accepts_accented_persona(self):
        req = ChatStreamRequest(
            thread_id="t-1",
            message="Hola",
            agent_persona_name="María",
        )
        assert req.agent_persona_name == "María"


class TestResolvePersona:
    """resolve_persona is what every conversational/close prompt calls."""

    def test_returns_default_when_state_missing_key(self):
        from src.agents.utils import resolve_persona

        assert resolve_persona({}, "Helena") == "Helena"

    def test_returns_default_when_empty_string(self):
        from src.agents.utils import resolve_persona

        assert resolve_persona({"agent_persona_name": ""}, "Helena") == "Helena"

    def test_returns_default_when_whitespace_only(self):
        from src.agents.utils import resolve_persona

        assert resolve_persona({"agent_persona_name": "   "}, "Helena") == "Helena"

    def test_returns_persona_when_set(self):
        from src.agents.utils import resolve_persona

        assert (
            resolve_persona({"agent_persona_name": "Admisión"}, "Helena")
            == "Admisión"
        )

    def test_trims_surrounding_whitespace(self):
        from src.agents.utils import resolve_persona

        assert (
            resolve_persona({"agent_persona_name": "  Helena  "}, "Default")
            == "Helena"
        )


class TestAgentStateAcceptsPersona:
    """AgentState is a TypedDict, but downstream code reads it via .get(),
    so we verify the field is declared and round-trips through state init."""

    def test_state_carries_persona(self):
        from src.graphs.state import AgentState

        # TypedDict has no runtime validation; this asserts the key is documented.
        assert "agent_persona_name" in AgentState.__annotations__

        state: AgentState = {
            "messages": [],
            "thread_id": "t-1",
            "agent_type": "sales",
            "agent_code_name": "helena",
            "agent_persona_name": "Helena",
            "agent_version": 1,
            "capabilities": {},
            "domain_state": {},
            "tenant_id": "tenant-1",
            "conversation_id": "conv-1",
            "product_catalog": [],
            "knowledge": [],
            "user_context": {},
            "contact_id": "",
            "contact_tags": [],
            "language": "es",
            "faqs": [],
            "intent": "sales",
            "structured_intent": None,
            "sales_phase": "product_selection",
            "cart": [],
            "cart_confirmed": False,
            "backend_cart_sync_failed": False,
            "product_selection_turns": 0,
            "pending_unknown_items": [],
            "product_selection_complete": False,
            "customer_data_complete": False,
            "order_data": {},
            "order_confirmed": False,
            "tracking_data": {},
            "tracking_complete": False,
            "complaint_data": {},
            "complaint_complete": False,
            "admissions_data": {},
            "admissions_complete": False,
            "restaurant_order_data": {},
            "restaurant_order_complete": False,
            "booking_data": {},
            "booking_complete": False,
            "booking_confirmed": False,
            "deal_created": False,
            "faq_used": None,
            "last_command_key": None,
            "execute_confirmed": False,
            "turn_usage": [],
            "attachments": [],
            "school_name_captured": False,
            "handoff_reason": None,
        }
        assert state["agent_persona_name"] == "Helena"
