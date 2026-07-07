"""Matrix tests for the shared name-capture routing policy."""

from __future__ import annotations

import pytest

from src.graphs.shared_routing import (
    URGENT_INTENTS,
    has_name,
    is_deferred,
    should_capture_name,
)


def _state(**overrides) -> dict:
    state = {
        "user_context": {},
        "intent": "faq",
    }
    state.update(overrides)
    return state


class TestHasName:
    def test_name_in_user_context(self):
        assert has_name(_state(user_context={"name": "Ana"})) is True

    def test_whitespace_name_is_no_name(self):
        assert has_name(_state(user_context={"name": "   "})) is False

    def test_generic_latch(self):
        assert has_name(_state(name_captured=True)) is True

    def test_legacy_camila_latch(self):
        # Old camila checkpoints carry only school_name_captured.
        assert has_name(_state(school_name_captured=True)) is True

    def test_non_dict_user_context(self):
        assert has_name(_state(user_context="not-a-dict")) is False

    def test_empty_state(self):
        assert has_name({}) is False


class TestIsDeferred:
    def test_backend_snapshot_flag(self):
        assert is_deferred(_state(user_context={"name_capture_deferred": True})) is True

    def test_expired_deferral_from_backend(self):
        # The backend re-evaluates the window every turn: expired → False.
        assert (
            is_deferred(_state(user_context={"name_capture_deferred": False})) is False
        )

    def test_local_state_flag(self):
        assert is_deferred(_state(name_capture_deferred=True)) is True

    def test_default(self):
        assert is_deferred(_state()) is False


class TestShouldCaptureName:
    @pytest.mark.parametrize("intent", sorted(URGENT_INTENTS))
    def test_urgent_intents_skip(self, intent):
        assert should_capture_name(_state(), intent) is False

    def test_urgent_intent_read_from_state(self):
        assert should_capture_name(_state(intent="complaint")) is False

    def test_urgent_intent_case_insensitive(self):
        assert should_capture_name(_state(), "COMPLAINT") is False

    def test_new_user_non_urgent_asks(self):
        assert should_capture_name(_state(intent="greeting")) is True

    def test_known_name_skips(self):
        assert (
            should_capture_name(_state(user_context={"name": "Ana"}), "greeting")
            is False
        )

    def test_deferred_skips(self):
        state = _state(user_context={"name_capture_deferred": True})
        assert should_capture_name(state, "greeting") is False

    def test_expired_deferral_re_asks(self):
        state = _state(user_context={"name_capture_deferred": False})
        assert should_capture_name(state, "greeting") is True

    def test_legacy_latch_skips(self):
        assert should_capture_name(_state(school_name_captured=True), "faq") is False

    def test_explicit_intent_beats_state_intent(self):
        # Caller-supplied intent wins over the (stale) state intent.
        state = _state(intent="complaint")
        assert should_capture_name(state, "greeting") is True

    def test_urgent_set_matches_locked_taxonomy(self):
        assert URGENT_INTENTS == {
            "complaint",
            "escalation",
            "payment_proof",
            "correction_request",
            "identity_conflict",
            "appointment_cancel",
        }
