"""Unit tests for helpers in src.agents.appointments.appointment_collect.

Covers the sole-type auto-selection that fixes the "agent goes silent after
promising to search availability" bug: a bare confirmation ("sí, agéndala")
to a one-option offer must resolve the appointment type structurally so
``booking_complete`` can become True and the graph chains into
``availability_lookup`` instead of ending the turn.
"""

from src.agents.appointments.appointment_collect import _autoselect_sole_type


class TestAutoselectSoleType:
    def test_keeps_existing_selection(self):
        types = [{"id": "a"}, {"id": "b"}]
        assert _autoselect_sole_type(types, "b") == "b"

    def test_adopts_sole_type_when_none_selected(self):
        # The reported scenario: one "Concejería" type, user just says "sí".
        assert _autoselect_sole_type([{"id": "only"}], None) == "only"

    def test_no_autoselect_when_multiple_types(self):
        # Ambiguous — must ask the user which one.
        assert _autoselect_sole_type([{"id": "a"}, {"id": "b"}], None) is None

    def test_no_autoselect_when_empty_catalog(self):
        assert _autoselect_sole_type([], None) is None

    def test_existing_selection_wins_over_sole_type(self):
        assert _autoselect_sole_type([{"id": "only"}], "already") == "already"
