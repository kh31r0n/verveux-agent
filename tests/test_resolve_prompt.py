"""
Coverage for `resolve_prompt` — the helper every agent node uses to load
its system prompt. The persona substitution path added when `state` is
threaded through is load-bearing across 20 sites in `src/agents/`.
"""

from src.agents.utils import resolve_prompt


def make_config(prompts=None):
    return {"configurable": {"prompts": prompts or {}}}


class TestFallback:
    def test_returns_fallback_when_config_has_no_prompts(self):
        out = resolve_prompt(make_config(), "SALES_FAQ", "fallback content")
        assert out == "fallback content"

    def test_returns_fallback_when_prompt_type_missing(self):
        out = resolve_prompt(
            make_config({"OTHER_TYPE": {"content": "other"}}),
            "SALES_FAQ",
            "fallback content",
        )
        assert out == "fallback content"

    def test_returns_fallback_when_payload_lacks_content(self):
        out = resolve_prompt(
            make_config({"SALES_FAQ": {"version": 3}}),
            "SALES_FAQ",
            "fallback content",
        )
        assert out == "fallback content"


class TestCustomizedContent:
    def test_returns_customized_content_over_fallback(self):
        out = resolve_prompt(
            make_config({"SALES_FAQ": {"content": "tenant edition"}}),
            "SALES_FAQ",
            "fallback content",
        )
        assert out == "tenant edition"


class TestPersonaSubstitution:
    fallback_with_persona = "Eres {persona}, asistente de ventas. {language_rule}"

    def test_no_substitution_when_state_is_none(self):
        out = resolve_prompt(make_config(), "SALES_FAQ", self.fallback_with_persona)
        # `{persona}` is preserved verbatim so the caller's own .format()
        # could still substitute it the legacy way.
        assert "{persona}" in out
        assert "{language_rule}" in out

    def test_substitutes_when_state_has_persona(self):
        state = {"agent_persona_name": "María"}
        out = resolve_prompt(
            make_config(), "SALES_FAQ", self.fallback_with_persona, state
        )
        assert "{persona}" not in out
        assert "María" in out
        # Other placeholders are left alone for downstream .format()
        assert "{language_rule}" in out

    def test_falls_back_to_helena_when_state_lacks_persona(self):
        state = {"agent_persona_name": ""}
        out = resolve_prompt(
            make_config(), "SALES_FAQ", self.fallback_with_persona, state
        )
        assert "{persona}" not in out
        assert "Helena" in out

    def test_falls_back_to_helena_when_state_missing_key(self):
        out = resolve_prompt(
            make_config(), "SALES_FAQ", self.fallback_with_persona, {}
        )
        assert "Helena" in out

    def test_treats_whitespace_only_persona_as_unset(self):
        out = resolve_prompt(
            make_config(),
            "SALES_FAQ",
            self.fallback_with_persona,
            {"agent_persona_name": "   "},
        )
        assert "Helena" in out
        assert "   " not in out  # not propagated as literal persona

    def test_substitutes_customized_content_too(self):
        state = {"agent_persona_name": "Admisiones"}
        out = resolve_prompt(
            make_config({"SCHOOL_TRIAGE": {"content": "Hola, soy {persona}. ¿En qué te ayudo?"}}),
            "SCHOOL_TRIAGE",
            "fallback",
            state,
        )
        assert out == "Hola, soy Admisiones. ¿En qué te ayudo?"

    def test_supports_multiple_persona_occurrences(self):
        state = {"agent_persona_name": "Helena"}
        out = resolve_prompt(
            make_config(),
            "X",
            "Soy {persona}. Te habla {persona}.",
            state,
        )
        assert out == "Soy Helena. Te habla Helena."

    def test_passes_through_template_without_persona_placeholder(self):
        state = {"agent_persona_name": "Helena"}
        out = resolve_prompt(
            make_config(),
            "X",
            "Eres un asistente. {language_rule}",
            state,
        )
        # Identical to the input — nothing to replace.
        assert out == "Eres un asistente. {language_rule}"


class TestComposesWithDownstreamFormat:
    """The replacement runs before the caller's .format(...), so subsequent
    formatting continues to work for other placeholders."""

    def test_persona_substituted_then_format_fills_remaining_placeholders(self):
        state = {"agent_persona_name": "Helena"}
        template = "Eres {persona}. {language_rule}"
        resolved = resolve_prompt(make_config(), "X", template, state)
        final = resolved.format(language_rule="Always respond in Spanish.")
        assert final == "Eres Helena. Always respond in Spanish."

    def test_extra_persona_kwarg_to_format_is_harmless(self):
        """Some nodes (the ones edited in the earlier phase) still pass
        persona=... to .format(). After resolve_prompt's pre-substitution
        the placeholder is gone, so the extra kwarg is a no-op."""
        state = {"agent_persona_name": "Helena"}
        resolved = resolve_prompt(make_config(), "X", "Eres {persona}.", state)
        # No KeyError, no double substitution.
        final = resolved.format(persona="ignored")
        assert final == "Eres Helena."
