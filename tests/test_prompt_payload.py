"""
Rolling-deploy contract for PromptPayload: the NestJS backend and this
service deploy independently, so the model must (a) accept payloads that
carry the new `id` provenance field, and (b) silently ignore fields it does
not know yet — in either deploy order.
"""

from src.main import PromptPayload


class TestPromptPayloadDeployContract:
    def test_parses_id_when_present(self):
        payload = PromptPayload.model_validate(
            {
                "content": "hola",
                "version": 3,
                "id": "5f2b7d1e-aaaa-bbbb-cccc-000000000000",
            }
        )
        assert payload.id == "5f2b7d1e-aaaa-bbbb-cccc-000000000000"
        assert payload.version == 3

    def test_id_defaults_to_empty_for_older_backends(self):
        payload = PromptPayload.model_validate({"content": "hola", "version": 1})
        assert payload.id == ""

    def test_unknown_fields_are_ignored(self):
        """pydantic v2 default extra='ignore' — a newer backend adding fields
        must never 422 this service."""
        payload = PromptPayload.model_validate(
            {
                "content": "hola",
                "version": 2,
                "id": "u-1",
                "isDefault": False,
                "modelConfig": {"model": "gpt-5"},
                "some_future_field": [1, 2, 3],
            }
        )
        assert payload.content == "hola"
        assert payload.id == "u-1"
        assert not hasattr(payload, "some_future_field")

    def test_default_slot_has_empty_id(self):
        """The chat_stream_start custom_prompts log keys on truthy id — a
        default slot (id "") must be excluded by that filter."""
        payload = PromptPayload()
        assert payload.id == ""
        assert not payload.id
