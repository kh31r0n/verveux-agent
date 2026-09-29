"""Unit tests for the shared LLM-JSON fence stripper."""

import json

from src.json_utils import strip_json_fences


class TestStripJsonFences:
    def test_plain_json_passthrough(self):
        raw = '{"intent": "order"}'
        assert strip_json_fences(raw) == raw
        assert json.loads(strip_json_fences(raw)) == {"intent": "order"}

    def test_json_fenced(self):
        raw = '```json\n{"intent": "order"}\n```'
        assert json.loads(strip_json_fences(raw)) == {"intent": "order"}

    def test_bare_fenced(self):
        raw = '```\n{"intent": "order"}\n```'
        assert json.loads(strip_json_fences(raw)) == {"intent": "order"}

    def test_fenced_with_surrounding_whitespace(self):
        raw = '  ```json\n{"a": 1}\n```  \n'
        assert json.loads(strip_json_fences(raw)) == {"a": 1}

    def test_multiline_json_fenced(self):
        raw = '```json\n{\n  "items": [\n    {"name": "Tacos", "quantity": 2}\n  ]\n}\n```'
        parsed = json.loads(strip_json_fences(raw))
        assert parsed["items"][0]["name"] == "Tacos"

    def test_non_json_text_unchanged(self):
        assert strip_json_fences("hola, quiero tacos") == "hola, quiero tacos"

    def test_empty_string(self):
        assert strip_json_fences("") == ""
