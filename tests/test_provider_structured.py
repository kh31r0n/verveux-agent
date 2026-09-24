"""Structured output on the provider layer (`generate_structured`)."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from src.config import settings
from src.providers import gemini as gemini_module
from src.providers.base import ChatProvider, UsageInfo, parse_structured
from src.providers.errors import StructuredOutputError
from src.providers.gemini import GeminiProvider


class Answer(BaseModel):
    label: str
    score: int


class _Response:
    def __init__(self, text, *, finish="STOP", prompt=100, visible=20, thoughts=30):
        self.text = text
        self.candidates = [SimpleNamespace(finish_reason=SimpleNamespace(name=finish))]
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=prompt,
            candidates_token_count=visible,
            thoughts_token_count=thoughts,
            cached_content_token_count=0,
            total_token_count=prompt + visible + thoughts,
        )


class _Models:
    def __init__(self, results):
        self.results = list(results)
        self.configs = []

    async def generate_content(self, *, model, contents, config):
        self.configs.append(config)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _Err(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def _gemini(models) -> GeminiProvider:
    provider = object.__new__(GeminiProvider)
    provider._client = SimpleNamespace(aio=SimpleNamespace(models=models))
    provider.last_usage = UsageInfo()
    return provider


MESSAGES = [{"role": "system", "content": "Clasifica."}, {"role": "user", "content": "hola"}]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _instant(_seconds):
        return None

    monkeypatch.setattr(gemini_module.asyncio, "sleep", _instant)


async def test_gemini_uses_json_mode_schema_and_thinking_budget():
    models = _Models([_Response('{"label": "x", "score": 3}')])
    result = await _gemini(models).generate_structured(MESSAGES, "gemini-test", Answer, thinking_budget=0)
    assert result == Answer(label="x", score=3)
    config = models.configs[0]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is Answer
    assert config.system_instruction == "Clasifica."
    assert config.thinking_config.thinking_budget == 0


async def test_default_thinking_leaves_the_budget_unset():
    models = _Models([_Response('{"label": "x", "score": 3}')])
    await _gemini(models).generate_structured(MESSAGES, "m", Answer)
    assert models.configs[0].thinking_config is None


async def test_thinking_is_billed_as_output():
    provider = _gemini(_Models([_Response('{"label": "x", "score": 1}', visible=20, thoughts=30)]))
    await provider.generate_structured(MESSAGES, "m", Answer)
    assert provider.last_usage.output_tokens == 50
    assert provider.last_usage.reasoning_tokens == 30
    assert provider.last_usage.input_tokens == 100


async def test_truncation_is_reported_as_such():
    provider = _gemini(_Models([_Response('{"label": "x"', finish="MAX_TOKENS")]))
    with pytest.raises(StructuredOutputError) as err:
        await provider.generate_structured(MESSAGES, "m", Answer)
    assert err.value.kind == "truncated"


async def test_schema_violation_is_classified():
    provider = _gemini(_Models([_Response('{"label": "x", "score": "many"}')]))
    with pytest.raises(StructuredOutputError) as err:
        await provider.generate_structured(MESSAGES, "m", Answer)
    assert err.value.kind == "schema_mismatch"


async def test_rate_limit_is_retried_with_the_existing_policy(monkeypatch):
    monkeypatch.setattr(settings, "gemini_max_retries", 2)
    models = _Models([_Err(429, "RESOURCE_EXHAUSTED"), _Response('{"label": "x", "score": 1}')])
    await _gemini(models).generate_structured(MESSAGES, "m", Answer)
    assert len(models.configs) == 2


async def test_rate_limit_gives_up_after_the_budget(monkeypatch):
    monkeypatch.setattr(settings, "gemini_max_retries", 1)
    models = _Models([_Err(429, "RESOURCE_EXHAUSTED")] * 3)
    with pytest.raises(_Err):
        await _gemini(models).generate_structured(MESSAGES, "m", Answer)
    assert len(models.configs) == 2


async def test_model_that_cannot_disable_thinking_is_retried_without_a_budget():
    models = _Models(
        [_Err(400, "INVALID_ARGUMENT: thinking_budget 0 is not supported"), _Response('{"label": "x", "score": 1}')]
    )
    await _gemini(models).generate_structured(MESSAGES, "gemini-pro", Answer, thinking_budget=0)
    assert models.configs[0].thinking_config is not None
    assert models.configs[1].thinking_config is None


async def test_other_bad_requests_are_not_retried():
    models = _Models([_Err(400, "INVALID_ARGUMENT: bad schema")])
    with pytest.raises(_Err):
        await _gemini(models).generate_structured(MESSAGES, "m", Answer, thinking_budget=0)
    assert len(models.configs) == 1


class _TextProvider(ChatProvider):
    def __init__(self, text):
        super().__init__()
        self.text = text

    async def stream_chat(self, messages, model, **kwargs):
        self.last_usage = UsageInfo(input_tokens=7, output_tokens=3)
        yield self.text


async def test_base_fallback_parses_fenced_json():
    provider = _TextProvider('```json\n{"label": "y", "score": 2}\n```')
    assert await provider.generate_structured(MESSAGES, "gpt", Answer) == Answer(label="y", score=2)
    assert provider.last_usage.input_tokens == 7


@pytest.mark.parametrize(
    "text, kind", [("", "empty"), ("not json", "invalid_json"), ('{"label": 1}', "schema_mismatch")]
)
def test_parse_structured_classifies_failures(text, kind):
    with pytest.raises(StructuredOutputError) as err:
        parse_structured(text, Answer)
    assert err.value.kind == kind
