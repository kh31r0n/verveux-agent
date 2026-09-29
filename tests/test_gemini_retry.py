"""Gemini-only 429/RESOURCE_EXHAUSTED backoff retry (src/providers/gemini.py).

Guards that the quota-exhaustion retry lives entirely in the Gemini provider:
it retries a pre-stream 429, gives up after gemini_max_retries, never retries a
non-rate-limit error, and never retries once a chunk has been yielded.
"""

import pytest

from src.config import settings
from src.providers.base import UsageInfo
from src.providers.gemini import GeminiProvider, _is_gemini_rate_limit


class _RateLimitErr(Exception):
    def __init__(self, msg="429 RESOURCE_EXHAUSTED"):
        super().__init__(msg)
        self.code = 429


class _Chunk:
    def __init__(self, text: str):
        self.text = text
        self.usage_metadata = None


async def _aiter(chunks):
    for c in chunks:
        yield c


class _FakeModels:
    def __init__(self, fail_times: int, error: Exception, mid_stream: bool = False):
        self.calls = 0
        self.fail_times = fail_times
        self.error = error
        self.mid_stream = mid_stream

    async def generate_content_stream(self, **kwargs):
        self.calls += 1
        if self.mid_stream:
            async def _boom():
                yield _Chunk("parcial")
                raise self.error
            return _boom()
        if self.calls <= self.fail_times:
            raise self.error
        return _aiter([_Chunk("hola")])


class _FakeClient:
    def __init__(self, models):
        self.aio = type("Aio", (), {"models": models})()


def _provider(models) -> GeminiProvider:
    # Bypass __init__ (it builds a real google-genai Client needing creds).
    p = object.__new__(GeminiProvider)
    p._client = _FakeClient(models)
    p.last_usage = UsageInfo()
    return p


async def _collect(provider) -> str:
    out = ""
    async for t in provider.stream_chat(
        messages=[{"role": "user", "content": "x"}], model="gemini"
    ):
        out += t
    return out


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    # base=0 → full-jitter delay is 0, so the retries don't actually sleep.
    monkeypatch.setattr(settings, "gemini_retry_base_seconds", 0.0)
    monkeypatch.setattr(settings, "gemini_max_retries", 3)


def test_is_gemini_rate_limit_detects_429_and_text():
    assert _is_gemini_rate_limit(_RateLimitErr())
    assert _is_gemini_rate_limit(Exception("boom RESOURCE_EXHAUSTED"))
    assert not _is_gemini_rate_limit(ValueError("bad request"))


async def test_retries_then_succeeds():
    models = _FakeModels(fail_times=2, error=_RateLimitErr())
    out = await _collect(_provider(models))
    assert out == "hola"
    assert models.calls == 3  # 2 failures + 1 success


async def test_gives_up_after_max_retries():
    models = _FakeModels(fail_times=99, error=_RateLimitErr())
    with pytest.raises(Exception):
        await _collect(_provider(models))
    assert models.calls == 4  # initial + gemini_max_retries (3)


async def test_non_rate_limit_error_not_retried():
    models = _FakeModels(fail_times=99, error=ValueError("bad request"))
    with pytest.raises(ValueError):
        await _collect(_provider(models))
    assert models.calls == 1  # no retry


async def test_mid_stream_error_not_retried():
    # A 429 raised after a chunk was already yielded must NOT be replayed.
    models = _FakeModels(fail_times=0, error=_RateLimitErr(), mid_stream=True)
    with pytest.raises(Exception):
        await _collect(_provider(models))
    assert models.calls == 1


# ── ADC credential fallback ──────────────────────────────────────────────────


class TestGeminiAdcFallback:
    """With no credential blob anywhere, fall back to Application Default
    Credentials — on Cloud Run that is the agent's own service account, which
    already holds roles/aiplatform.user. Explicit credentials still win."""

    def test_falls_back_to_adc_when_nothing_is_configured(self, monkeypatch):
        from src.config import settings as app_settings
        from src.providers import gemini

        monkeypatch.setattr(app_settings, "gemini_service_account_json", "")
        assert gemini.resolve_gemini_credentials({"configurable": {}}) == {
            "type": "adc"
        }

    def test_per_tenant_credentials_take_precedence(self, monkeypatch):
        from src.config import settings as app_settings
        from src.providers import gemini

        monkeypatch.setattr(app_settings, "gemini_service_account_json", "")
        blob = {"type": "service_account", "project_id": "tenant-proj"}
        assert (
            gemini.resolve_gemini_credentials(
                {"configurable": {"gemini_credentials": blob}}
            )
            == blob
        )

    def test_project_comes_from_the_credential_blob_before_adc(self, monkeypatch):
        """ADC would report the *platform* project and bill the wrong one."""
        from src.config import settings as app_settings
        from src.providers import gemini

        monkeypatch.setattr(app_settings, "gemini_project_id", "")
        monkeypatch.setattr(
            gemini, "_load_adc", lambda: (object(), "platform-proj")
        )
        blob = {"type": "service_account", "project_id": "tenant-proj"}
        assert (
            gemini.resolve_gemini_project_id({"configurable": {}}, blob)
            == "tenant-proj"
        )

    def test_project_falls_back_to_adc(self, monkeypatch):
        from src.config import settings as app_settings
        from src.providers import gemini

        monkeypatch.setattr(app_settings, "gemini_project_id", "")
        monkeypatch.setattr(
            gemini, "_load_adc", lambda: (object(), "platform-proj")
        )
        assert (
            gemini.resolve_gemini_project_id({"configurable": {}}, {"type": "adc"})
            == "platform-proj"
        )

    def test_location_defaults_to_global_not_a_region(self, monkeypatch):
        """ADC carries no region, and refusing to serve is worse than a default.

        The default must stay "global": Gemini 3 is published only to the global
        endpoint, so a regional default (us-central1, as this was) 404s every
        3.x model id while the 2.5 family keeps working — a downgrade that looks
        like a model-name typo.
        """
        from src.config import settings as app_settings
        from src.providers import gemini

        monkeypatch.setattr(app_settings, "gemini_location", "")
        assert gemini.resolve_gemini_location({"configurable": {}}) == "global"

    def test_configured_location_still_wins(self, monkeypatch):
        from src.providers import gemini

        assert (
            gemini.resolve_gemini_location(
                {"configurable": {"gemini_location": "europe-west4"}}
            )
            == "europe-west4"
        )
