"""
Tests for the spend-file repair logic (pure coercion/mapping) and the machine
system-key dependency. No pandas/boto3/openai required — file I/O and the LLM call
are patched.
"""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

import hashlib
import hmac

from src.finops import spend_fix
from src.auth.machine import require_system_key, verify_hmac
from src.config import settings


class _FakeReq:
    """Minimal stand-in for a Starlette Request exposing an awaitable body()."""

    def __init__(self, body: bytes):
        self._body = body

    async def body(self) -> bytes:
        return self._body


# ─── Value coercion ───────────────────────────────────────────────────────────

class TestCoercion:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1.000,50", "1000.50"),   # european thousands + decimal comma
            ("1,200.50", "1200.50"),   # us thousands + decimal point
            ("1500", "1500"),
            ("$ 1.234,00", "1234.00"),
            ("abc", None),
        ],
    )
    def test_amount(self, raw, expected):
        assert spend_fix._coerce_amount(raw) == expected

    def test_currency(self):
        assert spend_fix._coerce_currency("usd") == "USD"
        assert spend_fix._coerce_currency("US$ Dollar") == "USD"
        assert spend_fix._coerce_currency("X") is None

    @pytest.mark.parametrize(
        "raw,expected",
        [("2026-02-15", "2026-02-15"), ("15/02/2026", "2026-02-15"), ("15.02.2026", "2026-02-15"), ("nope", None)],
    )
    def test_date(self, raw, expected):
        assert spend_fix._coerce_date(raw) == expected

    def test_payment_origin_synonyms_and_default(self):
        assert spend_fix._coerce_payment_origin("Tarjeta") == "CARD"
        assert spend_fix._coerce_payment_origin("SAP") == "SAP"
        assert spend_fix._coerce_payment_origin("somethingelse") == "OTHER"  # aggressive
        assert spend_fix._coerce_payment_origin("") is None


# ─── Mapping ──────────────────────────────────────────────────────────────────

class TestMapping:
    def test_alias_mapping_with_defaults(self):
        rows = [{"Monto": "1.000,50", "Moneda": "usd", "Fecha": "15/02/2026", "Origen": "Tarjeta"}]
        out = spend_fix.apply_mapping(rows)
        assert out[0]["amount"] == "1000.50"
        assert out[0]["currency"] == "USD"
        assert out[0]["spendDate"] == "2026-02-15"
        assert out[0]["paymentOrigin"] == "CARD"
        assert out[0]["ingestionChannel"] == "EXCEL"  # default
        assert out[0]["raw"]["Monto"] == "1.000,50"

    def test_llm_column_map_takes_precedence(self):
        rows = [{"col_a": "100", "col_b": "EUR", "col_c": "2026-01-01"}]
        out = spend_fix.apply_mapping(rows, {"col_a": "amount", "col_b": "currency", "col_c": "spendDate"})
        assert out[0]["amount"] == "100"
        assert out[0]["currency"] == "EUR"
        assert out[0]["spendDate"] == "2026-01-01"

    def test_find_unfixable(self):
        records = [
            {"amount": "10", "currency": "USD", "spendDate": "2026-01-01"},
            {"amount": "10", "currency": "USD"},  # missing spendDate
        ]
        unfixable = spend_fix.find_unfixable(records)
        assert len(unfixable) == 1
        assert unfixable[0]["rowIndex"] == 1


# ─── Orchestration (I/O + LLM patched) ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_fix_spend_file_end_to_end():
    raw_rows = [{"Monto": "100", "Moneda": "USD", "Fecha": "2026-02-15", "Origen": "SAP"}]

    with (
        patch.object(spend_fix.s3, "read_object", return_value=b"bytes"),
        patch.object(spend_fix.s3, "write_object", return_value="ok"),
        patch.object(spend_fix, "read_rows", return_value=raw_rows),
        patch.object(spend_fix, "write_rows", return_value=b"out"),
        patch.object(spend_fix, "infer_column_map", new=AsyncMock(return_value={})),
    ):
        result = await spend_fix.fix_spend_file(
            tenant_id="t1",
            s3_bucket="bucket",
            s3_key="finops/imports/t1/b1/spend.xlsx",
            batch_id="b1",
            errors=[],
        )

    assert result["fixedRowCount"] == 1
    assert result["unfixableRows"] == []
    # Corrected files stay under the tenant's own prefix — one tenant's output is
    # never reachable under another's.
    assert result["correctedS3Key"].startswith("finops/imports/t1/b1/corrected-")
    assert result["correctedS3Key"].endswith(".xlsx")


@pytest.mark.asyncio
async def test_corrected_key_is_tenant_scoped():
    """The same batch id under two tenants must not collide."""
    raw_rows = [{"Monto": "100", "Moneda": "USD", "Fecha": "2026-02-15"}]
    keys = []

    for tenant in ("tenant-a", "tenant-b"):
        with (
            patch.object(spend_fix.s3, "read_object", return_value=b"bytes"),
            patch.object(spend_fix.s3, "write_object", return_value="ok"),
            patch.object(spend_fix, "read_rows", return_value=raw_rows),
            patch.object(spend_fix, "write_rows", return_value=b"out"),
            patch.object(spend_fix, "infer_column_map", new=AsyncMock(return_value={})),
        ):
            result = await spend_fix.fix_spend_file(
                tenant_id=tenant, s3_bucket="bucket", s3_key="x.csv", batch_id="same-batch"
            )
        keys.append(result["correctedS3Key"])

    assert keys[0].startswith("finops/imports/tenant-a/same-batch/")
    assert keys[1].startswith("finops/imports/tenant-b/same-batch/")


# ─── Machine system-key dependency ─────────────────────────────────────────────

class TestSystemKeyGuard:
    @pytest.mark.asyncio
    async def test_missing_key_401(self, monkeypatch):
        monkeypatch.setattr(settings, "webhook_api_key", "secret")
        with pytest.raises(HTTPException) as exc:
            await require_system_key(x_system_key=None)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_key_403(self, monkeypatch):
        monkeypatch.setattr(settings, "webhook_api_key", "secret")
        with pytest.raises(HTTPException) as exc:
            await require_system_key(x_system_key="wrong")
        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_not_configured_503(self, monkeypatch):
        monkeypatch.setattr(settings, "webhook_api_key", "")
        with pytest.raises(HTTPException) as exc:
            await require_system_key(x_system_key="anything")
        assert exc.value.status_code == 503

    @pytest.mark.asyncio
    async def test_valid_key_passes(self, monkeypatch):
        monkeypatch.setattr(settings, "webhook_api_key", "secret")
        assert await require_system_key(x_system_key="secret") is None


class TestHmac:
    def test_verify_hmac_roundtrip(self):
        body = b'{"a":1}'
        sig = "sha256=" + hmac.new(b"shh", body, hashlib.sha256).hexdigest()
        assert verify_hmac("shh", sig, body) is True
        assert verify_hmac("shh", sig, b"tampered") is False
        assert verify_hmac("shh", None, body) is False

    @pytest.mark.asyncio
    async def test_guard_requires_signature_when_secret_set(self, monkeypatch):
        monkeypatch.setattr(settings, "webhook_api_key", "k")
        monkeypatch.setattr(settings, "agent_machine_hmac_secret", "shh")
        with pytest.raises(HTTPException) as exc:
            await require_system_key(request=_FakeReq(b"{}"), x_system_key="k", x_hmac_signature=None)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_guard_rejects_bad_signature(self, monkeypatch):
        monkeypatch.setattr(settings, "webhook_api_key", "k")
        monkeypatch.setattr(settings, "agent_machine_hmac_secret", "shh")
        with pytest.raises(HTTPException) as exc:
            await require_system_key(
                request=_FakeReq(b"{}"), x_system_key="k", x_hmac_signature="sha256=bad"
            )
        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_guard_accepts_valid_signature(self, monkeypatch):
        monkeypatch.setattr(settings, "webhook_api_key", "k")
        monkeypatch.setattr(settings, "agent_machine_hmac_secret", "shh")
        body = b'{"x":1}'
        sig = "sha256=" + hmac.new(b"shh", body, hashlib.sha256).hexdigest()
        assert (
            await require_system_key(request=_FakeReq(body), x_system_key="k", x_hmac_signature=sig)
            is None
        )
