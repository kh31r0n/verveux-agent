"""
Spend-file repair for the FinOps spend import.

When a batch insert rejects rows, the caller posts POST /agent/fix-spend-file. This
module reads the original file from S3, maps its columns toward the canonical
NormalizedSpendRecord shape (deterministic alias map + an aggressive LLM
column-mapping pass), coerces values, writes a corrected file back to S3 (same
format), and reports rows it still could not fix.

Design notes:
  - pandas/openpyxl are imported lazily so the pure coercion/mapping logic is
    unit-testable without them.
  - Per the all-or-nothing contract, unfixable rows are KEPT in the corrected file
    (so the batch keeps failing and ultimately terminates FAILED rather than
    silently dropping data). `unfixableRows` is surfaced for the tracker.
  - Aggressive mode (chosen): non-identifying fields are defaulted
    (ingestionChannel→EXCEL, unknown paymentOrigin→OTHER). Identifying/critical
    fields (amount, currency, spendDate) are never fabricated.
  - Everything is tenant-scoped: the corrected object lands under the same
    `finops/imports/{tenantId}/{batchId}/` prefix the backend writes originals to,
    so one tenant's files are never reachable under another's.
"""

from __future__ import annotations

import io
import json
import re
from datetime import datetime
from uuid import uuid4

import structlog

from ..config import settings
from ..integrations import s3

logger = structlog.get_logger(__name__)

CANONICAL_FIELDS = [
    "amount", "currency", "spendDate", "paymentOrigin", "ingestionChannel",
    "invoiceId", "sourceRef", "costItemId", "providerId", "productId",
    "countryId", "costCenterId", "categoryId", "serviceId", "businessUnitId",
]
REQUIRED_FIELDS = ["amount", "currency", "spendDate"]
PAYMENT_ORIGINS = {"SAP", "CARD", "TRANSFER", "AGREEMENT", "CLOUD", "OTHER"}
_PAYMENT_SYNONYMS = {
    "TARJETA": "CARD", "CREDIT": "CARD", "CREDITO": "CARD", "DEBIT": "CARD",
    "TRANSFERENCIA": "TRANSFER", "WIRE": "TRANSFER",
    "CONVENIO": "AGREEMENT", "CONTRATO": "AGREEMENT",
    "AWS": "CLOUD", "AZURE": "CLOUD", "GCP": "CLOUD", "NUBE": "CLOUD",
}

_ALIASES: dict[str, list[str]] = {
    "amount": ["amount", "monto", "importe", "valor", "total"],
    "currency": ["currency", "moneda", "divisa"],
    "spendDate": ["spenddate", "date", "fecha", "fechagasto", "fechadegasto"],
    "paymentOrigin": ["paymentorigin", "origen", "origenpago", "mediodepago"],
    "ingestionChannel": ["ingestionchannel", "canal"],
    "invoiceId": ["invoiceid", "invoice", "factura", "nrofactura", "numerofactura"],
    "sourceRef": ["sourceref", "ref", "referencia"],
    "costItemId": ["costitemid"],
    "providerId": ["providerid", "proveedorid"],
    "productId": ["productid", "productoid"],
    "countryId": ["countryid", "paisid"],
    "costCenterId": ["costcenterid", "centrocostoid"],
    "categoryId": ["categoryid", "categoriaid"],
    "serviceId": ["serviceid", "servicioid"],
    "businessUnitId": ["businessunitid", "unidadnegocioid"],
}
_ALIAS_LOOKUP: dict[str, str] = {}
for _c, _al in _ALIASES.items():
    _ALIAS_LOOKUP[_c.lower()] = _c
    for _a in _al:
        _ALIAS_LOOKUP[_a] = _c

_DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d", "%d.%m.%Y"]


def _norm_key(key: str) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


# ─── Value coercion (pure) ────────────────────────────────────────────────────

def _coerce_amount(value: object) -> str | None:
    s = re.sub(r"[^\d,.\-]", "", str(value).strip())
    if not s or s in {"-", ".", ","}:
        return None
    if "," in s and "." in s:
        # The right-most separator is the decimal point.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts) == 2 and 1 <= len(parts[1]) <= 2:
            s = s.replace(",", ".")  # decimal comma
        else:
            s = s.replace(",", "")   # thousands
    try:
        float(s)
        return s
    except ValueError:
        return None


def _coerce_currency(value: object) -> str | None:
    s = re.sub(r"[^A-Za-z]", "", str(value)).upper()
    return s[:3] if len(s) >= 3 else None


def _coerce_date(value: object) -> str | None:
    s = str(value).strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _coerce_payment_origin(value: object) -> str | None:
    s = re.sub(r"[^A-Za-z]", "", str(value)).upper()
    if not s:
        return None
    if s in PAYMENT_ORIGINS:
        return s
    return _PAYMENT_SYNONYMS.get(s, "OTHER")  # aggressive: unknown → OTHER


def coerce_value(field: str, value: object) -> str | None:
    if value is None:
        return None
    if field == "amount":
        return _coerce_amount(value)
    if field == "currency":
        return _coerce_currency(value)
    if field == "spendDate":
        return _coerce_date(value)
    if field == "paymentOrigin":
        return _coerce_payment_origin(value)
    if field == "ingestionChannel":
        s = re.sub(r"[^A-Za-z]", "", str(value)).upper()
        return s if s in {"MANUAL", "EXCEL", "EMAIL", "API"} else None
    s = str(value).strip()
    return s or None


# ─── Mapping (pure) ───────────────────────────────────────────────────────────

def apply_mapping(rows: list[dict], column_map: dict[str, str] | None = None) -> list[dict]:
    """
    Build canonical records from raw rows. `column_map` (raw header → canonical
    field, typically from the LLM) takes precedence; the alias map is the fallback.
    Aggressive defaults are applied for non-identifying fields. Keeps `raw`.
    """
    column_map = column_map or {}
    out: list[dict] = []
    for row in rows:
        rec: dict = {}
        for header, value in row.items():
            canonical = column_map.get(header) or _ALIAS_LOOKUP.get(_norm_key(header))
            if canonical in CANONICAL_FIELDS:
                cv = coerce_value(canonical, value)
                if cv is not None:
                    rec.setdefault(canonical, cv)
        rec.setdefault("ingestionChannel", "EXCEL")
        rec.setdefault("paymentOrigin", "OTHER")
        rec["raw"] = {str(k): row[k] for k in row}
        out.append(rec)
    return out


def find_unfixable(records: list[dict]) -> list[dict]:
    """Rows still missing a required (non-fabricatable) field after mapping."""
    unfixable: list[dict] = []
    for i, rec in enumerate(records):
        missing = [f for f in REQUIRED_FIELDS if not rec.get(f)]
        if missing:
            unfixable.append({"rowIndex": i, "reason": f"missing required field(s): {missing}"})
    return unfixable


# ─── File I/O (lazy pandas) ────────────────────────────────────────────────────

def _ext(key: str) -> str:
    return (key.rsplit(".", 1)[-1].lower() if "." in key else "xlsx")


def read_rows(data: bytes, ext: str) -> list[dict]:
    import pandas as pd

    buf = io.BytesIO(data)
    df = pd.read_csv(buf, dtype=str) if ext == "csv" else pd.read_excel(buf, dtype=str)
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")


def write_rows(rows: list[dict], ext: str) -> bytes:
    import pandas as pd

    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    if ext == "csv":
        df.to_csv(buf, index=False)
    else:
        df.to_excel(buf, index=False)  # needs openpyxl
    return buf.getvalue()


def _content_type(ext: str) -> str:
    if ext == "csv":
        return "text/csv"
    if ext == "xls":
        return "application/vnd.ms-excel"
    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# ─── LLM column-mapping inference ──────────────────────────────────────────────

_MAP_SYSTEM = (
    "You map spreadsheet column headers to a fixed spend-record schema. "
    "Reply ONLY with a JSON object mapping each ORIGINAL header to one of these "
    f"fields or null if none apply: {CANONICAL_FIELDS}. Do not invent headers."
)


async def infer_column_map(
    headers: list[str],
    sample_rows: list[dict],
    errors: list[dict],
    api_key: str | None = None,
) -> dict[str, str]:
    """Ask the LLM for a header→canonical map. Returns {} on any failure.

    Deliberately a direct AsyncOpenAI call rather than the ChatProvider abstraction:
    this is a one-shot JSON-mode structured call with no streaming, no persona and
    no per-tenant model routing, so `get_provider(config)` would buy nothing.
    """
    key = api_key or settings.openai_api_key
    if not key or not headers:
        return {}
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=key)
    user = json.dumps(
        {"headers": headers, "sampleRows": sample_rows[:5], "rejectedRows": errors[:20]},
        default=str,
    )
    resp = await client.chat.completions.create(
        model=settings.spend_fix_model,
        messages=[{"role": "system", "content": _MAP_SYSTEM}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = json.loads(resp.choices[0].message.content or "{}")
    return {
        str(h): str(f)
        for h, f in raw.items()
        if isinstance(f, str) and f in CANONICAL_FIELDS
    }


# ─── Orchestration ─────────────────────────────────────────────────────────────

async def fix_spend_file(
    *,
    tenant_id: str,
    s3_bucket: str,
    s3_key: str,
    batch_id: str,
    errors: list[dict] | None = None,
    expected_schema: str = "NormalizedSpendRecord",
    openai_api_key: str | None = None,
) -> dict:
    """Read → map/coerce → write corrected file → report. Returns the agent contract."""
    errors = errors or []
    ext = _ext(s3_key)

    rows = read_rows(s3.read_object(s3_bucket, s3_key), ext)
    headers = list(rows[0].keys()) if rows else []

    column_map: dict[str, str] = {}
    try:
        column_map = await infer_column_map(headers, rows[:5], errors, api_key=openai_api_key)
    except Exception as exc:  # noqa: BLE001 — fall back to the deterministic alias map
        logger.warning("column_map_inference_failed", error=str(exc))

    records = apply_mapping(rows, column_map)
    unfixable = find_unfixable(records)

    # Write canonical-header rows (drop the audit `raw` blob) so the caller's
    # re-parse maps cleanly. Same format as the original, same tenant prefix.
    corrected = [{k: v for k, v in r.items() if k != "raw"} for r in records]
    corrected_key = f"finops/imports/{tenant_id}/{batch_id}/corrected-{uuid4().hex[:8]}.{ext}"
    s3.write_object(s3_bucket, corrected_key, write_rows(corrected, ext), _content_type(ext))

    logger.info(
        "spend_file_fixed",
        tenant_id=tenant_id,
        batch_id=batch_id,
        rows=len(records),
        unfixable=len(unfixable),
        corrected_key=corrected_key,
        used_llm_map=bool(column_map),
    )
    return {
        "correctedS3Key": corrected_key,
        "fixedRowCount": len(records) - len(unfixable),
        "unfixableRows": unfixable,
    }
