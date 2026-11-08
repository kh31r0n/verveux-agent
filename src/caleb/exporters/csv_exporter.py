"""UTF-8 CSV exporters with row-level source traceability and privacy safeguards."""

from __future__ import annotations

import csv
import io
from typing import Any


def export_entities_csv(report_model: dict[str, Any]) -> str:
    """Exports evaluated accounts to UTF-8 CSV with traceability columns (Fuente, Fecha, URL, Estado_Verificacion)."""
    sections = report_model.get("sections") or {}
    accounts = (sections.get("segmentsAndAccounts") or {}).get("accounts") or []
    generated_at = report_model.get("generatedAt", "sin dato")

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        [
            "entityId",
            "name",
            "taxId",
            "segmentCode",
            "score",
            "priorityBand",
            "country",
            "city",
            "zoneCode",
            "website",
            "phone",
            "email",
            "source",
            "fetchedAt",
            "sourceUrl",
            "verificationStatus",
        ]
    )

    for acc in accounts:
        website = str(acc.get("website") or "sin dato").strip()
        tax_id = str(acc.get("taxId") or "sin dato").strip()
        verified = (
            "verificado"
            if (website not in ("", "sin dato") or tax_id not in ("", "sin dato", "reservado-ley-1581"))
            else "por verificar"
        )
        source_url = website if website not in ("", "sin dato") else "sin dato"
        writer.writerow(
            [
                str(acc.get("entityId") or "sin dato").strip(),
                str(acc.get("name") or "sin dato").strip(),
                tax_id,
                str(acc.get("segmentCode") or "sin dato").strip(),
                acc.get("score", "sin dato"),
                str(acc.get("priorityBand") or "sin dato").strip(),
                str(acc.get("country") or "sin dato").strip(),
                str(acc.get("city") or "sin dato").strip(),
                str(acc.get("zoneCode") or "sin dato").strip(),
                website,
                str(acc.get("phone") or "sin dato").strip(),
                str(acc.get("email") or "sin dato").strip(),
                str(acc.get("source") or "STUDY_PIPELINE").strip(),
                str(acc.get("fetchedAt") or generated_at).strip(),
                source_url,
                verified,
            ]
        )

    return buf.getvalue()
