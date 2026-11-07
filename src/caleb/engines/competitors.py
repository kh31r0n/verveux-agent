"""Supplier / Competitor tier classifier and explicit-range operational scale estimator."""

from __future__ import annotations

from typing import Any


def classify_competitor_and_scale(
    entity: dict[str, Any],
    reference_brands: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classifies a supplier/competitor into competitive tiers and estimates scale using explicit ranges.

    Never invents single-point revenue numbers: when verified signals exist, emits an
    explicit bracketed range with documented rationale; when no scale signals exist,
    emits ``"sin dato"``.
    """
    signals = [str(s).lower() for s in (entity.get("signals") or [])]
    observed_brands = [
        str(b).strip()
        for b in (entity.get("brands") or [])
        if b and str(b).strip().lower() != "sin dato"
    ]
    branches = entity.get("branchesCount")
    is_importer = bool(entity.get("isImporter")) or any(
        "importador" in s or "comercio exterior" in s for s in signals
    )
    has_national = any("cobertura nacional" in s or "multi-sede" in s for s in signals)
    public_contracts = int(entity.get("publicContractsCount") or 0)

    matched_ref_brands: list[dict[str, str]] = []
    for ref in reference_brands or []:
        ref_name = str(ref.get("name") or "").strip()
        if not ref_name:
            continue
        if any(ref_name.lower() == ob.lower() for ob in observed_brands) or any(
            ref_name.lower() in s for s in signals
        ):
            matched_ref_brands.append(
                {
                    "name": ref_name,
                    "role": str(ref.get("role") or "MARKET_REFERENCE"),
                }
            )

    if is_importer or has_national or (isinstance(branches, int) and branches >= 3):
        competitive_level = "NIVEL_1_IMPORTADOR_MAYORISTA"
        scale_range = "USD 5M - 15M (rango estimado por perfil importador / multi-sede)"
        scale_confidence = "ALTA" if is_importer and isinstance(branches, int) else "MEDIA"
    elif (
        (isinstance(branches, int) and branches >= 2)
        or public_contracts >= 2
        or len(matched_ref_brands) >= 1
        or any("mayorista" in s or "distribuidor autorizado" in s for s in signals)
    ):
        competitive_level = "NIVEL_2_DISTRIBUIDOR_REGIONAL"
        scale_range = "USD 1M - 5M (rango estimado por operación regional B2B)"
        scale_confidence = "MEDIA"
    elif len(signals) > 0 or (entity.get("website") and entity.get("website") != "sin dato"):
        competitive_level = "NIVEL_3_COMERCIALIZADOR_LOCAL"
        scale_range = "USD 200K - 1M (rango estimado por punto comercial único)"
        scale_confidence = "BAJA"
    else:
        competitive_level = "NO_CLASIFICADO"
        scale_range = "sin dato"
        scale_confidence = "SIN_DATO"

    result = dict(entity)
    result["competitiveLevel"] = competitive_level
    result["estimatedAnnualRevenueRangeUsd"] = scale_range
    result["scaleConfidence"] = scale_confidence
    result["matchedReferenceBrands"] = matched_ref_brands
    return result
