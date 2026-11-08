"""Parameterized deterministic rubric evaluators for entities (0-100 + A/B/C bands) and zones/countries (GO/EXPLORAR/NO_PRIORIZAR)."""

from __future__ import annotations

from typing import Any


def _validate_weights(items: list[dict[str, Any]], label: str) -> None:
    if not items:
        raise ValueError(f"{label} must contain at least one criterion/dimension")
    weight_sum = sum(float(c.get("weight", 0.0)) for c in items)
    if abs(weight_sum - 1.0) > 0.01 or any(float(c.get("weight", 0.0)) <= 0 for c in items):
        raise ValueError(
            f"{label} weights must be positive and sum to 1.0 (got {weight_sum:.4f})"
        )


def _infer_criterion_value_from_evidence(
    criterion_id: str,
    entity: dict[str, Any],
    product_lines: list[dict[str, Any]] | None = None,
) -> str:
    """Derives "alta" | "media" | "baja" | "sin dato" deterministically from verified evidence."""
    explicit = (entity.get("criterionValues") or {}).get(criterion_id)
    if explicit is not None and explicit != "":
        return str(explicit)

    signals = [str(s).lower() for s in (entity.get("signals") or [])]
    corpus = " ".join(
        [
            str(entity.get("name") or "").lower(),
            str(entity.get("description") or "").lower(),
            " ".join(signals),
        ]
    )

    if criterion_id == "product_affinity":
        if not corpus.strip():
            return "sin dato"
        matched_lines = 0
        for pl in product_lines or []:
            kws = [str(k).lower() for k in (pl.get("keywords") or [])]
            if any(k in corpus for k in kws if k):
                matched_lines += 1
        if matched_lines >= 2 or any("catálogo industrial" in s or "importador directo" in s for s in signals):
            return "alta"
        if matched_lines == 1 or len(signals) >= 1:
            return "media"
        return "baja"

    if criterion_id == "operational_scale":
        branches = entity.get("branchesCount")
        is_importer = bool(entity.get("isImporter")) or any(
            "importador" in s or "multi-sede" in s or "cobertura nacional" in s
            for s in signals
        )
        has_procurement = bool(entity.get("publicContractsCount", 0) > 0)
        if is_importer or (isinstance(branches, int) and branches >= 3) or has_procurement:
            return "alta"
        if (isinstance(branches, int) and branches >= 1) or len(signals) >= 1:
            return "media"
        if entity.get("registryStatus") == "sin dato" and not signals:
            return "sin dato"
        return "baja"

    if criterion_id == "contactability_and_digital":
        has_web = bool(entity.get("website") and entity.get("website") != "sin dato")
        has_phone = bool(entity.get("phone") and entity.get("phone") != "sin dato")
        has_email = bool(entity.get("email") and entity.get("email") != "sin dato")
        if has_web and (has_phone or has_email):
            return "alta"
        if has_phone or has_email or has_web:
            return "media"
        return "baja"

    return "sin dato"


def evaluate_entity_rubric(
    entity: dict[str, Any],
    entity_rubric: dict[str, Any],
    product_lines: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Scores a candidate entity (0..maxScore) using the configured weighted rubric.

    - Validates criterion weights sum to 1.0.
    - Never hallucinates missing data: unknown criterion values are recorded as ``"sin dato"``
      and receive 0 points.
    - Assigns priorityBand (A/B/C) deterministically from ``priorityBands``.
    """
    criteria: list[dict[str, Any]] = list(entity_rubric.get("criteria") or [])
    _validate_weights(criteria, "entityRubric.criteria")

    max_score = float(entity_rubric.get("maxScore", 100.0))
    priority_bands: list[dict[str, Any]] = sorted(
        list(entity_rubric.get("priorityBands") or []),
        key=lambda b: float(b.get("minScore", 0.0)),
        reverse=True,
    )

    total_score = 0.0
    breakdown: dict[str, dict[str, Any]] = {}
    missing_criteria: list[str] = []

    for crit in criteria:
        crit_id = str(crit["id"])
        weight = float(crit["weight"])
        crit_max_pts = float(crit.get("maxPoints") or (weight * max_score))
        options: list[dict[str, Any]] = list(crit.get("options") or [])

        observed_val = _infer_criterion_value_from_evidence(
            crit_id, entity, product_lines
        )

        matched_opt = next(
            (opt for opt in options if str(opt.get("value")) == observed_val),
            None,
        )

        if matched_opt is None or observed_val == "sin dato":
            awarded_weighted = 0.0
            missing_criteria.append(crit_id)
            breakdown[crit_id] = {
                "value": "sin dato",
                "awardedPoints": 0.0,
                "maxPoints": round(weight * max_score, 2),
                "label": "Sin dato verificado",
            }
        else:
            raw_pts = float(matched_opt.get("points", 0.0))
            ratio = (
                min(1.0, max(0.0, raw_pts / crit_max_pts))
                if crit_max_pts > 0
                else 0.0
            )
            awarded_weighted = round(ratio * weight * max_score, 2)
            total_score += awarded_weighted
            breakdown[crit_id] = {
                "value": observed_val,
                "awardedPoints": awarded_weighted,
                "maxPoints": round(weight * max_score, 2),
                "label": str(matched_opt.get("label") or observed_val),
            }

    final_score = round(min(max_score, max(0.0, total_score)), 2)

    if entity.get("disqualified"):
        final_score = 0.0
        assigned_band = "C"
        band_label = "Descalificado"
    else:
        assigned_band = "C"
        band_label = "Prioridad Baja"
        for band in priority_bands:
            if final_score >= float(band.get("minScore", 0.0)):
                assigned_band = str(band["band"])
                band_label = str(band.get("label") or assigned_band)
                break

    result = dict(entity)
    result["score"] = final_score
    result["priorityBand"] = assigned_band
    result["priorityBandLabel"] = band_label
    result["criterionBreakdown"] = breakdown
    result["missingCriteria"] = missing_criteria
    result["needsLlmRationale"] = (
        assigned_band in ("A", "B")
        and not entity.get("disqualified")
        and len(missing_criteria) > 0
    )
    return result


def evaluate_zone_or_country(
    target: dict[str, Any],
    zone_rubric: dict[str, Any],
) -> dict[str, Any]:
    """Evaluates a geographic zone or target export country and emits GO / EXPLORAR / NO_PRIORIZAR."""
    dimensions: list[dict[str, Any]] = list(zone_rubric.get("dimensions") or [])
    _validate_weights(dimensions, "zoneOrCountryRubric.dimensions")

    max_score = float(zone_rubric.get("maxScore", 100.0))
    thresholds = zone_rubric.get("thresholds") or {"goMin": 70.0, "exploreMin": 45.0}
    go_min = float(thresholds.get("goMin", 70.0))
    explore_min = float(thresholds.get("exploreMin", 45.0))
    if go_min <= explore_min:
        raise ValueError("thresholds.goMin must be strictly greater than thresholds.exploreMin")

    dim_scores_input: dict[str, Any] = dict(target.get("dimensionScores") or {})
    weighted_sum = 0.0
    dimension_breakdown: dict[str, dict[str, Any]] = {}
    missing_dimensions: list[str] = []

    for dim in dimensions:
        dim_id = str(dim["id"])
        weight = float(dim["weight"])
        raw_val = dim_scores_input.get(dim_id)

        if raw_val is None or raw_val == "sin dato" or raw_val == "":
            missing_dimensions.append(dim_id)
            dimension_breakdown[dim_id] = {
                "rawScore": "sin dato",
                "weightedPoints": 0.0,
                "weight": weight,
            }
        else:
            numeric_val = min(max_score, max(0.0, float(raw_val)))
            contrib = round(numeric_val * weight, 2)
            weighted_sum += contrib
            dimension_breakdown[dim_id] = {
                "rawScore": numeric_val,
                "weightedPoints": contrib,
                "weight": weight,
            }

    final_score = round(min(max_score, max(0.0, weighted_sum)), 2)
    if final_score >= go_min:
        decision = "GO"
    elif final_score >= explore_min:
        decision = "EXPLORAR"
    else:
        decision = "NO_PRIORIZAR"

    result = dict(target)
    result["score"] = final_score
    result["decision"] = decision
    result["dimensionBreakdown"] = dimension_breakdown
    result["missingDimensions"] = missing_dimensions
    return result
