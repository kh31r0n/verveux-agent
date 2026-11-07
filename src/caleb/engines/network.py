"""Network, geographic zone coverage, whitespace gaps, and distribution route analyzer."""

from __future__ import annotations

from typing import Any

from .evaluator import evaluate_zone_or_country


def analyze_network_and_coverage(
    entities: list[dict[str, Any]],
    local_zones: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    zone_rubric: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyzes zone-level entity density, segment coverage gaps, and hub-to-spoke distribution routes."""
    zone_summaries: list[dict[str, Any]] = []
    coverage_gaps: list[dict[str, Any]] = []
    segment_codes = [str(s["code"]) for s in segments if s.get("code")]

    active_entities = [e for e in entities if not e.get("disqualified")]

    for zone in local_zones:
        z_code = str(zone["code"])
        z_name = str(zone.get("name") or z_code)
        z_cities = {str(c).strip().lower() for c in (zone.get("cities") or [])}

        z_entities = [
            e
            for e in active_entities
            if e.get("zoneCode") == z_code
            or str(e.get("city") or "").strip().lower() in z_cities
        ]

        by_segment: dict[str, int] = {sc: 0 for sc in segment_codes}
        by_band: dict[str, int] = {"A": 0, "B": 0, "C": 0}
        tier1_importers = 0
        verified_contacts = 0

        for ent in z_entities:
            seg = str(ent.get("segmentCode") or "")
            if seg in by_segment:
                by_segment[seg] += 1
            band = str(ent.get("priorityBand") or "C")
            if band in by_band:
                by_band[band] += 1
            if ent.get("competitiveLevel") == "NIVEL_1_IMPORTADOR_MAYORISTA":
                tier1_importers += 1
            if (ent.get("phone") and ent.get("phone") != "sin dato") or (
                ent.get("email") and ent.get("email") != "sin dato"
            ):
                verified_contacts += 1

        for sc in segment_codes:
            count = by_segment.get(sc, 0)
            if count == 0:
                coverage_gaps.append(
                    {
                        "zoneCode": z_code,
                        "zoneName": z_name,
                        "segmentCode": sc,
                        "gapSeverity": "VACIO_TOTAL",
                        "entityCount": 0,
                        "recommendation": f"Ampliar búsqueda o cobertura comercial para {sc} en {z_name}.",
                    }
                )
            elif count == 1:
                coverage_gaps.append(
                    {
                        "zoneCode": z_code,
                        "zoneName": z_name,
                        "segmentCode": sc,
                        "gapSeverity": "COBERTURA_BAJA",
                        "entityCount": 1,
                        "recommendation": f"Profundizar validación de cuentas {sc} en {z_name}.",
                    }
                )

        # Derive deterministic dimension scores when not explicitly supplied on the zone
        explicit_dims = zone.get("dimensionScores")
        if isinstance(explicit_dims, dict) and explicit_dims:
            dim_scores = dict(explicit_dims)
        else:
            total_n = len(z_entities)
            demand_score = min(100.0, 45.0 + total_n * 15.0 + by_band["A"] * 10.0) if total_n > 0 else 25.0
            whitespace_score = max(30.0, 85.0 - tier1_importers * 15.0) if total_n > 0 else 50.0
            logistics_score = 80.0 if len(zone.get("industrialCorridors") or []) > 0 else 65.0
            dim_scores = {
                "demand_density": demand_score,
                "competitive_whitespace": whitespace_score,
                "logistics_accessibility": logistics_score,
            }

        zone_target = {
            "code": z_code,
            "name": z_name,
            "kind": "ZONE",
            "country": zone.get("country", "CO"),
            "entityCount": len(z_entities),
            "bySegment": by_segment,
            "byPriorityBand": by_band,
            "tier1ImportersCount": tier1_importers,
            "verifiedContactRate": (
                round(verified_contacts / len(z_entities), 2) if z_entities else 0.0
            ),
            "dimensionScores": dim_scores,
        }

        if zone_rubric and zone_rubric.get("dimensions"):
            evaluated = evaluate_zone_or_country(zone_target, zone_rubric)
        else:
            avg_score = round(sum(dim_scores.values()) / max(1, len(dim_scores)), 2)
            evaluated = {
                **zone_target,
                "score": avg_score,
                "decision": "GO" if avg_score >= 70 else ("EXPLORAR" if avg_score >= 45 else "NO_PRIORIZAR"),
            }

        zone_summaries.append(evaluated)

    zone_summaries.sort(key=lambda z: float(z.get("score", 0.0)), reverse=True)

    # Identify hub-to-spoke distribution routes
    hubs = [
        z
        for z in zone_summaries
        if z.get("tier1ImportersCount", 0) > 0 or z.get("decision") == "GO"
    ]
    spokes = [z for z in zone_summaries if z not in hubs]

    distribution_routes: list[dict[str, Any]] = []
    if hubs:
        primary_hub = hubs[0]
        for spoke in spokes:
            distribution_routes.append(
                {
                    "hubZoneCode": primary_hub["code"],
                    "hubZoneName": primary_hub["name"],
                    "targetZoneCode": spoke["code"],
                    "targetZoneName": spoke["name"],
                    "strategy": "Atención desde hub mayorista mediante distribuidor regional o ruta técnica programada",
                }
            )

    return {
        "zones": zone_summaries,
        "coverageGaps": coverage_gaps,
        "distributionRoutes": distribution_routes,
    }
