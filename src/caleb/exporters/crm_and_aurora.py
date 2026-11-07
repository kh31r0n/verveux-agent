"""CRM prospect export payload builder (with batch + CRM deduplication) and reversible Aurora proposal builder."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

BAND_ORDER = {"A": 3, "B": 2, "C": 1}


def _normalize_company_key(name: str) -> str:
    lowered = re.sub(r"[^a-z0-9\s]", " ", (name or "").lower())
    # Strip corporate suffixes
    lowered = re.sub(r"\b(s\s*a\s*s|s\s*a|ltda|inc|llc|corp|s\s*r\s*l)\b", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _extract_domain(url: str) -> str:
    if not url or url == "sin dato":
        return ""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def build_crm_export_payload(
    state: dict[str, Any],
    existing_crm_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Builds the deduplicated CRM export payload for Vervux One (`origin = AI_PROSPECTING`, locked until human conversion)."""
    cfg = state.get("config") or {}
    study_run_id = str(state.get("study_run_id") or "sin dato")
    export_cfg = ((cfg.get("deliverables") or {}).get("exportToOneCrm")) or {}
    if not export_cfg.get("enabled", False):
        return {
            "enabled": False,
            "studyRunId": study_run_id,
            "candidatesEvaluated": 0,
            "exportedCount": 0,
            "skippedDuplicatesCount": 0,
            "skippedUnverifiedCount": 0,
            "prospects": [],
        }

    min_band = str(export_cfg.get("minPriorityBand") or "A")
    min_rank = BAND_ORDER.get(min_band, 3)
    seen_keys: set[str] = set(existing_crm_keys or set())

    scored = list(state.get("scored_entities") or [])
    prospects: list[dict[str, Any]] = []
    skipped_duplicates = 0
    skipped_unverified = 0

    for ent in scored:
        if ent.get("isNaturalPerson"):
            continue
        band = str(ent.get("priorityBand") or "C")
        if BAND_ORDER.get(band, 0) < min_rank:
            continue

        website = str(ent.get("website") or "sin dato").strip()
        tax_id = str(ent.get("taxId") or "sin dato").strip()
        email = str(ent.get("email") or "sin dato").strip()
        # Google Places phone numbers must never be exported to CRM (§10)
        raw_phone = str(ent.get("phone") or "sin dato").strip()
        phone = (
            "sin dato"
            if ent.get("source") == "GOOGLE_PLACES" or ent.get("phoneSource") == "GOOGLE_PLACES"
            else raw_phone
        )

        # Safeguard: allowUnverifiedContacts is locked to False
        has_verified_anchor = (
            website not in ("", "sin dato")
            or tax_id not in ("", "sin dato", "reservado-ley-1581")
            or email not in ("", "sin dato")
        )
        if not has_verified_anchor:
            skipped_unverified += 1
            continue

        dedup_candidates = [
            f"tax:{tax_id.lower()}" if tax_id not in ("", "sin dato") else "",
            f"dom:{_extract_domain(website)}" if _extract_domain(website) else "",
            f"name:{_normalize_company_key(str(ent.get('name') or ''))}",
        ]
        dedup_candidates = [k for k in dedup_candidates if k and not k.endswith(":")]

        if any(k in seen_keys for k in dedup_candidates):
            skipped_duplicates += 1
            continue
        for k in dedup_candidates:
            seen_keys.add(k)

        source_url = website if website not in ("", "sin dato") else f"caleb://study/{study_run_id}"
        prospects.append(
            {
                "entityId": ent.get("entityId", "sin dato"),
                "companyName": ent.get("name", "sin dato"),
                "taxId": tax_id,
                "country": ent.get("country", "CO"),
                "city": ent.get("city", "sin dato"),
                "website": website,
                "phone": phone,
                "email": email,
                "origin": "AI_PROSPECTING",
                "sourceUrl": source_url,
                "convertedAt": None,
                "outboundBlockedUntilConverted": True,
                "tags": list(export_cfg.get("tagNames") or ["caleb-market-study"]),
                "profileData": {
                    "caleb": {
                        "studyRunId": study_run_id,
                        "studyTitle": cfg.get("studyTitle", "Estudio de Mercado"),
                        "segmentCode": ent.get("segmentCode", "sin dato"),
                        "deterministicScore": ent.get("score", 0.0),
                        "priorityBand": band,
                        "competitorRole": ent.get("competitorRole", "NONE"),
                        "estimatedScale": ent.get("estimatedScale", "sin dato"),
                        "evidenceSignals": list(ent.get("signals") or []),
                        "source": ent.get("source", "STUDY_PIPELINE"),
                    }
                },
            }
        )

    return {
        "enabled": True,
        "studyRunId": study_run_id,
        "minPriorityBand": min_band,
        "candidatesEvaluated": len(scored),
        "exportedCount": len(prospects),
        "skippedDuplicatesCount": skipped_duplicates,
        "skippedUnverifiedCount": skipped_unverified,
        "prospects": prospects,
    }


def build_aurora_proposals(
    state: dict[str, Any],
    previous_aurora_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds single-country Aurora proposals (`prospectingNiche` + `prospectingLocation`) with daily spend impact and reversibility (§8)."""
    cfg = state.get("config") or {}
    bp = cfg.get("businessProfile") or {}
    taxonomy = cfg.get("taxonomy") or {}
    geography = cfg.get("geography") or {}

    search_terms: list[str] = []
    for pl in bp.get("productLines") or []:
        for kw in pl.get("keywords") or []:
            if kw not in search_terms:
                search_terms.append(kw)
    for seg in taxonomy.get("segments") or []:
        for kw in seg.get("keywords") or []:
            if kw not in search_terms:
                search_terms.append(kw)

    niche_key = re.sub(r"[^a-z0-9]+", "_", (bp.get("industry") or "b2b_industrial").lower()).strip("_")
    prospecting_niche = {
        "key": niche_key or "b2b_industrial",
        "label": bp.get("industry") or "Prospección Industrial B2B",
        "search_terms": search_terms[:12] or ["distribuidor industrial"],
    }

    # Group best cities by country (Aurora requires a single country per configuration)
    country_cities: dict[str, list[str]] = {}
    for z in geography.get("localZones") or []:
        c_code = str(z.get("country") or "CO").upper()
        country_cities.setdefault(c_code, [])
        for city in z.get("cities") or []:
            if city not in country_cities[c_code]:
                country_cities[c_code].append(city)

    for tc in geography.get("targetCountries") or []:
        c_code = str(tc.get("countryCode") or "").upper()
        if not c_code:
            continue
        country_cities.setdefault(c_code, [])
        for city in tc.get("priorityCities") or []:
            if city not in country_cities[c_code]:
                country_cities[c_code].append(city)

    if not country_cities:
        country_cities["CO"] = ["Bogotá"]

    options_by_country: list[dict[str, Any]] = []
    for country_code, cities in country_cities.items():
        selected_cities = cities[:6] or ["Capital"]
        hl = "pt" if country_code == "BR" else ("en" if country_code in ("BZ", "GY") else "es")
        est_queries_per_day = len(prospecting_niche["search_terms"]) * len(selected_cities)
        est_daily_spend_usd = round(est_queries_per_day * 0.015, 2)
        options_by_country.append(
            {
                "country": country_code,
                "prospectingNiche": prospecting_niche,
                "prospectingLocation": {
                    "country": country_code,
                    "gl": country_code.lower(),
                    "hl": hl,
                    "cities": selected_cities,
                },
                "estimatedDailyQueries": est_queries_per_day,
                "estimatedDailySpendUsd": est_daily_spend_usd,
            }
        )

    return {
        "requiresExplicitHumanAcceptance": True,
        "isMultiCountryStudy": len(options_by_country) > 1,
        "optionsByCountry": options_by_country,
        "previousConfigSnapshot": previous_aurora_config or {"prospectingNiche": None, "prospectingLocation": None},
    }
