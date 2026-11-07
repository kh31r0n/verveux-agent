"""Parameterized segment classifier and entity deduplicator for Caleb (MARKET_STUDY)."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _extract_domain(url_or_domain: str | None) -> str:
    if not url_or_domain or url_or_domain == "sin dato":
        return ""
    raw = url_or_domain.strip().lower()
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    host = parsed.netloc or parsed.path
    host = re.sub(r"^www\.", "", host).split("/")[0].strip()
    return host


def _build_entity_corpus(entity: dict[str, Any]) -> str:
    parts: list[str] = [
        str(entity.get("name") or ""),
        str(entity.get("description") or ""),
        str(entity.get("websiteText") or ""),
        str(entity.get("registryActivity") or ""),
    ]
    for list_key in ("signals", "keywords", "categories", "brands"):
        val = entity.get(list_key)
        if isinstance(val, list):
            parts.extend(str(item) for item in val if item and item != "sin dato")
    return _normalize_text(" ".join(parts))


def classify_entity_segment(
    entity: dict[str, Any],
    taxonomy: dict[str, Any],
    disqualifiers: list[str] | None = None,
) -> dict[str, Any]:
    """Classifies an entity into a configured segment using positive/negative signals and weights.

    - Checks ``disqualifiers`` first.
    - Scores each segment in ``taxonomy['segments']`` deterministically.
    - Tie-breaks by higher ``priorityWeight``, then by declaration order.
    - Falls back to ``taxonomy['defaultSegmentCode']`` when no positive score is achieved.
    """
    corpus = _build_entity_corpus(entity)
    result = dict(entity)

    # 1. Check disqualifiers
    for disq in disqualifiers or []:
        norm_disq = _normalize_text(disq)
        if norm_disq and norm_disq in corpus:
            result["disqualified"] = True
            result["disqualifierMatched"] = disq
            result["segmentCode"] = "DISQUALIFIED"
            result["segmentName"] = "Descalificado por criterio de exclusión"
            result["segmentConfidence"] = "ALTA"
            result["segmentScore"] = -999.0
            result["matchedSignals"] = [f"disqualifier:{disq}"]
            return result

    segments: list[dict[str, Any]] = list(taxonomy.get("segments") or [])
    default_code: str = str(taxonomy.get("defaultSegmentCode") or "")
    if not segments:
        raise ValueError("taxonomy.segments must contain at least one segment")

    best_seg: dict[str, Any] | None = None
    best_score: float = 0.0
    best_weight: float = -1.0
    best_matches: list[str] = []

    segment_scores: dict[str, float] = {}

    for seg in segments:
        code = str(seg.get("code") or "")
        weight = float(seg.get("priorityWeight", 1.0))
        pos_signals = seg.get("positiveSignals") or []
        neg_signals = seg.get("negativeSignals") or []
        keywords = seg.get("keywords") or []

        raw_score = 0.0
        matched: list[str] = []

        for sig in pos_signals:
            norm_sig = _normalize_text(sig)
            if norm_sig and norm_sig in corpus:
                raw_score += 2.0 * weight
                matched.append(f"+signal:{sig}")

        for kw in keywords:
            norm_kw = _normalize_text(kw)
            if norm_kw and norm_kw in corpus:
                raw_score += 1.0 * weight
                matched.append(f"+kw:{kw}")

        for neg in neg_signals:
            norm_neg = _normalize_text(neg)
            if norm_neg and norm_neg in corpus:
                raw_score -= 2.5
                matched.append(f"-neg:{neg}")

        rounded_score = round(raw_score, 4)
        segment_scores[code] = rounded_score

        if rounded_score > best_score or (
            rounded_score > 0.0
            and rounded_score == best_score
            and weight > best_weight
        ):
            best_score = rounded_score
            best_weight = weight
            best_seg = seg
            best_matches = matched

    result["disqualified"] = False
    result["segmentScores"] = segment_scores

    if best_seg is not None and best_score > 0.0:
        result["segmentCode"] = best_seg["code"]
        result["segmentName"] = best_seg.get("name", best_seg["code"])
        result["segmentScore"] = best_score
        result["matchedSignals"] = best_matches
        result["segmentConfidence"] = "ALTA" if best_score >= 3.0 else "MEDIA"
        return result

    # Fallback to defaultSegmentCode
    fallback_seg = next(
        (s for s in segments if s.get("code") == default_code),
        segments[0],
    )
    result["segmentCode"] = fallback_seg["code"]
    result["segmentName"] = fallback_seg.get("name", fallback_seg["code"])
    result["segmentScore"] = 0.0
    result["matchedSignals"] = []
    result["segmentConfidence"] = "BAJA"
    return result


def deduplicate_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicates candidate entities by taxId/NIT, domain, or normalized (name, city).

    Merges verified fields so that ``"sin dato"`` or empty strings never overwrite
    a verified value from another source, and accumulates all ``sources`` and ``signals``.
    """
    merged_by_key: dict[str, dict[str, Any]] = {}
    key_aliases: dict[str, str] = {}

    for raw in entities:
        name = _normalize_text(str(raw.get("name") or ""))
        if not name or name == "sin dato":
            continue

        tax_id = re.sub(r"[^0-9a-z]", "", _normalize_text(str(raw.get("taxId") or "")))
        if tax_id == "sindato":
            tax_id = ""
        domain = _extract_domain(raw.get("website"))
        city = _normalize_text(str(raw.get("city") or ""))
        if city == "sin dato":
            city = ""

        candidate_keys: list[str] = []
        if tax_id:
            candidate_keys.append(f"tax:{tax_id}")
        if domain:
            candidate_keys.append(f"dom:{domain}")
        candidate_keys.append(f"name:{name}|city:{city}")

        canonical_key: str | None = None
        for ck in candidate_keys:
            if ck in key_aliases:
                canonical_key = key_aliases[ck]
                break

        if canonical_key is None:
            canonical_key = candidate_keys[0]
            base = dict(raw)
            base.setdefault("phone", "sin dato")
            base.setdefault("email", "sin dato")
            base.setdefault("website", "sin dato")
            base.setdefault("taxId", "sin dato")
            src = raw.get("source")
            sources_list = list(raw.get("sources") or ([] if not src else [src]))
            base["sources"] = list(dict.fromkeys(s for s in sources_list if s))
            base["signals"] = list(dict.fromkeys(raw.get("signals") or []))
            merged_by_key[canonical_key] = base
        else:
            existing = merged_by_key[canonical_key]
            for field_name, val in raw.items():
                if field_name in ("sources", "signals"):
                    continue
                if val is not None and val != "" and val != "sin dato":
                    if existing.get(field_name) in (None, "", "sin dato"):
                        existing[field_name] = val
            src = raw.get("source")
            new_sources = list(raw.get("sources") or ([] if not src else [src]))
            for s in new_sources:
                if s and s not in existing["sources"]:
                    existing["sources"].append(s)
            for sig in raw.get("signals") or []:
                if sig and sig not in existing["signals"]:
                    existing["signals"].append(sig)

        for ck in candidate_keys:
            key_aliases[ck] = canonical_key

    return list(merged_by_key.values())
