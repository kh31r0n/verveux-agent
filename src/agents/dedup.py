"""Normalization utilities for prospecting dedupe.

The synthetic contact identity is derived purely from the institution's name so
that the same school found via two different pages (or on two different days)
collapses to one CRM contact. Domains are normalized with ``tldextract`` so
``www.colegio.edu.co`` and ``colegio.edu.co/inicio`` compare equal.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import tldextract

# tldextract normally fetches (and caches) the public-suffix list over the
# network on first use. In our container that call may be blocked, so pin it to
# the bundled snapshot — no network, deterministic in tests.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")


def normalize_name(raw: str | None) -> str:
    """Fold accents/case/punctuation so name variants compare equal.

    "Colegio San José" and "colegio san jose!!" both normalize to
    "colegio san jose".
    """
    if not raw:
        return ""
    # Decompose accents (NFKD) then drop the combining marks.
    decomposed = unicodedata.normalize("NFKD", raw)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = stripped.casefold()
    cleaned = _NON_ALNUM_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", cleaned).strip()


def prospect_external_id(raw_name: str | None) -> str:
    """Stable synthetic identity for a prospect: ``prospector:<sha256>``.

    Derived from the normalized name only, so the identity is idempotent across
    runs and pages. Returns "" for an empty name (caller must skip it).
    """
    normalized = normalize_name(raw_name)
    if not normalized:
        return ""
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"prospector:{digest}"


def normalize_domain(raw: str | None) -> str:
    """Return the lowercased registrable domain (eTLD+1) of a URL/host, or ""."""
    if not raw:
        return ""
    ext = _EXTRACT(raw.strip().lower())
    if not ext.domain:
        return ""
    if ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ext.domain
