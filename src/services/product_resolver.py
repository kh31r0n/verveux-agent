"""
ProductResolver — converts free-text product mentions to catalog product_ids.

Resolution pipeline (short-circuits on first hit):
  1. Exact match (case-insensitive, stripped)
  2. Substring match (query ⊂ catalog name OR catalog name ⊂ query)
  3. Fuzzy match via difflib (cutoff = FUZZY_THRESHOLD)
  4. LLM fallback via AsyncOpenAI — only reached when all heuristics fail,
     keeping API calls minimal.

Usage in a node:
    resolver = ProductResolver(state["product_catalog"])
    resolved, unresolved = await resolver.resolve_many(extracted_items, client)
"""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

FUZZY_THRESHOLD = 0.50      # minimum difflib similarity score
FUZZY_CANDIDATES = 3        # alternatives to surface when no exact hit

_LLM_PROMPT = """You are a product catalog matcher.  Given a user's description and a JSON catalog, find the best match.

Catalog:
{catalog_json}

User said: "{query}"

Return ONLY a JSON object — no markdown, no extra text.
- Match found:  {{"product_id": "<id>", "confidence": "high" | "medium"}}
- No match:     {{"product_id": null,  "confidence": "none"}}
"""


# ------------------------------------------------------------------ #
# Data classes                                                         #
# ------------------------------------------------------------------ #


@dataclass
class ResolvedProduct:
    product_id: str
    name: str
    price: float
    stock: int
    exact_match: bool


@dataclass
class ResolutionResult:
    resolved: Optional[ResolvedProduct]
    alternatives: list[dict] = field(default_factory=list)
    used_llm: bool = False


# ------------------------------------------------------------------ #
# Resolver                                                             #
# ------------------------------------------------------------------ #


class ProductResolver:
    def __init__(self, catalog: list[dict]) -> None:
        self.catalog: list[dict] = catalog or []
        # Pre-build lower-cased name index for O(1) exact lookup
        self._by_name: dict[str, dict] = {
            p.get("name", "").lower().strip(): p for p in self.catalog
        }

    # ---------------------------------------------------------------- #
    # Synchronous resolution (heuristics only, no I/O)                  #
    # ---------------------------------------------------------------- #

    def resolve(self, name: str) -> ResolutionResult:
        if not name or not self.catalog:
            return ResolutionResult(resolved=None)

        needle = name.lower().strip()

        # 1. Exact match
        if needle in self._by_name:
            return ResolutionResult(
                resolved=self._make(self._by_name[needle], exact_match=True),
            )

        # 2. Substring match — catalog name contains query or vice-versa
        substring_hits = [
            p for p in self.catalog
            if needle in p.get("name", "").lower() or p.get("name", "").lower() in needle
        ]
        if substring_hits:
            best, *rest = substring_hits
            return ResolutionResult(
                resolved=self._make(best, exact_match=False),
                alternatives=self._as_alts(rest[:FUZZY_CANDIDATES - 1]),
            )

        # 3. Fuzzy match
        catalog_names = [p.get("name", "") for p in self.catalog]
        close = difflib.get_close_matches(name, catalog_names, n=FUZZY_CANDIDATES, cutoff=FUZZY_THRESHOLD)
        if close:
            best_name = close[0]
            best_p = next(p for p in self.catalog if p.get("name", "") == best_name)
            rest_ps = [
                next(p for p in self.catalog if p.get("name", "") == n)
                for n in close[1:]
            ]
            return ResolutionResult(
                resolved=self._make(best_p, exact_match=False),
                alternatives=self._as_alts(rest_ps),
            )

        return ResolutionResult(resolved=None)

    # ---------------------------------------------------------------- #
    # Async resolution with LLM fallback                                 #
    # ---------------------------------------------------------------- #

    async def resolve_with_llm_fallback(
        self,
        name: str,
        client,               # AsyncOpenAI — typed loosely to avoid import cycle
        model: str = "gpt-5.4-nano",
    ) -> ResolutionResult:
        result = self.resolve(name)
        if result.resolved is not None:
            return result

        # Heuristics failed — ask the LLM
        catalog_json = json.dumps(
            [
                {
                    "product_id": p["product_id"],
                    "name": p.get("name", ""),
                    "description": p.get("description", ""),
                }
                for p in self.catalog
            ],
            ensure_ascii=False,
        )
        prompt = _LLM_PROMPT.format(catalog_json=catalog_json, query=name)
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = json.loads(raw)
            pid: Optional[str] = parsed.get("product_id")
            if pid:
                p = next((x for x in self.catalog if x["product_id"] == pid), None)
                if p:
                    logger.info("product_resolver_llm_match", query=name, product_id=pid)
                    return ResolutionResult(
                        resolved=self._make(p, exact_match=False),
                        used_llm=True,
                    )
        except Exception as exc:
            logger.warning("product_resolver_llm_failed", query=name, error=str(exc))

        return ResolutionResult(resolved=None, used_llm=True)

    # ---------------------------------------------------------------- #
    # Batch helper                                                       #
    # ---------------------------------------------------------------- #

    async def resolve_many(
        self,
        items: list[dict],
        client=None,
        model: str = "gpt-5.4-nano",
    ) -> tuple[list[dict], list[dict]]:
        """
        Resolve a batch of items extracted by the LLM.

        Args:
            items:  list of extracted dicts with keys:
                      name, quantity, operation, old_product_id (optional), notes (optional)
            client: AsyncOpenAI client; if None, LLM fallback is disabled.

        Returns:
            (resolved, unresolved)

            resolved items schema:
                {product_id, name, qty, price, stock, operation,
                 old_product_id, notes, exact_match, alternatives}

            unresolved items schema:
                {name, qty, alternatives}
        """
        resolved: list[dict] = []
        unresolved: list[dict] = []

        for item in items:
            query_name = item.get("name", "").strip()
            if not query_name:
                continue

            if client:
                result = await self.resolve_with_llm_fallback(query_name, client, model)
            else:
                result = self.resolve(query_name)

            if result.resolved:
                resolved.append(
                    {
                        "product_id": result.resolved.product_id,
                        "name": result.resolved.name,
                        "qty": max(1, int(item.get("quantity", 1))),
                        "price": result.resolved.price,
                        "stock": result.resolved.stock,
                        "operation": item.get("operation", "add"),
                        "old_product_id": item.get("old_product_id"),
                        "notes": item.get("notes", ""),
                        "exact_match": result.resolved.exact_match,
                        "alternatives": result.alternatives,
                    }
                )
            else:
                unresolved.append(
                    {
                        "name": query_name,
                        "qty": max(1, int(item.get("quantity", 1))),
                        "alternatives": result.alternatives,
                    }
                )

        return resolved, unresolved

    # ---------------------------------------------------------------- #
    # Private helpers                                                    #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _make(p: dict, *, exact_match: bool) -> ResolvedProduct:
        return ResolvedProduct(
            product_id=p["product_id"],
            name=p.get("name", ""),
            price=float(p.get("price", 0)),
            stock=int(p.get("stock", 0)),
            exact_match=exact_match,
        )

    @staticmethod
    def _as_alts(products: list[dict]) -> list[dict]:
        return [
            {
                "product_id": p["product_id"],
                "name": p.get("name", ""),
                "price": float(p.get("price", 0)),
            }
            for p in products
        ]
