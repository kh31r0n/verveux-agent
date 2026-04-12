"""
CartService — authoritative source of truth for the shopping cart.

Design principles:
  - Pure functional: every method takes a cart snapshot and returns a NEW list.
    No mutation in place.  The caller stores the result back into AgentState.
  - The LLM never writes to the cart directly; it only signals intent.
    The node code calls CartService based on that intent.
  - MAX_QTY_PER_ITEM guards against extraction hallucinations (e.g. qty=9999).

Cart item schema (dict):
    {
        "product_id": str,
        "name":       str,
        "qty":        int,   # always >= 1
        "price":      float,
        "notes":      str,   # optional free-text note from user
    }
"""
from __future__ import annotations

from typing import Optional


class CartService:
    MAX_QTY_PER_ITEM: int = 99

    # ------------------------------------------------------------------ #
    # Primitive operations — each returns a brand-new list                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def add_item(
        cart: list[dict],
        product_id: str,
        name: str,
        qty: int,
        price: float,
        notes: str = "",
    ) -> list[dict]:
        """Add *qty* units of *product_id*.  Merges into existing entry if present."""
        cart = list(cart or [])
        for item in cart:
            if item["product_id"] == product_id:
                merged_qty = min(item["qty"] + qty, CartService.MAX_QTY_PER_ITEM)
                return CartService.update_qty(cart, product_id, merged_qty)
        clamped = min(qty, CartService.MAX_QTY_PER_ITEM)
        return cart + [
            {
                "product_id": product_id,
                "name": name,
                "qty": clamped,
                "price": price,
                "notes": notes,
            }
        ]

    @staticmethod
    def remove_item(cart: list[dict], product_id: str) -> list[dict]:
        """Remove *product_id* from the cart entirely."""
        return [item for item in (cart or []) if item["product_id"] != product_id]

    @staticmethod
    def update_qty(cart: list[dict], product_id: str, qty: int) -> list[dict]:
        """Set the exact quantity for *product_id*.  qty <= 0 removes the item."""
        if qty <= 0:
            return CartService.remove_item(cart, product_id)
        clamped = min(qty, CartService.MAX_QTY_PER_ITEM)
        updated = [
            {**item, "qty": clamped} if item["product_id"] == product_id else item
            for item in (cart or [])
        ]
        # If product_id not in cart, insert it (requires name/price — caller must pass them)
        return updated

    @staticmethod
    def replace_item(
        cart: list[dict],
        old_product_id: str,
        new_product_id: str,
        new_name: str,
        new_price: float,
    ) -> list[dict]:
        """Swap *old_product_id* for a different product, keeping the same qty."""
        return [
            {**item, "product_id": new_product_id, "name": new_name, "price": new_price}
            if item["product_id"] == old_product_id
            else item
            for item in (cart or [])
        ]

    # ------------------------------------------------------------------ #
    # Operation dispatcher                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def apply_operation(
        cart: list[dict],
        operation: str,
        product_id: str,
        name: str,
        qty: int,
        price: float,
        old_product_id: Optional[str] = None,
        notes: str = "",
    ) -> list[dict]:
        """
        Dispatch a named operation extracted from the LLM output.

        Supported operations:
            add              — add qty units (merges if already present)
            remove           — remove item entirely
            update_quantity  — set exact qty
            replace          — swap old_product_id for new product, same qty

        Unknown operations fall back to "add" so the cart never silently breaks.
        """
        op = (operation or "add").strip().lower()

        if op == "add":
            return CartService.add_item(cart, product_id, name, qty, price, notes)
        if op == "remove":
            return CartService.remove_item(cart, product_id)
        if op == "update_quantity":
            return CartService.update_qty(cart, product_id, qty)
        if op == "replace" and old_product_id:
            return CartService.replace_item(cart, old_product_id, product_id, name, price)
        # Fallback
        return CartService.add_item(cart, product_id, name, qty, price, notes)

    # ------------------------------------------------------------------ #
    # Queries / presentation                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def total(cart: list[dict]) -> float:
        return sum(item["qty"] * item["price"] for item in (cart or []))

    @staticmethod
    def is_empty(cart: list[dict]) -> bool:
        return not bool(cart)

    @staticmethod
    def find_item(cart: list[dict], product_id: str) -> Optional[dict]:
        return next(
            (item for item in (cart or []) if item["product_id"] == product_id), None
        )

    @staticmethod
    def format_cart(cart: list[dict], currency: str = "$") -> str:
        """Return a WhatsApp-safe Markdown cart summary."""
        if not cart:
            return "🛒 El carrito está vacío."

        lines = ["🛒 *Tu carrito:*"]
        for i, item in enumerate(cart, 1):
            subtotal = item["qty"] * item["price"]
            notes_part = f" _{item['notes']}_" if item.get("notes") else ""
            lines.append(
                f"{i}. {item['name']} × {item['qty']} — "
                f"{currency}{item['price']:.2f} c/u = *{currency}{subtotal:.2f}*"
                f"{notes_part}"
            )
        lines.append(f"\n💰 *Total estimado: {currency}{CartService.total(cart):.2f}*")
        return "\n".join(lines)
