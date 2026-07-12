"""Shared backend cart synchronisation — used by sales_collect and
restaurant_order_collect. The in-memory state cart stays the source of
truth; the backend cart is a best-effort mirror consumed at checkout."""

import structlog
from httpx import HTTPStatusError

from .backend_client import upsert_cart_item

logger = structlog.get_logger(__name__)

_BACKEND_SYNC_RETRIES = 2


async def sync_full_cart_to_backend(
    cart: list,
    contact_id: str,
    conversation_id: str | None,
    thread_id: str,
) -> bool:
    """
    Sync the entire in-memory cart to the backend in one pass.

    Upserts each item's final quantity. Returns True when all upserts
    succeed, False if any fail after retries. Failures are logged but
    never raise — the in-memory cart remains the source of truth and
    order_summary falls back to it when the backend is empty.
    """
    if not contact_id or not cart:
        return True  # nothing to sync

    all_ok = True
    for item in cart:
        product_id = item["product_id"]
        qty = item["qty"]
        succeeded = False

        for attempt in range(_BACKEND_SYNC_RETRIES):
            try:
                await upsert_cart_item(
                    contact_id,
                    product_id,
                    qty,
                    conversation_id,
                    notes=item.get("notes") or None,
                )
                succeeded = True
                break
            except HTTPStatusError as exc:
                # 4xx are deterministic (validation, not-found, out of stock) —
                # retrying just repeats the same rejection, so give up now.
                status = exc.response.status_code
                if 400 <= status < 500:
                    logger.error(
                        "backend_cart_upsert_client_error",
                        thread_id=thread_id,
                        product_id=product_id,
                        qty=qty,
                        status=status,
                        error=str(exc),
                    )
                    break
                logger.warning(
                    "backend_cart_upsert_retry",
                    thread_id=thread_id,
                    product_id=product_id,
                    qty=qty,
                    attempt=attempt + 1,
                    max_attempts=_BACKEND_SYNC_RETRIES,
                    error=str(exc),
                )
            except Exception as exc:
                logger.warning(
                    "backend_cart_upsert_retry",
                    thread_id=thread_id,
                    product_id=product_id,
                    qty=qty,
                    attempt=attempt + 1,
                    max_attempts=_BACKEND_SYNC_RETRIES,
                    error=str(exc),
                )

        if not succeeded:
            logger.error(
                "backend_cart_upsert_failed",
                thread_id=thread_id,
                product_id=product_id,
                qty=qty,
            )
            all_ok = False

    if all_ok:
        logger.info(
            "backend_cart_sync_ok",
            thread_id=thread_id,
            cart_size=len(cart),
        )
    else:
        logger.error(
            "backend_cart_sync_partial_failure",
            thread_id=thread_id,
            cart_size=len(cart),
        )

    return all_ok
