"""Web services for StockAI.

Service layer for database operations and business logic.
"""

from stockai.web.services.watchlist import (
    WatchlistItemExistsError,
    WatchlistItemNotFoundError,
    StockNotFoundError,
    get_watchlist_items,
    get_watchlist_item_by_id,
    get_watchlist_item_by_stock_id,
    get_watchlist_item_by_symbol,
    add_to_watchlist,
    update_watchlist_item,
    remove_from_watchlist,
    remove_from_watchlist_by_symbol,
    get_or_create_stock_by_symbol,
)

__all__ = [
    # Exceptions
    "WatchlistItemExistsError",
    "WatchlistItemNotFoundError",
    "StockNotFoundError",
    # Functions
    "get_watchlist_items",
    "get_watchlist_item_by_id",
    "get_watchlist_item_by_stock_id",
    "get_watchlist_item_by_symbol",
    "add_to_watchlist",
    "update_watchlist_item",
    "remove_from_watchlist",
    "remove_from_watchlist_by_symbol",
    "get_or_create_stock_by_symbol",
]
