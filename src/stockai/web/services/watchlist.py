"""Watchlist Service for StockAI.

Database operations for watchlist CRUD functionality.
"""

from decimal import Decimal
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from stockai.data.database import session_scope
from stockai.data.models import Stock, WatchlistItem


class WatchlistItemExistsError(Exception):
    """Raised when trying to add a stock that's already in the watchlist."""

    def __init__(self, stock_id: int, symbol: str):
        self.stock_id = stock_id
        self.symbol = symbol
        super().__init__(f"Stock {symbol} (id={stock_id}) is already in the watchlist")


class WatchlistItemNotFoundError(Exception):
    """Raised when a watchlist item is not found."""

    def __init__(self, identifier: int | str, by_field: str = "id"):
        self.identifier = identifier
        self.by_field = by_field
        super().__init__(f"Watchlist item with {by_field}={identifier} not found")


class StockNotFoundError(Exception):
    """Raised when a stock is not found."""

    def __init__(self, identifier: int | str, by_field: str = "symbol"):
        self.identifier = identifier
        self.by_field = by_field
        super().__init__(f"Stock with {by_field}={identifier} not found")


def get_watchlist_items() -> list[WatchlistItem]:
    """Get all watchlist items with associated stock information.

    Returns:
        List of WatchlistItem objects with stock relationship loaded.
    """
    with session_scope() as session:
        items = (
            session.query(WatchlistItem)
            .options(joinedload(WatchlistItem.stock))
            .order_by(WatchlistItem.created_at.desc())
            .all()
        )
        # Detach from session while keeping loaded relationships
        for item in items:
            # Must expunge both item and its stock to fully detach
            if item.stock:
                session.expunge(item.stock)
            session.expunge(item)
        return items


def get_watchlist_item_by_id(item_id: int) -> Optional[WatchlistItem]:
    """Get a single watchlist item by its ID.

    Args:
        item_id: The watchlist item ID.

    Returns:
        WatchlistItem if found, None otherwise.
    """
    with session_scope() as session:
        item = (
            session.query(WatchlistItem)
            .options(joinedload(WatchlistItem.stock))
            .filter(WatchlistItem.id == item_id)
            .first()
        )
        if item:
            # Must expunge both item and its stock to fully detach
            if item.stock:
                session.expunge(item.stock)
            session.expunge(item)
        return item


def get_watchlist_item_by_stock_id(stock_id: int) -> Optional[WatchlistItem]:
    """Get a watchlist item by its stock ID.

    Args:
        stock_id: The stock ID to look for.

    Returns:
        WatchlistItem if found, None otherwise.
    """
    with session_scope() as session:
        item = (
            session.query(WatchlistItem)
            .options(joinedload(WatchlistItem.stock))
            .filter(WatchlistItem.stock_id == stock_id)
            .first()
        )
        if item:
            # Must expunge both item and its stock to fully detach
            if item.stock:
                session.expunge(item.stock)
            session.expunge(item)
        return item


def get_watchlist_item_by_symbol(symbol: str) -> Optional[WatchlistItem]:
    """Get a watchlist item by stock symbol.

    Args:
        symbol: The stock symbol (case-insensitive).

    Returns:
        WatchlistItem if found, None otherwise.
    """
    symbol = symbol.upper().strip()
    with session_scope() as session:
        item = (
            session.query(WatchlistItem)
            .options(joinedload(WatchlistItem.stock))
            .join(Stock)
            .filter(Stock.symbol == symbol)
            .first()
        )
        if item:
            # Must expunge both item and its stock to fully detach
            if item.stock:
                session.expunge(item.stock)
            session.expunge(item)
        return item


def get_or_create_stock_by_symbol(
    symbol: str,
    session: Optional[Session] = None,
    name: Optional[str] = None,
) -> Stock:
    """Get an existing stock by symbol or create a new one.

    If the stock doesn't exist, creates a minimal stock record.
    This is useful when adding stocks from external sources.

    Args:
        symbol: The stock symbol (will be uppercased).
        session: Optional existing session. If None, creates a new session context.
        name: Optional name for the stock if creating new.

    Returns:
        The existing or newly created Stock.
    """
    symbol = symbol.upper().strip()

    def _get_or_create(sess: Session) -> Stock:
        stock = sess.query(Stock).filter(Stock.symbol == symbol).first()
        if stock:
            return stock

        # Create minimal stock record
        stock = Stock(
            symbol=symbol,
            name=name or symbol,  # Use symbol as name if not provided
            is_active=True,
        )
        sess.add(stock)
        sess.flush()  # Get the ID without committing
        return stock

    if session is not None:
        return _get_or_create(session)

    with session_scope() as sess:
        stock = _get_or_create(sess)
        sess.expunge(stock)
        return stock


def add_to_watchlist(
    stock_id: Optional[int] = None,
    symbol: Optional[str] = None,
    alert_price_above: Optional[float] = None,
    alert_price_below: Optional[float] = None,
    notes: Optional[str] = None,
) -> WatchlistItem:
    """Add a stock to the watchlist.

    Either stock_id or symbol must be provided. If symbol is provided and
    the stock doesn't exist in the database, a new stock record will be created.

    Args:
        stock_id: The stock ID to add.
        symbol: The stock symbol to add (alternative to stock_id).
        alert_price_above: Optional price alert threshold (above).
        alert_price_below: Optional price alert threshold (below).
        notes: Optional notes for this watchlist item.

    Returns:
        The created WatchlistItem with stock relationship loaded.

    Raises:
        ValueError: If neither stock_id nor symbol is provided.
        WatchlistItemExistsError: If the stock is already in the watchlist.
        StockNotFoundError: If stock_id is provided but stock doesn't exist.
    """
    if stock_id is None and symbol is None:
        raise ValueError("Either stock_id or symbol must be provided")

    with session_scope() as session:
        # Resolve stock_id from symbol if needed
        if stock_id is None and symbol is not None:
            stock = get_or_create_stock_by_symbol(symbol, session=session)
            stock_id = stock.id
            actual_symbol = stock.symbol
        else:
            # Verify stock exists
            stock = session.query(Stock).filter(Stock.id == stock_id).first()
            if not stock:
                raise StockNotFoundError(stock_id, by_field="id")
            actual_symbol = stock.symbol

        # Check if stock is already in watchlist
        existing = (
            session.query(WatchlistItem)
            .filter(WatchlistItem.stock_id == stock_id)
            .first()
        )
        if existing:
            raise WatchlistItemExistsError(stock_id, actual_symbol)

        # Create watchlist item
        item = WatchlistItem(
            stock_id=stock_id,
            alert_price_above=Decimal(str(alert_price_above)) if alert_price_above else None,
            alert_price_below=Decimal(str(alert_price_below)) if alert_price_below else None,
            notes=notes,
        )
        session.add(item)

        try:
            session.flush()
        except IntegrityError:
            # Handle race condition where stock was added between check and insert
            session.rollback()
            raise WatchlistItemExistsError(stock_id, actual_symbol)

        # Reload with stock relationship
        session.refresh(item)
        # Eagerly load stock and expunge both
        if item.stock:
            session.expunge(item.stock)
        session.expunge(item)
        return item


def update_watchlist_item(
    item_id: int,
    alert_price_above: Optional[float] = None,
    alert_price_below: Optional[float] = None,
    notes: Optional[str] = None,
    clear_alert_above: bool = False,
    clear_alert_below: bool = False,
    clear_notes: bool = False,
) -> WatchlistItem:
    """Update a watchlist item.

    Supports partial updates - only provided fields are updated.
    To clear a field, set the corresponding clear_* flag to True.

    Args:
        item_id: The watchlist item ID to update.
        alert_price_above: New price alert threshold (above).
        alert_price_below: New price alert threshold (below).
        notes: New notes.
        clear_alert_above: If True, clear the alert_price_above field.
        clear_alert_below: If True, clear the alert_price_below field.
        clear_notes: If True, clear the notes field.

    Returns:
        The updated WatchlistItem with stock relationship loaded.

    Raises:
        WatchlistItemNotFoundError: If the watchlist item doesn't exist.
    """
    with session_scope() as session:
        item = (
            session.query(WatchlistItem)
            .options(joinedload(WatchlistItem.stock))
            .filter(WatchlistItem.id == item_id)
            .first()
        )

        if not item:
            raise WatchlistItemNotFoundError(item_id)

        # Update fields
        if clear_alert_above:
            item.alert_price_above = None
        elif alert_price_above is not None:
            item.alert_price_above = Decimal(str(alert_price_above))

        if clear_alert_below:
            item.alert_price_below = None
        elif alert_price_below is not None:
            item.alert_price_below = Decimal(str(alert_price_below))

        if clear_notes:
            item.notes = None
        elif notes is not None:
            item.notes = notes

        session.flush()
        session.refresh(item)
        # Eagerly load stock and expunge both
        if item.stock:
            session.expunge(item.stock)
        session.expunge(item)
        return item


def remove_from_watchlist(item_id: int) -> WatchlistItem:
    """Remove a stock from the watchlist by watchlist item ID.

    Args:
        item_id: The watchlist item ID to remove.

    Returns:
        The removed WatchlistItem (for confirmation/response).

    Raises:
        WatchlistItemNotFoundError: If the watchlist item doesn't exist.
    """
    with session_scope() as session:
        item = (
            session.query(WatchlistItem)
            .options(joinedload(WatchlistItem.stock))
            .filter(WatchlistItem.id == item_id)
            .first()
        )

        if not item:
            raise WatchlistItemNotFoundError(item_id)

        # Make a copy of the data before deletion
        # We need to expunge both item and stock first to detach, then delete the original
        if item.stock:
            session.expunge(item.stock)
        session.expunge(item)

        # Re-fetch and delete
        item_to_delete = (
            session.query(WatchlistItem)
            .filter(WatchlistItem.id == item_id)
            .first()
        )
        session.delete(item_to_delete)

        return item


def remove_from_watchlist_by_symbol(symbol: str) -> WatchlistItem:
    """Remove a stock from the watchlist by stock symbol.

    Args:
        symbol: The stock symbol (case-insensitive).

    Returns:
        The removed WatchlistItem (for confirmation/response).

    Raises:
        WatchlistItemNotFoundError: If the stock is not in the watchlist.
    """
    symbol = symbol.upper().strip()

    with session_scope() as session:
        item = (
            session.query(WatchlistItem)
            .options(joinedload(WatchlistItem.stock))
            .join(Stock)
            .filter(Stock.symbol == symbol)
            .first()
        )

        if not item:
            raise WatchlistItemNotFoundError(symbol, by_field="symbol")

        # Make a copy of the data before deletion
        # We need to expunge both item and stock to fully detach
        if item.stock:
            session.expunge(item.stock)
        session.expunge(item)

        # Re-fetch and delete
        item_to_delete = (
            session.query(WatchlistItem)
            .join(Stock)
            .filter(Stock.symbol == symbol)
            .first()
        )
        session.delete(item_to_delete)

        return item
