"""Unit Tests for Watchlist Service.

Tests for all CRUD operations in the watchlist service module.
Uses mocked database session to test edge cases and error handling.
"""

import os
import tempfile
import uuid
from datetime import datetime
from decimal import Decimal

import pytest

from stockai.data.database import DatabaseManager, init_database, session_scope
from stockai.data.models import Stock, WatchlistItem
from stockai.web.services.watchlist import (
    WatchlistItemExistsError,
    WatchlistItemNotFoundError,
    StockNotFoundError,
    get_watchlist_items,
    get_watchlist_item_by_id,
    get_watchlist_item_by_stock_id,
    get_watchlist_item_by_symbol,
    get_or_create_stock_by_symbol,
    add_to_watchlist,
    update_watchlist_item,
    remove_from_watchlist,
    remove_from_watchlist_by_symbol,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing.

    Uses a unique database file per test to ensure proper isolation.
    """
    # Import here to avoid circular imports
    from stockai.data import database as db_module
    from stockai import config as config_module

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use unique DB file per test invocation
        unique_id = uuid.uuid4().hex[:8]
        db_path = os.path.join(tmpdir, f"test_{unique_id}.db")
        os.environ["STOCKAI_DB_PATH"] = db_path

        # Clear settings cache so new env vars are picked up
        config_module.get_settings.cache_clear()

        # Clear singleton completely before each test
        if DatabaseManager._engine is not None:
            try:
                DatabaseManager._engine.dispose()
            except Exception:
                pass
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._SessionLocal = None
        db_module._db_manager = None

        # Initialize database
        init_database()

        yield tmpdir

        # Cleanup after test
        if DatabaseManager._engine is not None:
            try:
                DatabaseManager._engine.dispose()
            except Exception:
                pass
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._SessionLocal = None
        db_module._db_manager = None

        # Clear settings cache again
        config_module.get_settings.cache_clear()

        if "STOCKAI_DB_PATH" in os.environ:
            del os.environ["STOCKAI_DB_PATH"]


@pytest.fixture
def sample_stock(temp_db):
    """Create a sample stock for testing."""
    with session_scope() as session:
        stock = Stock(
            symbol="BBCA",
            name="Bank Central Asia Tbk",
            sector="Financial",
            is_active=True,
        )
        session.add(stock)
        session.flush()
        stock_id = stock.id

    # Re-fetch to get a detached instance
    with session_scope() as session:
        stock = session.query(Stock).filter(Stock.id == stock_id).first()
        session.expunge(stock)
        return stock


@pytest.fixture
def sample_watchlist_item(temp_db, sample_stock):
    """Create a sample watchlist item for testing."""
    item = add_to_watchlist(
        stock_id=sample_stock.id,
        alert_price_above=10000.0,
        alert_price_below=8000.0,
        notes="Test watchlist item",
    )
    return item


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_watchlist_item_exists_error(self):
        """Test WatchlistItemExistsError message format."""
        error = WatchlistItemExistsError(stock_id=1, symbol="BBCA")
        assert error.stock_id == 1
        assert error.symbol == "BBCA"
        assert "BBCA" in str(error)
        assert "id=1" in str(error)
        assert "already in the watchlist" in str(error)

    def test_watchlist_item_not_found_error_by_id(self):
        """Test WatchlistItemNotFoundError for ID lookup."""
        error = WatchlistItemNotFoundError(identifier=123, by_field="id")
        assert error.identifier == 123
        assert error.by_field == "id"
        assert "id=123" in str(error)
        assert "not found" in str(error)

    def test_watchlist_item_not_found_error_by_symbol(self):
        """Test WatchlistItemNotFoundError for symbol lookup."""
        error = WatchlistItemNotFoundError(identifier="TLKM", by_field="symbol")
        assert error.identifier == "TLKM"
        assert error.by_field == "symbol"
        assert "symbol=TLKM" in str(error)

    def test_stock_not_found_error(self):
        """Test StockNotFoundError message format."""
        error = StockNotFoundError(identifier=999, by_field="id")
        assert error.identifier == 999
        assert error.by_field == "id"
        assert "id=999" in str(error)
        assert "not found" in str(error)


class TestGetWatchlistItems:
    """Test get_watchlist_items function."""

    def test_get_empty_watchlist(self, temp_db):
        """Test getting items when watchlist is empty."""
        items = get_watchlist_items()
        assert items == []

    def test_get_single_item(self, sample_watchlist_item):
        """Test getting items when watchlist has one item."""
        items = get_watchlist_items()
        assert len(items) == 1
        assert items[0].id == sample_watchlist_item.id
        assert items[0].stock is not None
        assert items[0].stock.symbol == "BBCA"

    def test_get_multiple_items(self, temp_db):
        """Test getting multiple watchlist items."""
        # Create multiple stocks and watchlist items
        symbols = ["BBCA", "TLKM", "ASII"]
        for symbol in symbols:
            add_to_watchlist(symbol=symbol)

        items = get_watchlist_items()
        assert len(items) == 3

        # Check that stock relationships are loaded
        for item in items:
            assert item.stock is not None
            assert item.stock.symbol in symbols

    def test_items_ordered_by_created_at_desc(self, temp_db):
        """Test that items are ordered by created_at descending."""
        # Add items in order
        symbols = ["BBCA", "TLKM", "ASII"]
        for symbol in symbols:
            add_to_watchlist(symbol=symbol)

        items = get_watchlist_items()
        # Most recent should be first
        assert items[0].stock.symbol == "ASII"
        assert items[2].stock.symbol == "BBCA"


class TestGetWatchlistItemById:
    """Test get_watchlist_item_by_id function."""

    def test_get_existing_item(self, sample_watchlist_item):
        """Test getting an existing watchlist item by ID."""
        item = get_watchlist_item_by_id(sample_watchlist_item.id)
        assert item is not None
        assert item.id == sample_watchlist_item.id
        assert item.stock is not None
        assert item.stock.symbol == "BBCA"

    def test_get_non_existent_item(self, temp_db):
        """Test getting a non-existent watchlist item."""
        item = get_watchlist_item_by_id(99999)
        assert item is None

    def test_item_includes_stock_relationship(self, sample_watchlist_item):
        """Test that returned item includes stock relationship."""
        item = get_watchlist_item_by_id(sample_watchlist_item.id)
        assert item.stock is not None
        assert item.stock.symbol == "BBCA"
        assert item.stock.name == "Bank Central Asia Tbk"


class TestGetWatchlistItemByStockId:
    """Test get_watchlist_item_by_stock_id function."""

    def test_get_existing_item_by_stock_id(self, sample_watchlist_item, sample_stock):
        """Test getting watchlist item by stock ID."""
        item = get_watchlist_item_by_stock_id(sample_stock.id)
        assert item is not None
        assert item.stock_id == sample_stock.id
        assert item.stock.symbol == "BBCA"

    def test_get_non_existent_by_stock_id(self, temp_db):
        """Test getting non-existent item by stock ID."""
        item = get_watchlist_item_by_stock_id(99999)
        assert item is None

    def test_stock_not_in_watchlist(self, temp_db):
        """Test getting item for stock that exists but not in watchlist."""
        # Create stock but don't add to watchlist
        with session_scope() as session:
            stock = Stock(symbol="TLKM", name="Telkom Indonesia", is_active=True)
            session.add(stock)
            session.flush()
            stock_id = stock.id

        item = get_watchlist_item_by_stock_id(stock_id)
        assert item is None


class TestGetWatchlistItemBySymbol:
    """Test get_watchlist_item_by_symbol function."""

    def test_get_existing_item_by_symbol(self, sample_watchlist_item):
        """Test getting watchlist item by symbol."""
        item = get_watchlist_item_by_symbol("BBCA")
        assert item is not None
        assert item.stock.symbol == "BBCA"

    def test_get_item_by_symbol_case_insensitive(self, sample_watchlist_item):
        """Test that symbol lookup is case-insensitive."""
        item = get_watchlist_item_by_symbol("bbca")
        assert item is not None
        assert item.stock.symbol == "BBCA"

    def test_get_item_by_symbol_with_whitespace(self, sample_watchlist_item):
        """Test that symbol lookup strips whitespace."""
        item = get_watchlist_item_by_symbol("  BBCA  ")
        assert item is not None
        assert item.stock.symbol == "BBCA"

    def test_get_non_existent_by_symbol(self, temp_db):
        """Test getting non-existent item by symbol."""
        item = get_watchlist_item_by_symbol("NONEXISTENT")
        assert item is None


class TestGetOrCreateStockBySymbol:
    """Test get_or_create_stock_by_symbol function."""

    def test_get_existing_stock(self, sample_stock):
        """Test getting an existing stock."""
        stock = get_or_create_stock_by_symbol("BBCA")
        assert stock is not None
        assert stock.id == sample_stock.id
        assert stock.symbol == "BBCA"
        assert stock.name == "Bank Central Asia Tbk"

    def test_create_new_stock(self, temp_db):
        """Test creating a new stock when it doesn't exist."""
        stock = get_or_create_stock_by_symbol("NEWSTOCK")
        assert stock is not None
        assert stock.symbol == "NEWSTOCK"
        assert stock.is_active is True

    def test_create_stock_with_name(self, temp_db):
        """Test creating a new stock with custom name."""
        stock = get_or_create_stock_by_symbol("TLKM", name="Telkom Indonesia")
        assert stock.symbol == "TLKM"
        assert stock.name == "Telkom Indonesia"

    def test_create_stock_default_name(self, temp_db):
        """Test that default name is the symbol."""
        stock = get_or_create_stock_by_symbol("XYZ")
        assert stock.symbol == "XYZ"
        assert stock.name == "XYZ"

    def test_symbol_case_normalization(self, temp_db):
        """Test that symbol is uppercased."""
        stock = get_or_create_stock_by_symbol("lowercase")
        assert stock.symbol == "LOWERCASE"

    def test_symbol_whitespace_stripped(self, temp_db):
        """Test that symbol whitespace is stripped."""
        stock = get_or_create_stock_by_symbol("  TRIM  ")
        assert stock.symbol == "TRIM"


class TestAddToWatchlist:
    """Test add_to_watchlist function."""

    def test_add_by_stock_id(self, temp_db, sample_stock):
        """Test adding stock to watchlist by stock ID."""
        item = add_to_watchlist(stock_id=sample_stock.id)
        assert item is not None
        assert item.stock_id == sample_stock.id
        assert item.stock.symbol == "BBCA"

    def test_add_by_symbol_existing_stock(self, temp_db, sample_stock):
        """Test adding to watchlist by symbol when stock exists."""
        item = add_to_watchlist(symbol="BBCA")
        assert item is not None
        assert item.stock.symbol == "BBCA"

    def test_add_by_symbol_new_stock(self, temp_db):
        """Test adding to watchlist creates new stock if needed."""
        item = add_to_watchlist(symbol="NEWSTOCK")
        assert item is not None
        assert item.stock.symbol == "NEWSTOCK"

    def test_add_with_alert_prices(self, temp_db, sample_stock):
        """Test adding with alert price thresholds."""
        item = add_to_watchlist(
            stock_id=sample_stock.id,
            alert_price_above=10000.0,
            alert_price_below=8000.0,
        )
        assert item.alert_price_above == Decimal("10000.0")
        assert item.alert_price_below == Decimal("8000.0")

    def test_add_with_notes(self, temp_db, sample_stock):
        """Test adding with notes."""
        item = add_to_watchlist(
            stock_id=sample_stock.id,
            notes="Buy on dip",
        )
        assert item.notes == "Buy on dip"

    def test_add_duplicate_raises_error(self, sample_watchlist_item, sample_stock):
        """Test that adding duplicate stock raises WatchlistItemExistsError."""
        with pytest.raises(WatchlistItemExistsError) as exc_info:
            add_to_watchlist(stock_id=sample_stock.id)

        assert exc_info.value.stock_id == sample_stock.id
        assert exc_info.value.symbol == "BBCA"

    def test_add_duplicate_by_symbol_raises_error(self, sample_watchlist_item):
        """Test that adding duplicate by symbol raises error."""
        with pytest.raises(WatchlistItemExistsError):
            add_to_watchlist(symbol="BBCA")

    def test_add_without_stock_id_or_symbol_raises_error(self, temp_db):
        """Test that adding without stock_id or symbol raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            add_to_watchlist()

        assert "Either stock_id or symbol must be provided" in str(exc_info.value)

    def test_add_with_non_existent_stock_id_raises_error(self, temp_db):
        """Test that adding with non-existent stock_id raises StockNotFoundError."""
        with pytest.raises(StockNotFoundError) as exc_info:
            add_to_watchlist(stock_id=99999)

        assert exc_info.value.identifier == 99999
        assert exc_info.value.by_field == "id"

    def test_add_returns_item_with_stock_loaded(self, temp_db, sample_stock):
        """Test that returned item has stock relationship loaded."""
        item = add_to_watchlist(stock_id=sample_stock.id)
        assert item.stock is not None
        assert item.stock.symbol == "BBCA"

    def test_add_item_has_created_at(self, temp_db, sample_stock):
        """Test that created item has created_at timestamp."""
        item = add_to_watchlist(stock_id=sample_stock.id)
        assert item.created_at is not None


class TestUpdateWatchlistItem:
    """Test update_watchlist_item function."""

    def test_update_alert_price_above(self, sample_watchlist_item):
        """Test updating alert_price_above."""
        updated = update_watchlist_item(
            sample_watchlist_item.id,
            alert_price_above=12000.0,
        )
        assert updated.alert_price_above == Decimal("12000.0")
        # Other fields unchanged
        assert updated.alert_price_below == sample_watchlist_item.alert_price_below
        assert updated.notes == sample_watchlist_item.notes

    def test_update_alert_price_below(self, sample_watchlist_item):
        """Test updating alert_price_below."""
        updated = update_watchlist_item(
            sample_watchlist_item.id,
            alert_price_below=7000.0,
        )
        assert updated.alert_price_below == Decimal("7000.0")

    def test_update_notes(self, sample_watchlist_item):
        """Test updating notes."""
        updated = update_watchlist_item(
            sample_watchlist_item.id,
            notes="Updated notes",
        )
        assert updated.notes == "Updated notes"

    def test_update_multiple_fields(self, sample_watchlist_item):
        """Test updating multiple fields at once."""
        updated = update_watchlist_item(
            sample_watchlist_item.id,
            alert_price_above=15000.0,
            alert_price_below=5000.0,
            notes="All fields updated",
        )
        assert updated.alert_price_above == Decimal("15000.0")
        assert updated.alert_price_below == Decimal("5000.0")
        assert updated.notes == "All fields updated"

    def test_update_non_existent_item_raises_error(self, temp_db):
        """Test that updating non-existent item raises WatchlistItemNotFoundError."""
        with pytest.raises(WatchlistItemNotFoundError) as exc_info:
            update_watchlist_item(99999, notes="test")

        assert exc_info.value.identifier == 99999

    def test_clear_alert_above(self, sample_watchlist_item):
        """Test clearing alert_price_above."""
        updated = update_watchlist_item(
            sample_watchlist_item.id,
            clear_alert_above=True,
        )
        assert updated.alert_price_above is None

    def test_clear_alert_below(self, sample_watchlist_item):
        """Test clearing alert_price_below."""
        updated = update_watchlist_item(
            sample_watchlist_item.id,
            clear_alert_below=True,
        )
        assert updated.alert_price_below is None

    def test_clear_notes(self, sample_watchlist_item):
        """Test clearing notes."""
        updated = update_watchlist_item(
            sample_watchlist_item.id,
            clear_notes=True,
        )
        assert updated.notes is None

    def test_update_returns_item_with_stock_loaded(self, sample_watchlist_item):
        """Test that updated item has stock relationship loaded."""
        updated = update_watchlist_item(
            sample_watchlist_item.id,
            notes="test",
        )
        assert updated.stock is not None
        assert updated.stock.symbol == "BBCA"


class TestRemoveFromWatchlist:
    """Test remove_from_watchlist function."""

    def test_remove_existing_item(self, sample_watchlist_item):
        """Test removing an existing watchlist item."""
        removed = remove_from_watchlist(sample_watchlist_item.id)
        assert removed is not None
        assert removed.id == sample_watchlist_item.id

        # Verify it's actually removed
        item = get_watchlist_item_by_id(sample_watchlist_item.id)
        assert item is None

    def test_remove_non_existent_raises_error(self, temp_db):
        """Test that removing non-existent item raises WatchlistItemNotFoundError."""
        with pytest.raises(WatchlistItemNotFoundError) as exc_info:
            remove_from_watchlist(99999)

        assert exc_info.value.identifier == 99999

    def test_remove_returns_deleted_item_data(self, sample_watchlist_item):
        """Test that remove returns the deleted item's data."""
        removed = remove_from_watchlist(sample_watchlist_item.id)
        assert removed.stock.symbol == "BBCA"
        assert removed.notes == "Test watchlist item"


class TestRemoveFromWatchlistBySymbol:
    """Test remove_from_watchlist_by_symbol function."""

    def test_remove_by_symbol(self, sample_watchlist_item):
        """Test removing by stock symbol."""
        removed = remove_from_watchlist_by_symbol("BBCA")
        assert removed is not None
        assert removed.stock.symbol == "BBCA"

        # Verify it's actually removed
        item = get_watchlist_item_by_symbol("BBCA")
        assert item is None

    def test_remove_by_symbol_case_insensitive(self, sample_watchlist_item):
        """Test that symbol removal is case-insensitive."""
        removed = remove_from_watchlist_by_symbol("bbca")
        assert removed is not None
        assert removed.stock.symbol == "BBCA"

    def test_remove_by_symbol_strips_whitespace(self, sample_watchlist_item):
        """Test that symbol removal strips whitespace."""
        removed = remove_from_watchlist_by_symbol("  BBCA  ")
        assert removed is not None

    def test_remove_non_existent_symbol_raises_error(self, temp_db):
        """Test that removing non-existent symbol raises error."""
        with pytest.raises(WatchlistItemNotFoundError) as exc_info:
            remove_from_watchlist_by_symbol("NONEXISTENT")

        assert exc_info.value.identifier == "NONEXISTENT"
        assert exc_info.value.by_field == "symbol"

    def test_remove_stock_not_in_watchlist_raises_error(self, temp_db):
        """Test removing stock that exists but not in watchlist."""
        # Create stock without adding to watchlist
        with session_scope() as session:
            stock = Stock(symbol="TLKM", name="Telkom Indonesia", is_active=True)
            session.add(stock)

        with pytest.raises(WatchlistItemNotFoundError):
            remove_from_watchlist_by_symbol("TLKM")


class TestWatchlistIntegration:
    """Integration tests for complete watchlist workflows."""

    def test_full_crud_workflow(self, temp_db):
        """Test complete create-read-update-delete workflow."""
        # Create
        item = add_to_watchlist(
            symbol="BBRI",
            alert_price_above=5000.0,
            alert_price_below=4000.0,
            notes="Initial notes",
        )
        assert item is not None
        item_id = item.id

        # Read
        fetched = get_watchlist_item_by_id(item_id)
        assert fetched is not None
        assert fetched.stock.symbol == "BBRI"

        # Update
        updated = update_watchlist_item(
            item_id,
            alert_price_above=5500.0,
            notes="Updated notes",
        )
        assert updated.alert_price_above == Decimal("5500.0")
        assert updated.notes == "Updated notes"

        # Delete
        removed = remove_from_watchlist(item_id)
        assert removed is not None

        # Verify deleted
        final = get_watchlist_item_by_id(item_id)
        assert final is None

    def test_multiple_stocks_in_watchlist(self, temp_db):
        """Test managing multiple stocks in watchlist."""
        symbols = ["BBCA", "BBRI", "BMRI", "TLKM", "ASII"]

        # Add all stocks
        for symbol in symbols:
            add_to_watchlist(symbol=symbol)

        # Verify all added
        items = get_watchlist_items()
        assert len(items) == 5

        # Remove one
        remove_from_watchlist_by_symbol("TLKM")

        # Verify removal
        items = get_watchlist_items()
        assert len(items) == 4
        item_symbols = {item.stock.symbol for item in items}
        assert "TLKM" not in item_symbols

    def test_stock_relationship_preserved_after_detach(self, temp_db):
        """Test that stock relationship is accessible after session closes."""
        item = add_to_watchlist(symbol="BBCA", notes="Test")

        # Access stock outside of any session context
        assert item.stock.symbol == "BBCA"
        assert item.stock.name == "BBCA"  # Default name when created

    def test_decimal_precision_maintained(self, temp_db):
        """Test that decimal precision is maintained for alert prices."""
        item = add_to_watchlist(
            symbol="BBCA",
            alert_price_above=10000.50,
            alert_price_below=8000.25,
        )

        fetched = get_watchlist_item_by_id(item.id)
        assert fetched.alert_price_above == Decimal("10000.50")
        assert fetched.alert_price_below == Decimal("8000.25")
